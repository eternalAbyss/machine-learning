import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", app_title="Apple Health Explorer")


@app.cell
def _():
    import marimo as mo
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import json
    import os
    import urllib.request
    import warnings
    from datetime import datetime
    warnings.filterwarnings("ignore")

    # ── Load .env manually (no python-dotenv needed) ──────────────────────────
    def load_env(env_path: str = ".env") -> dict:
        """Parse a .env file and return a dict. Also sets os.environ."""
        env = {}
        _p = Path(env_path).expanduser()
        if not _p.exists():
            return env
        for line in _p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            env[key.strip()] = val
            os.environ.setdefault(key.strip(), val)
        return env

    ENV = load_env(".env")

    EXPORT_XML     = Path(os.environ.get("HEALTH_EXPORT_XML", "~/Downloads/apple_health_export/export.xml")).expanduser()
    OUTPUT_DIR     = Path(os.environ.get("HEALTH_OUTPUT_DIR", "~/Downloads/apple_health_export")).expanduser()
    TIMEZONE       = os.environ.get("TIMEZONE", "Asia/Kolkata")
    ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL= os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    print(f"export.xml  : {EXPORT_XML}")
    print(f"output dir  : {OUTPUT_DIR}")
    print(f"timezone    : {TIMEZONE}")
    print(f"API key set : {'yes' if ANTHROPIC_KEY else 'no'}")
    return (
        ANTHROPIC_KEY,
        ANTHROPIC_MODEL,
        ET,
        EXPORT_XML,
        OUTPUT_DIR,
        TIMEZONE,
        defaultdict,
        json,
        mdates,
        mo,
        pd,
        plt,
        urllib,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🍎 Apple Health Explorer
    **Parse → Clean → Analyse → AI Insights**

    Config loaded from `.env`. All data stays local.
    """)
    return


@app.cell
def _(EXPORT_XML, OUTPUT_DIR, mo):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPORT_XML.exists():
        mo.stop(True, mo.callout(mo.md(
            f"❌ `export.xml` not found at `{EXPORT_XML}`\n\n"
            "Set `HEALTH_EXPORT_XML` in your `.env` file."
        ), kind="danger"))

    mo.callout(mo.md(
        f"✅ `{EXPORT_XML.name}` — **{EXPORT_XML.stat().st_size / 1_048_576:.1f} MB**\n\n"
        f"📁 Output dir: `{OUTPUT_DIR}`"
    ), kind="success")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📥 Step 1 · Parse
    """)
    return


@app.cell
def _(ET, EXPORT_XML, defaultdict, mo, pd):
    with mo.status.spinner(title="Parsing XML…"):
        tree = ET.parse(EXPORT_XML)
        root = tree.getroot()

    _buckets = defaultdict(list)
    for _rec in root.iter("Record"):
        _a = _rec.attrib
        _buckets[_a.get("type", "Unknown")].append({
            "type":         _a.get("type"),
            "sourceName":   _a.get("sourceName"),
            "unit":         _a.get("unit"),
            "value":        _a.get("value"),
            "startDate":    _a.get("startDate"),
            "endDate":      _a.get("endDate"),
            "creationDate": _a.get("creationDate"),
        })

    raw_df = pd.concat(
        [pd.DataFrame(rows) for rows in _buckets.values()],
        ignore_index=True,
    )

    # Parse workouts
    _workout_rows = []
    for _w in root.iter("Workout"):
        _a = _w.attrib
        _workout_rows.append({
            "workoutType":      _a.get("workoutActivityType", "").replace("HKWorkoutActivityType", ""),
            "duration_min":     float(_a.get("duration", 0) or 0),
            "totalDistance_km": float(_a.get("totalDistance", 0) or 0),
            "totalEnergy_kcal": float(_a.get("totalEnergyBurned", 0) or 0),
            "startDate":        _a.get("startDate"),
            "endDate":          _a.get("endDate"),
        })
    workouts_raw = pd.DataFrame(_workout_rows) if _workout_rows else pd.DataFrame()

    print(f"Raw records  : {len(raw_df):,}")
    print(f"Workout rows : {len(workouts_raw):,}")
    return raw_df, workouts_raw


@app.cell
def _(mo):
    mo.md("""
    ## 🧹 Step 2 · Clean
    """)
    return


@app.cell
def _(TIMEZONE, mo, pd, raw_df, workouts_raw):
    with mo.status.spinner(title="Cleaning…"):

        df = raw_df.copy()

        # Dates
        for _col in ["startDate", "endDate", "creationDate"]:
            df[_col] = pd.to_datetime(df[_col], utc=True, errors="coerce").dt.tz_convert(TIMEZONE)

        # Numeric value
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")

        # Clean metric name — strip all HK prefixes
        df["metric"] = (
            df["type"]
            .str.replace(r"HKQuantityTypeIdentifier", "", regex=True)
            .str.replace(r"HKCategoryTypeIdentifier", "", regex=True)
            .str.replace(r"HKDataType", "", regex=True)
        )

        # Drop rows with neither numeric nor category value
        df = df[df["value_num"].notna() | df["value"].notna()]

        # Duration sanity
        df["duration_sec"] = (df["endDate"] - df["startDate"]).dt.total_seconds()
        df = df[df["duration_sec"] >= 0]

        # Deduplication
        df = df.drop_duplicates(subset=["type", "startDate", "endDate", "value"])

        # Physiological outlier rules
        _outlier_rules = {
            "HeartRate":                    (20,  250),
            "HeartRateVariabilitySDNN":     (1,   300),
            "StepCount":                    (0,   100_000),
            "BodyMass":                     (20,  300),
            "BloodOxygen":                  (70,  100),
            "RespiratoryRate":              (4,   60),
            "VO2Max":                       (10,  90),
            "ActiveEnergyBurned":           (0,   5_000),
            "BasalEnergyBurned":            (500, 4_000),
            "BodyFatPercentage":            (3,   60),
            "BodyMassIndex":                (10,  60),
            "BloodPressureSystolic":        (60,  250),
            "BloodPressureDiastolic":       (40,  150),
            "DietaryEnergyConsumed":        (0,   10_000),
            "BloodGlucose":                 (2,   30),
        }
        for _metric, (_lo, _hi) in _outlier_rules.items():
            _mask = (df["metric"] == _metric) & df["value_num"].notna()
            df = df[~(_mask & ((df["value_num"] < _lo) | (df["value_num"] > _hi)))]

        # Clean workouts
        workouts = pd.DataFrame()
        if not workouts_raw.empty:
            workouts = workouts_raw.copy()
            for _col in ["startDate", "endDate"]:
                workouts[_col] = pd.to_datetime(workouts[_col], utc=True, errors="coerce").dt.tz_convert(TIMEZONE)
            workouts = workouts[workouts["duration_min"] > 0]
            workouts = workouts.drop_duplicates(subset=["workoutType", "startDate"])

    _removed = len(raw_df) - len(df)
    print(f"Clean : {len(df):,}  |  Removed : {_removed:,} ({_removed / len(raw_df) * 100:.1f}%)")
    return df, workouts


@app.cell
def _(df, mo, raw_df):
    _removed = len(raw_df) - len(df)
    mo.callout(mo.md(
        f"**Cleaning summary**\n\n"
        f"- Raw: **{len(raw_df):,}** records\n"
        f"- Removed (dupes + outliers + bad dates): **{_removed:,}**\n"
        f"- Clean: **{len(df):,}** records"
    ), kind="success")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Step 3 · Metric Overview
    """)
    return


@app.cell
def _(df, mo):
    _summary = (
        df.groupby("metric")
        .agg(
            records=("value_num", "count"),
            unit=("unit", "first"),
            earliest=("startDate", "min"),
            latest=("startDate", "max"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )
    _summary["earliest"] = _summary["earliest"].dt.strftime("%Y-%m-%d")
    _summary["latest"]   = _summary["latest"].dt.strftime("%Y-%m-%d")
    mo.ui.table(_summary, label="All metrics — cleaned", selection=None)
    return


@app.cell
def _(OUTPUT_DIR, df, mo, workouts):
    _records_path  = OUTPUT_DIR / "health_records_clean.parquet"
    _workouts_path = OUTPUT_DIR / "health_workouts_clean.parquet"

    df.to_parquet(_records_path, index=False)
    if not workouts.empty:
        workouts.to_parquet(_workouts_path, index=False)

    mo.callout(mo.md(
        f"💾 **Saved to disk**\n\n"
        f"- `{_records_path}`\n"
        f"- `{_workouts_path}`"
    ), kind="info")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📈 Step 4 · Explore Any Metric
    """)
    return


@app.cell
def _(df, mo):
    _metrics = sorted(df["metric"].unique().tolist())
    metric_picker = mo.ui.dropdown(
        options=_metrics,
        value="StepCount" if "StepCount" in _metrics else _metrics[0],
        label="Metric",
    )
    agg_picker = mo.ui.dropdown(
        options=["sum", "mean", "max", "min"],
        value="sum",
        label="Daily aggregation",
    )
    mo.hstack([metric_picker, agg_picker])
    return agg_picker, metric_picker


@app.cell
def _(agg_picker, df, mdates, metric_picker, pd, plt):
    _sel = df[df["metric"] == metric_picker.value].dropna(subset=["value_num"]).copy()
    _sel["date"] = _sel["startDate"].dt.date
    _daily = _sel.groupby("date")["value_num"].agg(agg_picker.value).reset_index()
    _daily["date"] = pd.to_datetime(_daily["date"])
    _roll  = _daily["value_num"].rolling(7, min_periods=1).mean()
    _unit  = _sel["unit"].iloc[0] if not _sel.empty else ""

    fig_metric, ax_metric = plt.subplots(figsize=(13, 4))
    ax_metric.bar(_daily["date"], _daily["value_num"], width=0.8, color="#4F8EF7", alpha=0.4, label="Daily")
    ax_metric.plot(_daily["date"], _roll, color="#1A5FD4", linewidth=1.8, label="7-day avg")
    ax_metric.set_title(f"{metric_picker.value}  ·  {agg_picker.value}  ({_unit})", fontsize=13, pad=10)
    ax_metric.set_ylabel(_unit)
    ax_metric.legend()
    ax_metric.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_metric.xaxis.set_major_locator(mdates.MonthLocator())
    ax_metric.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_metric
    return


@app.cell
def _(mo):
    mo.md("""
    ## 😴 Step 5 · Sleep
    """)
    return


@app.cell
def _(df, mdates, mo, pd, plt):
    _sleep = df[df["type"].str.contains("SleepAnalysis", na=False)].copy()
    if _sleep.empty:
        mo.stop(True, mo.callout(mo.md("No sleep data found."), kind="warn"))

    _sleep["date"]       = _sleep["startDate"].dt.date
    _sleep["duration_h"] = (_sleep["endDate"] - _sleep["startDate"]).dt.total_seconds() / 3600
    _sleep = _sleep[_sleep["duration_h"].between(0, 18)]

    _daily_sleep = _sleep.groupby("date")["duration_h"].sum().reset_index()
    _daily_sleep = _daily_sleep[_daily_sleep["duration_h"] < 18]
    _daily_sleep["date"] = pd.to_datetime(_daily_sleep["date"])
    _roll_sleep  = _daily_sleep["duration_h"].rolling(7, min_periods=1).mean()
    _avg_sleep   = _daily_sleep["duration_h"].mean()

    fig_sleep, ax_sleep = plt.subplots(figsize=(13, 4))
    ax_sleep.bar(_daily_sleep["date"], _daily_sleep["duration_h"], width=0.8, color="#7C5CBF", alpha=0.5)
    ax_sleep.plot(_daily_sleep["date"], _roll_sleep, color="#4A1F8C", linewidth=1.8, label="7-day avg")
    ax_sleep.axhline(8,         color="#aaa",     linestyle="--", linewidth=0.9, label="8 h target")
    ax_sleep.axhline(_avg_sleep, color="#E05C5C", linestyle=":",  linewidth=1.1, label=f"Your avg {_avg_sleep:.1f} h")
    ax_sleep.set_title("Sleep duration per night", fontsize=13, pad=10)
    ax_sleep.set_ylabel("Hours")
    ax_sleep.legend()
    ax_sleep.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_sleep.xaxis.set_major_locator(mdates.MonthLocator())
    ax_sleep.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_sleep
    return


@app.cell
def _(mo):
    mo.md("""
    ## ❤️ Step 6 · Heart Rate
    """)
    return


@app.cell
def _(df, mdates, mo, pd, plt):
    _hr = df[(df["metric"] == "HeartRate") & df["value_num"].between(30, 200)].copy()
    if _hr.empty:
        mo.stop(True, mo.callout(mo.md("No heart rate data found."), kind="warn"))

    _hr["date"] = _hr["startDate"].dt.date
    _daily_hr   = _hr.groupby("date")["value_num"].agg(["mean", "min", "max"]).reset_index()
    _daily_hr["date"] = pd.to_datetime(_daily_hr["date"])
    _roll_hr    = _daily_hr["mean"].rolling(7, min_periods=1).mean()

    fig_hr, ax_hr = plt.subplots(figsize=(13, 4))
    ax_hr.fill_between(_daily_hr["date"], _daily_hr["min"], _daily_hr["max"], alpha=0.12, color="#E05C5C", label="Daily range")
    ax_hr.scatter(_daily_hr["date"], _daily_hr["mean"], s=6, color="#E05C5C", alpha=0.6, label="Daily mean")
    ax_hr.plot(_daily_hr["date"], _roll_hr, color="#B01C1C", linewidth=1.8, label="7-day avg")
    ax_hr.set_title("Heart Rate (bpm)", fontsize=13, pad=10)
    ax_hr.set_ylabel("bpm")
    ax_hr.legend()
    ax_hr.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_hr.xaxis.set_major_locator(mdates.MonthLocator())
    ax_hr.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_hr
    return


@app.cell
def _(mo):
    mo.md("""
    ## 💚 Step 7 · Heart Rate Variability (HRV)
    """)
    return


@app.cell
def _(df, mdates, mo, pd, plt):
    _hrv = df[(df["metric"] == "HeartRateVariabilitySDNN") & df["value_num"].notna()].copy()
    if _hrv.empty:
        mo.stop(True, mo.callout(mo.md("No HRV data found."), kind="warn"))

    _hrv["date"]    = _hrv["startDate"].dt.date
    _daily_hrv      = _hrv.groupby("date")["value_num"].mean().reset_index()
    _daily_hrv["date"] = pd.to_datetime(_daily_hrv["date"])
    _roll_hrv       = _daily_hrv["value_num"].rolling(7, min_periods=1).mean()
    _avg_hrv        = _daily_hrv["value_num"].mean()

    fig_hrv, ax_hrv = plt.subplots(figsize=(13, 4))
    ax_hrv.scatter(_daily_hrv["date"], _daily_hrv["value_num"], s=8, color="#2ECC71", alpha=0.5, label="Daily")
    ax_hrv.plot(_daily_hrv["date"], _roll_hrv, color="#1A8A4A", linewidth=1.8, label="7-day avg")
    ax_hrv.axhline(_avg_hrv, color="#aaa", linestyle=":", linewidth=1, label=f"Overall avg {_avg_hrv:.0f} ms")
    ax_hrv.set_title("HRV — SDNN (ms)  ·  Higher = better recovery", fontsize=13, pad=10)
    ax_hrv.set_ylabel("ms")
    ax_hrv.legend()
    ax_hrv.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_hrv.xaxis.set_major_locator(mdates.MonthLocator())
    ax_hrv.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_hrv
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🏃 Step 8 · Activity & Energy
    """)
    return


@app.cell
def _(df, mdates, mo, pd, plt):
    _steps  = df[(df["metric"] == "StepCount")          & df["value_num"].notna()].copy()
    _energy = df[(df["metric"] == "ActiveEnergyBurned") & df["value_num"].notna()].copy()

    if _steps.empty and _energy.empty:
        mo.stop(True, mo.callout(mo.md("No activity data found."), kind="warn"))

    fig_act, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    if not _steps.empty:
        _steps["date"] = _steps["startDate"].dt.date
        _ds = _steps.groupby("date")["value_num"].sum().reset_index()
        _ds["date"] = pd.to_datetime(_ds["date"])
        _rs = _ds["value_num"].rolling(7, min_periods=1).mean()
        axes[0].bar(_ds["date"], _ds["value_num"], width=0.8, color="#F39C12", alpha=0.5)
        axes[0].plot(_ds["date"], _rs, color="#B7770D", linewidth=1.8, label="7-day avg")
        axes[0].axhline(10000, color="#aaa", linestyle="--", linewidth=0.8, label="10k target")
        axes[0].set_title("Daily Steps", fontsize=12)
        axes[0].set_ylabel("Steps")
        axes[0].legend()
        axes[0].spines[["top", "right"]].set_visible(False)

    if not _energy.empty:
        _energy["date"] = _energy["startDate"].dt.date
        _de = _energy.groupby("date")["value_num"].sum().reset_index()
        _de["date"] = pd.to_datetime(_de["date"])
        _re = _de["value_num"].rolling(7, min_periods=1).mean()
        axes[1].bar(_de["date"], _de["value_num"], width=0.8, color="#E74C3C", alpha=0.5)
        axes[1].plot(_de["date"], _re, color="#922B21", linewidth=1.8, label="7-day avg")
        axes[1].set_title("Active Energy Burned (kcal)", fontsize=12)
        axes[1].set_ylabel("kcal")
        axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_act
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🏋️ Step 9 · Workouts
    """)
    return


@app.cell
def _(mo, plt, workouts):
    if workouts.empty:
        mo.stop(True, mo.callout(mo.md("No workout data found."), kind="warn"))

    _wt = (
        workouts.groupby("workoutType")
        .agg(sessions=("duration_min", "count"), total_min=("duration_min", "sum"), avg_kcal=("totalEnergy_kcal", "mean"))
        .reset_index()
        .sort_values("sessions", ascending=True)
    )

    fig_wk, ax_wk = plt.subplots(figsize=(10, max(3, len(_wt) * 0.45)))
    ax_wk.barh(_wt["workoutType"], _wt["sessions"], color="#3498DB", alpha=0.8)
    ax_wk.set_title("Workout sessions by type", fontsize=13, pad=10)
    ax_wk.set_xlabel("Sessions")
    ax_wk.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig_wk
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🔗 Step 10 · Correlations
    """)
    return


@app.cell
def _(df, mo, pd, plt):
    _key_metrics = [
        "StepCount", "HeartRate", "HeartRateVariabilitySDNN",
        "ActiveEnergyBurned", "BasalEnergyBurned", "RespiratoryRate", "VO2Max",
    ]
    _available = [m for m in _key_metrics if m in df["metric"].unique()]

    if len(_available) < 2:
        mo.stop(True, mo.callout(mo.md("Not enough metrics for correlation analysis."), kind="warn"))

    _parts = []
    for _m in _available:
        _tmp = df[df["metric"] == _m].copy()
        _tmp["date"] = _tmp["startDate"].dt.date
        _parts.append(_tmp.groupby("date")["value_num"].mean().rename(_m))

    _corr = pd.concat(_parts, axis=1).dropna(how="all").corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    _im = ax_corr.imshow(_corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(_im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_xticks(range(len(_corr.columns)))
    ax_corr.set_yticks(range(len(_corr.columns)))
    ax_corr.set_xticklabels(_corr.columns, rotation=45, ha="right", fontsize=9)
    ax_corr.set_yticklabels(_corr.columns, fontsize=9)
    for _i in range(len(_corr)):
        for _j in range(len(_corr.columns)):
            ax_corr.text(_j, _i, f"{_corr.iloc[_i, _j]:.2f}", ha="center", va="center", fontsize=8)
    ax_corr.set_title("Metric correlations (daily averages)", fontsize=12, pad=12)
    plt.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🤖 Step 11 · AI Health Insights
    """)
    return


@app.cell
def _(df, mo, pd, workouts):
    def build_health_summary(df, workouts):
        summary = {}

        def _stats(metric):
            _d = df[(df["metric"] == metric) & df["value_num"].notna()]
            if _d.empty:
                return None
            return {
                "mean":       round(float(_d["value_num"].mean()), 2),
                "median":     round(float(_d["value_num"].median()), 2),
                "min":        round(float(_d["value_num"].min()), 2),
                "max":        round(float(_d["value_num"].max()), 2),
                "unit":       _d["unit"].iloc[0],
                "n_readings": int(len(_d)),
            }

        for _m in [
            "HeartRate", "HeartRateVariabilitySDNN", "StepCount",
            "ActiveEnergyBurned", "BasalEnergyBurned", "RespiratoryRate",
            "VO2Max", "BodyMass", "BloodOxygen", "BodyFatPercentage",
            "BodyMassIndex", "RespiratoryRate", "BloodPressureSystolic",
            "BloodPressureDiastolic", "DietaryEnergyConsumed",
        ]:
            _s = _stats(_m)
            if _s:
                summary[_m] = _s

        # Sleep
        _sleep = df[df["type"].str.contains("SleepAnalysis", na=False)].copy()
        if not _sleep.empty:
            _sleep["duration_h"] = (_sleep["endDate"] - _sleep["startDate"]).dt.total_seconds() / 3600
            _sleep = _sleep[_sleep["duration_h"].between(0, 18)]
            _sleep["date"] = _sleep["startDate"].dt.date
            _ds = _sleep.groupby("date")["duration_h"].sum()
            _ds = _ds[_ds < 18]
            summary["Sleep"] = {
                "avg_hours":      round(float(_ds.mean()), 2),
                "median_hours":   round(float(_ds.median()), 2),
                "min_hours":      round(float(_ds.min()), 2),
                "max_hours":      round(float(_ds.max()), 2),
                "nights_tracked": int(len(_ds)),
            }

        # Steps — recent 30d vs overall
        if "StepCount" in summary:
            _steps = df[(df["metric"] == "StepCount") & df["value_num"].notna()].copy()
            _steps["date"] = _steps["startDate"].dt.date
            _ds2 = _steps.groupby("date")["value_num"].sum()
            _recent = _ds2[_ds2.index >= (pd.Timestamp.now().date() - pd.Timedelta(days=30))]
            summary["StepCount"]["recent_30d_avg"] = round(float(_recent.mean()), 0) if not _recent.empty else None

        # Workouts
        if not workouts.empty:
            summary["Workouts"] = {
                "total_sessions":  int(len(workouts)),
                "types":           workouts["workoutType"].value_counts().head(5).to_dict(),
                "avg_duration_min": round(float(workouts["duration_min"].mean()), 1),
                "avg_energy_kcal": round(float(workouts["totalEnergy_kcal"].mean()), 1),
            }

        summary["date_range"] = {
            "from": str(df["startDate"].min().date()),
            "to":   str(df["startDate"].max().date()),
        }

        return summary

    health_summary = build_health_summary(df, workouts)
    mo.callout(mo.md(
        f"✅ Health summary ready — **{len(health_summary)} categories** built"
    ), kind="success")
    return (health_summary,)


@app.cell
def _(mo):
    insight_topic = mo.ui.dropdown(
        options=[
            "Overall health overview",
            "Sleep quality & habits",
            "Cardiovascular health",
            "Activity & fitness trends",
            "Recovery & HRV",
            "What should I focus on improving?",
            "Custom question",
        ],
        value="Overall health overview",
        label="Insight topic",
    )
    custom_question = mo.ui.text(
        value="",
        label="Custom question (if selected above)",
        full_width=True,
    )
    run_insights = mo.ui.run_button(label="✨ Get AI Insights")
    mo.vstack([insight_topic, custom_question, run_insights])
    return custom_question, insight_topic, run_insights


@app.cell
def _(
    ANTHROPIC_KEY,
    ANTHROPIC_MODEL,
    custom_question,
    health_summary,
    insight_topic,
    json,
    mo,
    run_insights,
    urllib,
):
    mo.stop(not run_insights.value)

    if not ANTHROPIC_KEY:
        mo.stop(True, mo.callout(mo.md(
            "⚠️ `ANTHROPIC_API_KEY` not set.\n\nAdd it to your `.env` file and restart."
        ), kind="warn"))

    _topic = (
        custom_question.value.strip()
        if insight_topic.value == "Custom question" and custom_question.value.strip()
        else insight_topic.value
    )

    _system = """You are a knowledgeable, thoughtful health analyst.
    You are given a structured summary of a person's Apple Health data collected via Apple Watch.
    Provide clear, personalised, evidence-based insights — be specific and reference actual numbers.
    Be constructive: highlight what's going well, flag patterns worth watching, suggest 2–3 concrete actions.
    Do not diagnose. Do not be alarmist. Speak like a smart, informed friend.
    Format your response in clean markdown with clear sections."""

    _user = f"""Here is my Apple Health data summary:

    {json.dumps(health_summary, indent=2)}

    Please give me insights on: **{_topic}**

    Reference my actual numbers. Note trends, flag anything worth paying attention to, and suggest 2–3 concrete next steps."""

    with mo.status.spinner(title="Analysing your health data…"):
        _payload = json.dumps({
            "model":      ANTHROPIC_MODEL,
            "max_tokens": 1500,
            "system":     _system,
            "messages":   [{"role": "user", "content": _user}],
        }).encode()

        _req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=_payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(_req) as _resp:
                _data = json.loads(_resp.read())
                _insight_text = _data["content"][0]["text"]
        except Exception as _e:
            _insight_text = f"❌ API error: {_e}"

    mo.md(_insight_text)
    return


@app.cell
def _(health_summary, json, mo):
    mo.accordion({
        "📋 Raw health summary JSON (what gets sent to AI)": mo.md(
            f"```json\n{json.dumps(health_summary, indent=2)}\n```"
        )
    })
    return


if __name__ == "__main__":
    app.run()
