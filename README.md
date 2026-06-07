# machine-learning

A personal machine-learning learning repository — hands-on notebooks and small
projects covering the core Python data stack (NumPy, pandas, Matplotlib,
scikit-learn) alongside applied mini-projects.

Notebooks are authored as [Marimo](https://marimo.io) apps (plain `.py` files
that run as reactive notebooks), so they version-control cleanly and run as
scripts.

## Structure

- `HOML/` — Notebooks following the *Hands-On Machine Learning with
  Scikit-Learn, Keras & TensorFlow* book.
- `mini-projects/` — Self-contained applied projects (see below).
- `datasets/` — CSV files used across the notebooks:
  - `heart-disease.csv` — classification dataset.
  - `car-sales.csv`, `car-sales-extended.csv`,
    `car-sales-extended-missing-data.csv`, `car-sales-missing-data.csv` —
    regression / preprocessing datasets (including variants with missing data).

## Getting started

Install [Marimo](https://marimo.io) and the data stack:

```bash
pip install marimo pandas numpy matplotlib scikit-learn
```

Open a notebook in edit mode:

```bash
marimo edit "mini-projects/Apple health explorer.py"
```

Or run it as a read-only app:

```bash
marimo run "mini-projects/Apple health explorer.py"
```

## Mini-projects

### 🍎 Apple Health Explorer

A Marimo notebook that parses, cleans, analyses, and generates AI insights from
an Apple Health export — entirely on your machine.

It walks through an 11-step flow: parse `export.xml` → clean & deduplicate →
metric overview → explore any metric → sleep, heart rate, HRV, activity/energy,
and workouts → cross-metric correlations → AI-generated health insights (via the
Anthropic API).

**Setup:** copy `mini-projects/example.env` to `.env` and fill in your values:

```bash
cp mini-projects/example.env mini-projects/.env
```

| Variable | Purpose |
| --- | --- |
| `HEALTH_EXPORT_XML` | Path to your unzipped Apple Health `export.xml`. |
| `HEALTH_OUTPUT_DIR` | Where cleaned files are written. |
| `TIMEZONE` | Local timezone for timestamp conversion. |
| `ANTHROPIC_API_KEY` | Anthropic API key for the AI insights step ([console](https://console.anthropic.com)). |
| `ANTHROPIC_MODEL` | Model used for AI insights. |

All health data stays local; only an anonymized summary is sent to the Anthropic
API when you run the insights step.

## Notes

`.env` files, `__marimo__/` session caches, and other local artifacts are
git-ignored — keep secrets and personal exports out of version control.
