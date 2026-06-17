import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ML imports
    import matplotlib.pyplot as plt
    import urllib.request
    import pandas as pd
    import numpy as np
    import tarfile

    # System imports
    from pathlib import Path

    return Path, np, pd, plt, tarfile, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **RMSE** $\mapsto$ Euclidean norm, called $l_2$ and noted as ${\parallel \cdot \parallel}_2$

    **MAE** $\mapsto$ Manhattan norm, called $l_1$ and noted as ${\parallel \cdot \parallel}_1$

    $l_k$ norm of a vector $v$ containing $n$ elements is defined as $\parallel V \parallel_k = {({|v_1|}^k + {|v_2|}^k + \ldots + {|v_n|}^k)} ^ \frac{1}{k} \cdot l_0$

    - The higher the norm index, the more it focuses on learge values and neglects small ones.
    - RMSE : more sensitive to outliers than MAE, performs well when outliers are exponentially rare (bell-curve)
    """)
    return


@app.cell
def _(Path, pd, tarfile, urllib):
    def load_housing_data():
        tarball_path = Path("datasets/housing.tgz")
        if not tarball_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))

    return (load_housing_data,)


@app.cell
def _(load_housing_data):
    housing = load_housing_data()
    return (housing,)


@app.cell
def _(housing):
    housing.head()
    return


@app.cell
def _(housing):
    housing.info()
    return


@app.cell
def _(housing):
    housing['ocean_proximity'].value_counts()
    return


@app.cell
def _(housing):
    housing.describe()
    return


@app.cell
def _(housing, plt):
    housing.hist(bins=50, figsize=(18,10))
    plt.show()
    return


@app.cell
def _(np):
    def shuffle_and_split_data(data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    return (shuffle_and_split_data,)


@app.cell
def _(housing, shuffle_and_split_data):
    train_set, test_set = shuffle_and_split_data(housing, 0.2)
    return test_set, train_set


@app.cell
def _(test_set, train_set):
    print("Length of the train set :", len(train_set))
    print("Length of the test set :", len(test_set))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
