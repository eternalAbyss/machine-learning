import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sample Notebook to demonstrate the use of notebook
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return pd, plt


@app.cell
def _(plt):
    x = [1,2,3,4,5,6]
    y = [1,2,3,10,5,6]
    plt.plot(x,y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heart disease project
    This Notebook is about the heart disease project
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('./../datasets/heart-disease.csv')
    df.head()
    return (df,)


@app.cell
def _(df):
    # df.describe()
    df.info()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df.sum()
    return


@app.cell
def _(df):
    df.mean()
    return


@app.cell
def _(df):
    df.head()
    return


if __name__ == "__main__":
    app.run()
