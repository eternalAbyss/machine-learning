import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.ui.slider(0,10,1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
