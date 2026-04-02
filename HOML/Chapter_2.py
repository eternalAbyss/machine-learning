import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # End-to-End Machine learning Project
    ---

    ## Objective
    <ol>
    <li>Look at the big picture.</li>
    <li>Get the data.</li>
    <li>Explore and visualize the data to gain insights.</li>
    <li>Prepare the data for machine learning algorithms.</li>
    <li>Select a model and train it.</li>
    <li>Fine-tune your model.</li>
    <li>Present your solution.</li>
    <li>Launch, monitor, and maintain your system.</li>
    </ol>
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
