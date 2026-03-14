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
    # NumPy Practice

    This notebook offers a set of exercises for different tasks with NumPy.

    It should be noted there may be more than one different way to answer a question or complete an exercise.

    Exercises are based off (and directly taken from) the quick introduction to NumPy notebook.

    Different tasks will be detailed by comments or text.

    For further reference and resources, it's advised to check out the [NumPy documentation](https://numpy.org/devdocs/user/index.html).

    And if you get stuck, try searching for a question in the following format: "how to do XYZ with numpy", where XYZ is the function you want to leverage from NumPy.
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    # Create a 1-dimensional NumPy array using np.array()
    arr_1d = np.array(range(10))

    # Create a 2-dimensional NumPy array using np.array()
    arr_2d = np.array([range(10), range(10)])

    # Create a 3-dimensional Numpy array using np.array()
    return (arr_2d,)


app._unparsable_cell(
    r"""
    for 

    arr_2d_try = np.array(
        [range(10),
        range(10)]
    )
    """,
    name="_"
)


@app.cell
def _(arr_2d):
    print(arr_2d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we've you've created 3 different arrays, let's find details about them.

    Find the shape, number of dimensions, data type, size and type of each array.
    """)
    return


@app.cell
def _():
    # Attributes of 1-dimensional array (shape, 
    # number of dimensions, data type, size and type)
    return


@app.cell
def _():
    # Attributes of 2-dimensional array
    return


@app.cell
def _():
    # Attributes of 3-dimensional array
    return


@app.cell
def _():
    # Import pandas and create a DataFrame out of one
    # of the arrays you've created
    return


@app.cell
def _():
    # Create an array of shape (10, 2) with only ones
    return


@app.cell
def _():
    # Create an array of shape (7, 2, 3) of only zeros
    return


@app.cell
def _():
    # Create an array within a range of 0 and 100 with step 3
    return


@app.cell
def _():
    # Create a random array with numbers between 0 and 10 of size (7, 2)
    return


@app.cell
def _():
    # Create a random array of floats between 0 & 1 of shape (3, 5)
    return


@app.cell
def _():
    # Set the random seed to 42


    # Create a random array of numbers between 0 & 10 of size (4, 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell above again, what happens?

    Are the numbers in the array different or the same? Why do think this is?
    """)
    return


@app.cell
def _():
    # Create an array of random numbers between 1 & 10 of size (3, 7)
    # and save it to a variable


    # Find the unique numbers in the array you just created
    return


@app.cell
def _():
    # Find the 0'th index of the latest array you created
    return


@app.cell
def _():
    # Get the first 2 rows of latest array you created
    return


@app.cell
def _():
    # Get the first 2 values of the first 2 rows of the latest array
    return


@app.cell
def _():
    # Create a random array of numbers between 0 & 10 and an array of ones
    # both of size (3, 5), save them both to variables
    return


@app.cell
def _():
    # Add the two arrays together
    return


@app.cell
def _():
    # Create another array of ones of shape (5, 3)
    return


@app.cell
def _():
    # Try add the array of ones and the other most recent array together
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you try the last cell, it produces an error. Why do think this is?

    How would you fix it?
    """)
    return


@app.cell
def _():
    # Create another array of ones of shape (3, 5)
    return


@app.cell
def _():
    # Subtract the new array of ones from the other most recent array
    return


@app.cell
def _():
    # Multiply the ones array with the latest array
    return


@app.cell
def _():
    # Take the latest array to the power of 2 using '**'
    return


@app.cell
def _():
    # Do the same thing with np.square()
    return


@app.cell
def _():
    # Find the mean of the latest array using np.mean()
    return


@app.cell
def _():
    # Find the maximum of the latest array using np.max()
    return


@app.cell
def _():
    # Find the minimum of the latest array using np.min()
    return


@app.cell
def _():
    # Find the standard deviation of the latest array
    return


@app.cell
def _():
    # Find the variance of the latest array
    return


@app.cell
def _():
    # Reshape the latest array to (3, 5, 1)
    return


@app.cell
def _():
    # Transpose the latest array
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What does the transpose do?
    """)
    return


@app.cell
def _():
    # Create two arrays of random integers between 0 to 10
    # one of size (3, 3) the other of size (3, 2)
    return


@app.cell
def _():
    # Perform a dot product on the two newest arrays you created
    return


@app.cell
def _():
    # Create two arrays of random integers between 0 to 10
    # both of size (4, 3)
    return


@app.cell
def _():
    # Perform a dot product on the two newest arrays you created
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It doesn't work. How would you fix it?
    """)
    return


@app.cell
def _():
    # Take the latest two arrays, perform a transpose on one of them and then perform 
    # a dot product on them both
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how performing a transpose allows the dot product to happen.

    Why is this?

    Checking out the documentation on [`np.dot()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) may help, as well as reading [Math is Fun's guide on the dot product](https://www.mathsisfun.com/algebra/vectors-dot-product.html).

    Let's now compare arrays.
    """)
    return


@app.cell
def _():
    # Create two arrays of random integers between 0 & 10 of the same shape
    # and save them to variables
    return


@app.cell
def _():
    # Compare the two arrays with '>'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What happens when you compare the arrays with `>`?
    """)
    return


@app.cell
def _():
    # Compare the two arrays with '>='
    return


@app.cell
def _():
    # Find which elements of the first array are greater than 7
    return


@app.cell
def _():
    # Which parts of each array are equal? (try using '==')
    return


@app.cell
def _():
    # Sort one of the arrays you just created in ascending order
    return


@app.cell
def _():
    # Sort the indexes of one of the arrays you just created
    return


@app.cell
def _():
    # Find the index with the maximum value in one of the arrays you've created
    return


@app.cell
def _():
    # Find the index with the minimum value in one of the arrays you've created
    return


@app.cell
def _():
    # Find the indexes with the maximum values down the 1st axis (axis=1)
    # of one of the arrays you created
    return


@app.cell
def _():
    # Find the indexes with the minimum values across the 0th axis (axis=0)
    # of one of the arrays you created
    return


@app.cell
def _():
    # Create an array of normally distributed random numbers
    return


@app.cell
def _():
    # Create an array with 10 evenly spaced numbers between 1 and 100
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extensions

    For more exercises, check out the [NumPy quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html). A good practice would be to read through it and for the parts you find interesting, add them into the end of this notebook.

    Pay particular attention to the section on broadcasting. And most importantly, get hands-on with the code as much as possible. If in dobut, run the code, see what it does.

    The next place you could go is the [Stack Overflow page for the top questions and answers for NumPy](https://stackoverflow.com/questions/tagged/numpy?sort=MostVotes&edited=true). Often, you'll find some of the most common and useful NumPy functions here. Don't forget to play around with the filters! You'll likely find something helpful here.

    Finally, as always, remember, the best way to learn something new is to try it. And try it relentlessly. If you get interested in some kind of NumPy function, asking yourself, "I wonder if NumPy could do that?", go and find out.
    """)
    return


if __name__ == "__main__":
    app.run()
