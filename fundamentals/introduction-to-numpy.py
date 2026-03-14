import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Datatype and Attributes
    Numpy's main datatype is ndarray which refers to N-Dimensional Arrays
    """)
    return


@app.cell
def _(np):
    a1 = np.array([1,2,3])
    a1
    return (a1,)


@app.cell
def _(a1):
    type(a1)
    return


@app.cell
def _(np):
    a2 = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

    a3 = np.array([
                   [[1,2,3],
                    [4,5,6],
                    [7,8,9]],
                   [[10,11,12],
                    [13,14,15],
                    [16,17,18]],
                   [[19,20,21],
                    [22,23,24],
                    [25,26,28]]
                  ])
    return a2, a3


@app.cell
def _(a2):
    a2.shape
    return


@app.cell
def _(a3):
    a3.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shape of the array
    The shape of the array defines the number of axes the array has.
    Shapes are quite useful while doing machine learning as the shape of the input dataframe should be equal to the output dataset.
    Arrays can be of any shape in Numpy
    """)
    return


@app.cell
def _(a1, a2, a3):
    a1.ndim, a2.ndim, a3.ndim
    return


@app.cell
def _(a1, a2, a3):
    a1.size, a2.size, a3.size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating a DataFrame from NumPy arrays
    """)
    return


@app.cell
def _(a2, pd):
    pd.DataFrame(a2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating arrays
    """)
    return


@app.cell
def _(np):
    ones = np.ones([3,3], dtype=int)
    ones
    return


@app.cell
def _(np):
    zeros = np.zeros(3, dtype=int)
    zeros
    return


@app.cell
def _(np):
    range_array = np.arange(0,3,0.5, dtype=float)
    range_array
    return


@app.cell
def _(np):
    random_array = np.random.randint(1,12,size=(3,5))
    random_array
    return (random_array,)


@app.cell
def _(random_array):
    random_array.size
    return


@app.cell
def _(np):
    random_array2 = np.random.random((3,5))
    random_array2.shape
    return


@app.cell
def _(np):
    random_array3 = np.random.rand(3,5)
    random_array3.shape
    return


@app.cell
def _(np):
    np.random.seed(seed=1)
    random_array4 = np.random.randint(10,size=(2,3))
    random_array4
    return (random_array4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Viewing Arrays and Matrices
    """)
    return


@app.cell
def _(np, random_array4):
    # Using np.unique() we can find unique numbers in an array
    np.unique(random_array4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manipulating and Comparing Arrays
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arithmetic
    """)
    return


@app.cell
def _(a1):
    a1
    return


@app.cell
def _(np):
    ones_1 = np.ones(3)
    return (ones_1,)


@app.cell
def _(a1, ones_1):
    a1 + ones_1
    return


@app.cell
def _(a1, ones_1):
    a1 - ones_1
    return


@app.cell
def _(a1, ones_1):
    a1 * ones_1
    return


@app.cell
def _(a2):
    a2
    return


@app.cell
def _(a1, a2):
    a1 * a2
    return


@app.cell
def _(np):
    a4 = np.ones((9,2),int)
    a4
    return


@app.cell
def _(np):
    a2_1 = np.ones((2, 3), int)
    a3_1 = np.ones((2, 3, 3), int)
    a3_1
    return a2_1, a3_1


@app.cell
def _(a2_1, a3_1):
    a2_1.reshape(2, 3, 1) * a3_1
    return


@app.cell
def _(a1, np):
    np.var(a1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance
    Measure of the average degree to which each number is different from the mean

    **Higher variance** -> Wider range of numbers

    **Lower variance** -> Smaller range of numbers

    ### Standard Deviation
    A Measure of how spread out a group of numbers is from the mean

    Standard Deviation = Square root of variance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demo of Standard Deviation and Variance
    """)
    return


@app.cell
def _(np):
    high_var_arr = np.array([1,100,200,300,400,5000,6000])
    low_var_arr = np.array([2,4,6,8,10,12])
    return high_var_arr, low_var_arr


@app.cell
def _(high_var_arr, low_var_arr, np):
    np.var(high_var_arr), np.var(low_var_arr)
    return


@app.cell
def _(high_var_arr, low_var_arr, np):
    np.std(high_var_arr), np.std(low_var_arr)
    return


@app.cell
def _(high_var_arr, low_var_arr, np):
    np.mean(high_var_arr), np.mean(low_var_arr)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(high_var_arr, plt):
    plt.hist(high_var_arr)
    plt.show()
    return


@app.cell
def _(low_var_arr, plt):
    plt.hist(low_var_arr)
    plt.show
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reshaping and Transposing
    """)
    return


@app.cell
def _(np):
    a2_2 = np.array([i for i in range(18)])
    a2_2
    return (a2_2,)


@app.cell
def _(a2_2):
    a2_2.reshape(3, 6)
    return


@app.cell
def _(np):
    a3_2 = list()
    temp = list()
    i = 1
    while i < 19:
        temp.append([i, i + 1, i + 2])
        i = i + 3
        if (i - 1) % 9 == 0:
            a3_2.append(temp)
            temp = list()
    a3_2 = np.array(a3_2)
    a3_2.shape
    return (a3_2,)


@app.cell
def _(a2_2, a3_2):
    a2_2.reshape(3, 3, 2) * a3_2.reshape(3, 3, 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Broadcast Rule**
    1. The dimensions should be equal
    2. One of them should be one
    """)
    return


@app.cell
def _(a2_2):
    a2_3 = a2_2.reshape(3, 6)
    return (a2_3,)


@app.cell
def _(a2_3):
    # Transpose = switches the axes
    a2_tran = a2_3.T
    (a2_3.shape, a2_tran.shape)
    return


@app.cell
def _(a2_3):
    a2_3.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dot Product
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)

    mat1 = np.random.randint(10, size=(5,3))
    mat2 = np.random.randint(10, size=(5,3))

    mat1, mat2
    return mat1, mat2


@app.cell
def _(mat1, mat2):
    # Element-wise multiplication
    mat1 * mat2
    return


@app.cell
def _(mat1, mat2, np):
    # Dot product
    np.dot(mat1, mat2.reshape(3,5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison Operators
    """)
    return


@app.cell
def _(a1, a2_3):
    (a1, a2_3)
    return


@app.cell
def _(a1, a2_3):
    a1.reshape(3, 1) > a2_3
    return


@app.cell
def _(a1):
    a1 > 1
    return


@app.cell
def _(a1):
    a1 < 3
    return


@app.cell
def _(a1):
    a1 == a1
    return


@app.cell
def _(a1, a2_3):
    a1.reshape(3, 1) == a2_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sorting arrays
    """)
    return


@app.cell
def _(np):
    random_arr = np.random.randint(10, size=(3,5))
    random_arr
    return (random_arr,)


@app.cell
def _(np, random_arr):
    np.sort(random_arr)
    return


@app.cell
def _(np, random_arr):
    np.argsort(random_arr)
    return


@app.cell
def _(np, random_arr):
    # Gives the maximum value along each column
    np.argmax(random_arr, axis=0)
    return


@app.cell
def _(np, random_arr):
    # Gives the maximum value along each row
    np.argmax(random_arr, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical Example : Numpy in action
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="./numpy-images/panda.png">
    """)
    return


@app.cell
def _():
    # Turn an image into a numpy array
    from matplotlib.image import imread

    panda = imread('./numpy-images/panda.png')
    return imread, panda


@app.cell
def _(panda):
    panda
    return


@app.cell
def _(panda):
    print(type(panda))
    return


@app.cell
def _(panda):
    panda.size, panda.shape, panda.ndim
    return


@app.cell
def _(panda):
    panda[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="./numpy-images/car-photo.png"/>
    """)
    return


@app.cell
def _(imread):
    car = imread("./numpy-images/car-photo.png")
    car[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="./numpy-images/dog-photo.png" />
    """)
    return


@app.cell
def _(imread):
    dog = imread('./numpy-images/dog-photo.png')
    dog[:5]
    return


if __name__ == "__main__":
    app.run()
