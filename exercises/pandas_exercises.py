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
    # Pandas Practice

    This notebook is dedicated to practicing different tasks with pandas. The solutions are available in a solutions notebook, however, you should always try to figure them out yourself first.

    It should be noted there may be more than one different way to answer a question or complete an exercise.

    Exercises are based off (and directly taken from) the quick introduction to pandas notebook.

    Different tasks will be detailed by comments or text.

    For further reference and resources, it's advised to check out the [pandas documnetation](https://pandas.pydata.org/pandas-docs/stable/).
    """)
    return


@app.cell
def _():
    # Import pandas
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # Create a series of three different colours
    colours = pd.Series(['Orange','Blue','Green'])
    return (colours,)


@app.cell
def _(colours):
    # View the series of different colours
    colours
    return


@app.cell
def _(pd):
    # Create a series of three different car types and view it
    cars = pd.Series(['BMW','Lamborghini','Ferrari'])
    cars
    return (cars,)


@app.cell
def _(cars, colours, pd):
    # Combine the Series of cars and colours into a DataFrame
    car_df = pd.DataFrame({"Cars":cars,'Colours':colours})
    car_df
    return


@app.cell
def _(pd):
    # Import "../data/car-sales.csv" and turn it into a DataFrame
    car_sales = pd.read_csv('./../datasets/car-sales.csv')
    return (car_sales,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** Since you've imported `../data/car-sales.csv` as a DataFrame, we'll now refer to this DataFrame as 'the car sales DataFrame'.
    """)
    return


@app.cell
def _(car_sales):
    # Export the DataFrame you created to a .csv file
    car_sales.to_csv('car_sales.csv')
    return


@app.cell
def _(car_sales):
    # Find the different datatypes of the car data DataFrame
    car_sales.dtypes
    return


@app.cell
def _(car_sales):
    # Describe your current car sales DataFrame using describe()
    car_sales.describe()
    return


@app.cell
def _(car_sales):
    # Get information about your DataFrame using info()
    car_sales.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What does it show you?
    """)
    return


@app.cell
def _(pd):
    # Create a Series of different numbers and find the mean of them
    pd.Series([2,3,4,5,1,2,3,6,7]).mean()
    return


@app.cell
def _(pd):
    # Create a Series of different numbers and find the sum of them
    pd.Series([2,3,4,5,1,2,3,6,7]).sum()
    return


@app.cell
def _(car_sales):
    # List out all the column names of the car sales DataFrame
    car_sales.columns
    return


@app.cell
def _(car_sales):
    # Find the length of the car sales DataFrame
    len(car_sales)
    return


@app.cell
def _(car_sales):
    # Show the first 5 rows of the car sales DataFrame
    car_sales.head()
    return


@app.cell
def _(car_sales):
    # Show the first 7 rows of the car sales DataFrame
    car_sales.head(7)
    return


@app.cell
def _(car_sales):
    # Show the bottom 5 rows of the car sales DataFrame
    car_sales.tail()
    return


@app.cell
def _(car_sales):
    # Use .loc to select the row at index 3 of the car sales DataFrame
    car_sales.loc[3]
    return


@app.cell
def _(car_sales):
    # Use .iloc to select the row at position 3 of the car sales DataFrame
    car_sales.iloc[3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how they're the same? Why do you think this is?

    Check the pandas documentation for [.loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html) and [.iloc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html). Think about a different situation each could be used for and try them out.
    """)
    return


@app.cell
def _(car_sales):
    # Select the "Odometer (KM)" column from the car sales DataFrame
    car_sales['Odometer (KM)']
    return


@app.cell
def _(car_sales):
    # Find the mean of the "Odometer (KM)" column in the car sales DataFrame
    car_sales['Odometer (KM)'].mean()
    return


@app.cell
def _(car_sales):
    # Select the rows with over 100,000 kilometers on the Odometer
    car_sales[car_sales['Odometer (KM)'] > 100000]
    return


@app.cell
def _(car_sales, pd):
    # Create a crosstab of the Make and Doors columns
    pd.crosstab(car_sales['Make'], car_sales['Doors'])
    return


@app.cell
def _(car_sales):
    # Group columns of the car sales DataFrame by the Make column and find the average
    car_sales.groupby('Make').mean()
    return


@app.cell
def _(car_sales):
    # Import Matplotlib and create a plot of the Odometer column
    # Don't forget to use %matplotlib inline
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    plt.plot(car_sales['Odometer (KM)'])
    return (plt,)


@app.cell
def _(car_sales, plt):
    # Create a histogram of the Odometer column using hist()
    plt.hist(car_sales['Odometer (KM)'])
    return


@app.cell
def _(car_sales, plt):
    # Try to plot the Price column using plot()
    plt.plot(car_sales['Price'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why didn't it work? Can you think of a solution?

    You might want to search for "how to convert a pandas string columb to numbers".

    And if you're still stuck, check out this [Stack Overflow question and answer on turning a price column into integers](https://stackoverflow.com/questions/44469313/price-column-object-to-int-in-pandas).

    See how you can provide the example code there to the problem here.
    """)
    return


@app.cell
def _(car_sales):
    # Remove the punctuation from price column

    car_sales['Price'] = car_sales['Price'].str.replace('[\$\,]','').astype(float)
    car_sales
    return


@app.cell
def _(car_sales):
    # Check the changes to the price column
    car_sales
    return


@app.cell
def _():
    # Remove the two extra zeros at the end of the price column
    return


@app.cell
def _():
    # Check the changes to the Price column
    return


@app.cell
def _(car_sales):
    # Change the datatype of the Price column to integers
    car_sales['Price'] = car_sales['Price'].astype(int)
    car_sales.dtypes
    return


@app.cell
def _(car_sales):
    # Lower the strings of the Make column
    car_sales['Make'].str.lower()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you check the car sales DataFrame, you'll notice the Make column hasn't been lowered.

    How could you make these changes permanent?

    Try it out.
    """)
    return


@app.cell
def _(car_sales):
    # Make lowering the case of the Make column permanent
    car_sales['Make'] = car_sales['Make'].str.lower()
    return


@app.cell
def _(car_sales):
    # Check the car sales DataFrame
    car_sales
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the Make column stays lowered after reassigning.

    Now let's deal with missing data.
    """)
    return


@app.cell
def _(pd):
    # Import the car sales DataFrame with missing data ("../data/car-sales-missing-data.csv")
    car_sales_missing = pd.read_csv('./../datasets/car-sales-missing-data.csv')

    # Check out the new DataFrame
    car_sales_missing
    return (car_sales_missing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the missing values are represented as `NaN` in pandas DataFrames.

    Let's try fill them.
    """)
    return


@app.cell
def _(car_sales_missing):
    # Fill the Odometer column missing values with the mean of the column inplace
    car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean(), inplace=True)
    return


@app.cell
def _(car_sales_missing):
    # View the car sales missing DataFrame and verify the changes
    car_sales_missing
    return


@app.cell
def _(car_sales_missing):
    # Remove the rest of the missing data inplace
    car_sales_missing.dropna(inplace=True)
    return


@app.cell
def _(car_sales_missing):
    # Verify the missing values are removed by viewing the DataFrame
    car_sales_missing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll now start to add columns to our DataFrame.
    """)
    return


@app.cell
def _(car_sales_missing):
    # Create a "Seats" column where every row has a value of 5
    car_sales_missing['Seats'] = 5
    car_sales_missing
    return


@app.cell
def _(car_sales_missing):
    # Create a column called "Engine Size" with random values between 1.3 and 4.5
    # Remember: If you're doing it from a Python list, the list has to be the same length
    # as the DataFrame
    import random
    col_len = len(car_sales_missing)
    engine_size_list = [random.randrange(13,45)/10 for each in range(col_len)]
    car_sales_missing['Engine Size'] = engine_size_list
    car_sales_missing
    return


app._unparsable_cell(
    r"""
    # Create a column which represents the price of a car per kilometer
    # Then view the DataFrame
    car_sales_missing['Price-Per-KM'] =
    """,
    name="_"
)


@app.cell
def _():
    # Remove the last column you added using .drop()
    return


@app.cell
def _():
    # Shuffle the DataFrame using sample() with the frac parameter set to 1
    # Save the the shuffled DataFrame to a new variable
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the index numbers get moved around. The [`sample()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) function is a great way to get random samples from your DataFrame. It's also another great way to shuffle the rows by setting `frac=1`.
    """)
    return


@app.cell
def _():
    # Reset the indexes of the shuffled DataFrame
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the index numbers have been changed to have order (start from 0).
    """)
    return


@app.cell
def _():
    # Change the Odometer values from kilometers to miles using a Lambda function
    # Then view the DataFrame
    return


@app.cell
def _(car_sales):
    # Change the title of the Odometer (KM) to represent miles instead of kilometers
    car_sales.rename(columns = {'Odometer (KM)':'Odometer (miles)'}, inplace=True)
    car_sales
    return


@app.cell
def _(car_sales):
    car_sales.drop(columns=['Odometers (miles)'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extensions

    For more exercises, check out the pandas documentation, particularly the [10-minutes to pandas section](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html).

    One great exercise would be to retype out the entire section into a Jupyter Notebook of your own.

    Get hands-on with the code and see what it does.

    The next place you should check out are the [top questions and answers on Stack Overflow for pandas](https://stackoverflow.com/questions/tagged/pandas?sort=MostVotes&edited=true). Often, these contain some of the most useful and common pandas functions. Be sure to play around with the different filters!

    Finally, always remember, the best way to learn something new to is try it. Make mistakes. Ask questions, get things wrong, take note of the things you do most often. And don't worry if you keep making the same mistake, pandas has many ways to do the same thing and is a big library. So it'll likely take a while before you get the hang of it.
    """)
    return


if __name__ == "__main__":
    app.run()
