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
    # Pandas
    **This notebook conatins code to demonstrate the use of Pandas library**
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing the data
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('./../datasets/heart-disease.csv')
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data types
    - Pandas has two main data types series and DataFrames

    ### Series
    A series is a one dimensional data structure that contains two columns indicating the index number and the value respectively
    """)
    return


@app.cell
def _(pd):
    series = pd.Series(['Apple','Banana','Carrot','Date'])
    series
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DataFrames
    DataFrames are two dimensional data structure that contains multiple rows and columns
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Describing a DataFrame
    Pandas can be used to describe the data using various attributes and fundtions.

    ### Difference between Attribute and Functions
    Functions perform steps to achieve a particular goal while Attributes just display the value already stored in the Data Structure
    """)
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df.index
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Working with a different dataset
    This dataset is small and is added for exploratory purposes.
    """)
    return


@app.cell
def _(pd):
    car_sales_df = pd.read_csv('../datasets/car-sales.csv')
    car_sales_df.head()
    return (car_sales_df,)


@app.cell
def _(car_sales_df):
    car_sales_df.columns
    return


@app.cell
def _(car_sales_df):
    mean_distance_driven = car_sales_df['Odometer (KM)'].mean()
    total_distance_driven = car_sales_df['Odometer (KM)'].sum()
    return mean_distance_driven, total_distance_driven


@app.cell
def _(mean_distance_driven, total_distance_driven):
    print(
        'The cars have been driven an average of {} KM \nAlso, in totality all the cars have driven around {} KM'
        .format(mean_distance_driven,total_distance_driven)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viewing and Selecting Data
    """)
    return


@app.cell
def _(car_sales_df):
    car_sales_df.tail() # Returns the bottom half of the dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **To get a specific number of rows either from the top or the bottom we can use df.head(3) or df.tail(9)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### .loc and .iloc
    These functions are used to locate the object within the table. Argument of the loc refers to the index of the object and will return the same.
    loc refers to the index while iloc refers to the actual position
    """)
    return


@app.cell
def _(pd):
    # Let this be a new series
    animals = pd.Series(['Lion','Tiger','Panda','Monkey','Koala'], index=[2,3,1,0,9])
    print(animals)
    return (animals,)


@app.cell
def _(animals):
    # loc will refer to the index as represented below
    animals.loc[1]
    return


@app.cell
def _(animals):
    # iloc will refer to the actual position of the object as represented
    animals.iloc[1]
    return


@app.cell
def _(car_sales_df):
    # If the dataframe is sorted both loc and iloc will refer to the same row
    print(car_sales_df.loc[3])
    print(car_sales_df.iloc[3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **iloc and loc can be used for slicing too**
    """)
    return


@app.cell
def _(animals):
    animals
    return


@app.cell
def _(animals):
    print("Using loc\n{}".format(animals.loc[:3]))
    print('\n')
    print("Using iloc\n{}".format(animals.iloc[:3]))
    return


@app.cell
def _(car_sales_df):
    # To select a specific column we can use the name of that column
    car_sales_df['Colour']
    return


@app.cell
def _(car_sales_df):
    car_sales_df.Colour
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Boolean indexing
    when we want to use a filter on the rows that we get we do it by using boolean indexing which is as such
    """)
    return


@app.cell
def _(car_sales_df):
    car_sales_df[car_sales_df["Make"] == 'Toyota']
    return


@app.cell
def _(car_sales_df, pd):
    pd.crosstab(car_sales_df["Make"],car_sales_df['Doors'])
    return


@app.cell
def _(car_sales_df):
    car_sales_df
    return


@app.cell
def _(car_sales_df):
    car_sales_df.groupby("Make").mean()
    return


@app.cell
def _(car_sales_df):
    car_sales_df.groupby("Colour").describe()
    return


@app.cell
def _(car_sales_df):
    car_sales_df["Odometer (KM)"].plot()
    return


@app.cell
def _(car_sales_df):
    car_sales_df["Odometer (KM)"].hist()
    return


@app.cell
def _(car_sales_df):
    car_sales_df
    return


@app.cell
def _(car_sales_df):
    car_sales_df["Price"] = car_sales_df["Price"].str.replace('[\$\,]','').astype(float)
    car_sales_df
    return


@app.cell
def _(car_sales_df, plt):
    # Toyota cars price compared to the kms driven
    toyota_cars = car_sales_df[car_sales_df["Make"]=="Toyota"]
    plt.plot(toyota_cars["Odometer (KM)"], toyota_cars["Price"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manipulating Data
    In this section we will demonstrate how we can manipulate the data as per our needs
    """)
    return


@app.cell
def _(car_sales_df):
    # We can access the objects as string values using the .str method
    car_sales_df["Make"].str.lower()
    return


@app.cell
def _(car_sales_df):
    car_sales_df["Make"] = car_sales_df['Make'].str.lower()
    car_sales_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dealing with missing values
    """)
    return


@app.cell
def _(pd):
    car_sales_missing = pd.read_csv('./../datasets/car-sales-missing-data.csv')
    car_sales_missing['Odometer'].fillna(car_sales_missing['Odometer'].mean(), inplace=True)
    car_sales_missing
    return (car_sales_missing,)


@app.cell
def _(car_sales_missing):
    # inplace attribute changes the original dataFrame and there is no need to reassign the variable

    car_sales_missing['Doors'].fillna(car_sales_missing['Doors'].mean(), inplace=True)
    car_sales_missing
    return


@app.cell
def _(pd):
    car_sales_missing_1 = pd.read_csv('./../datasets/car-sales-missing-data.csv')
    car_sales_missing_1
    return (car_sales_missing_1,)


@app.cell
def _(car_sales_missing_1):
    car_sales_missing_dropped = car_sales_missing_1.dropna()
    return (car_sales_missing_dropped,)


@app.cell
def _(car_sales_missing_dropped):
    car_sales_missing_dropped.to_csv('car-sales-missing-dropped.csv', index=False)
    return


@app.cell
def _(pd):
    car_sales_missing_dropped_1 = None
    car_sales_missing_dropped_1 = pd.read_csv('./car-sales-missing-dropped.csv')
    car_sales_missing_dropped_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating a custom dataframe column
    """)
    return


@app.cell
def _(pd):
    seats_col = pd.Series([5, 5, 5, 5, 5, 5])
    car_sales_df_1 = pd.read_csv('./../datasets/car-sales.csv')
    car_sales_df_1['Seats'] = seats_col
    car_sales_df_1
    return (car_sales_df_1,)


@app.cell
def _(car_sales_df_1):
    car_sales_df_1['Seats'].fillna(car_sales_df_1['Seats'].mean(), inplace=True)
    car_sales_df_1
    return


@app.cell
def _(car_sales_df_1):
    # Column from Python List
    fuel_economy = [3.4, 5.6, 7.6, 8.9, 6.4, 2.3, 2, 5.6, 1.2, 3]
    car_sales_df_1['fuel_economy'] = fuel_economy
    car_sales_df_1
    return


@app.cell
def _(car_sales_df_1):
    car_sales_df_1['fuel_used (Gallons)'] = car_sales_df_1['Odometer (KM)'] / 100 * car_sales_df_1['fuel_economy']
    car_sales_df_1
    return


@app.cell
def _(car_sales_df_1):
    car_sales_df_1.drop('fuel_used', 1, inplace=True)
    car_sales_df_1
    return


@app.cell
def _(car_sales_df_1):
    # Create a column from a single value
    car_sales_df_1['Number_of_wheels'] = 4
    car_sales_df_1
    return


@app.cell
def _(car_sales_df_1):
    car_sales_df_1.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shuffling the data
    sample method is used to shuffle the data. The frac attribute is used to decide the percentage of data that needs the shuffling (eg. 0.5 refers to 50% of the data)
    """)
    return


@app.cell
def _(car_sales_df_1):
    car_sales_shuffled = car_sales_df_1.sample(frac=1)
    return (car_sales_shuffled,)


@app.cell
def _(car_sales_shuffled):
    car_sales_shuffled
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When working with large datasets it is a good practice to select a small amount of sample data and perform the
    operations on it. This saves time as working with the entire data can be time consuming.
    """)
    return


@app.cell
def _(car_sales_shuffled):
    car_sales_shuffled.reset_index(drop=True)
    return


@app.cell
def _(car_sales_df_1):
    car_sales_df_1['Odometer (KM)'] = car_sales_df_1['Odometer (KM)'].apply(lambda x: x / 1.6)
    return


@app.cell
def _(car_sales_df_1):
    car_sales_df_1
    return


if __name__ == "__main__":
    app.run()
