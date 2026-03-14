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
    # Introduction to Matplotlib
    """)
    return


@app.cell
def _():
    # Importing the Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # '%matplotlib inline' command supported automatically in marimo
    return np, pd, plt


@app.cell
def _(plt):
    x = [1,2,3,4]
    y = [11,22,33,44]

    # Plotting a basic figure
    plt.plot(x,y)
    return x, y


@app.cell
def _(plt):
    # 1st method
    fig = plt.figure()  # Creates a figure
    _ax = fig.add_subplot()  # adds some axes
    plt.show()
    return


@app.cell
def _(plt, x, y):
    fig_1 = plt.figure()
    _ax = fig_1.add_axes([1, 1, 1, 1])
    _ax.plot(x, y)
    plt.show()
    return


@app.cell
def _(plt, x, y):
    fig_2, _ax = plt.subplots()
    _ax.plot(x, y)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matplotlib example workflow
    """)
    return


@app.cell
def _(plt):
    x_1 = [1, 2, 3, 4]
    y_1 = [10, 21, 36, 48]
    fig_3, _ax = plt.subplots(figsize=(10, 10))
    _ax.plot(x_1, y_1)
    _ax.set(title='Simple Plot', xlabel='X-axis', ylabel='Y-axis')
    fig_3.savefig('./sample-plot.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making figures with NumPy Arrays
    We will be making:
    * Line Plot
    * Scatter Plot
    * Bar Plot
    * Histogram
    * Subplot
    """)
    return


@app.cell
def _(np, plt):
    x_2 = np.linspace(0, 10, 100)
    fig_4, _ax = plt.subplots()
    _ax.plot(x_2, x_2 ** 2)
    return (x_2,)


@app.cell
def _(np, plt, x_2):
    fig_5, _ax = plt.subplots()
    _ax.scatter(x_2, np.exp(x_2))
    plt.show()
    return


@app.cell
def _(np, plt, x_2):
    fig_6, _ax = plt.subplots()
    _ax.scatter(x_2, np.sin(x_2))
    plt.show()
    return


@app.cell
def _(plt):
    nut_butter_prices = {'Peanut butter': 10, 'Almond butter': 22, 'Cashew butter': 30}
    fig_7, _ax = plt.subplots()
    _ax.bar(nut_butter_prices.keys(), nut_butter_prices.values())
    _ax.set(title="Dan's nut butter prices", xlabel='Type of butter', ylabel='Cost of 1kg jar($)')
    plt.show()
    return (nut_butter_prices,)


@app.cell
def _(nut_butter_prices, plt):
    fig_8, _ax = plt.subplots()
    _ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))
    _ax.set(title="Dan's nut butter prices", xlabel='Cost per kg ($)', ylabel='Type of butter')
    plt.show()
    return


@app.cell
def _(np, plt):
    x_3 = np.random.randint(0, 1000, size=20)
    fig_9, _ax = plt.subplots()
    _ax.hist(x_3)
    _ax.set(title='Energy Consumption', xlabel='Watt', ylabel='Hours')
    plt.show()
    return (x_3,)


@app.cell
def _(nut_butter_prices, plt):
    fig_10, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    _ax1.plot(nut_butter_prices.keys(), nut_butter_prices.values())
    _ax2.hist(nut_butter_prices.values())
    _ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values())
    _ax4.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))
    plt.show()
    return


@app.cell
def _(nut_butter_prices, plt, x_3):
    fig_11, _ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    _ax[0, 0].plot(x_3)
    _ax[0, 1].scatter(nut_butter_prices.keys(), nut_butter_prices.values())
    _ax[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values())
    _ax[1, 1].hist(nut_butter_prices.values())
    plt.show()
    return


@app.cell
def _(pd):
    car_sales = pd.read_csv('./../datasets/car-sales.csv')
    car_sales
    return (car_sales,)


@app.cell
def _(np, pd, plt):
    ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2020", periods = 1000))
    ts = ts.cumsum()
    ts.plot()
    plt.show()
    return


@app.cell
def _(car_sales):
    car_sales
    return


@app.cell
def _(car_sales):
    car_sales["Price"] = car_sales["Price"].str.replace('[$/,]','', regex=True).astype(float)
    return


@app.cell
def _(car_sales):
    car_sales
    return


@app.cell
def _(car_sales, pd):
    car_sales["Sale Date"] = pd.date_range('1/1/2020', periods=len(car_sales))
    return


@app.cell
def _(car_sales):
    car_sales
    return


@app.cell
def _(car_sales):
    car_sales["Total Sale"]=car_sales["Price"].astype(float).cumsum()
    return


@app.cell
def _(car_sales, plt):
    fig_12, (_ax1, _ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    _ax1.plot(car_sales['Sale Date'], car_sales['Total Sale'])
    _ax2.plot(car_sales['Sale Date'], car_sales['Price'])
    _ax1.set(title='Total revenue by car sales', xlabel='Date', ylabel='Amount ($)')
    _ax2.set(title='Price of Car Sold on daily basis', xlabel='Date', ylabel='Amount ($)')
    plt.show()
    return


@app.cell
def _(car_sales):
    car_sales.plot(x="Sale Date", y="Total Sale");
    return


@app.cell
def _(car_sales):
    car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter")
    return


@app.cell
def _(np, pd):
    # Creating new data
    x_4 = np.random.rand(10, 4)
    df = pd.DataFrame(x_4)
    df.plot.bar()
    return (df,)


@app.cell
def _(df):
    df.plot(kind="bar");
    return


@app.cell
def _(car_sales):
    car_sales.plot(y="Odometer (KM)", x="Make", kind="bar");
    return


@app.cell
def _(car_sales):
    car_sales["Odometer (KM)"].plot.hist();
    return


@app.cell
def _(car_sales):
    car_sales
    return


@app.cell
def _(car_sales, plt):
    fig_13, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    _ax1.bar(x=car_sales['Colour'], height=car_sales['Price'])
    _ax2.plot(car_sales['Odometer (KM)'])
    _ax2.set(xlabel='Index Number', ylabel='Kms Driven')
    _ax3.hist(car_sales['Price'])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importing another dataset to visualize it
    Practicing to visualize better
    """)
    return


@app.cell
def _(pd):
    heart_desease= pd.read_csv('./../datasets/heart-disease.csv')
    heart_desease.head()
    return (heart_desease,)


@app.cell
def _(heart_desease):
    heart_desease.plot.hist(subplots=True, figsize=(10,30));
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which should we use pyplot or matplotlib OO method?

    * When plotting something quickly, okay to use pyplot method
    * When plotting something more advanced, use the OO method
    """)
    return


@app.cell
def _(heart_desease):
    over_50 = heart_desease[heart_desease['age']>50]
    len(over_50)
    return (over_50,)


@app.cell
def _(over_50):
    # Pyplot method 
    over_50.plot(kind='scatter',
                 x='age',
                 y='chol',
                 c='target');
    return


@app.cell
def _(over_50, plt):
    fig_14, _ax = plt.subplots(figsize=(10, 6))
    over_50.plot(kind='scatter', x='age', y='chol', c='target', ax=_ax)
    plt.show()
    return


@app.cell
def _(over_50, plt):
    fig_15, _ax = plt.subplots(figsize=(10, 6))
    scatter = _ax.scatter(x=over_50['age'], y=over_50['chol'], c=over_50['target'])
    _ax.set(title='Heart Disease and Cholestrol Levels', xlabel='Age', ylabel='Cholestrol')
    _ax.legend(*scatter.legend_elements(), title='target')
    _ax.axhline(over_50['chol'].mean(), linestyle='--')
    plt.show()
    return (scatter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating a subplot of Age, cholestrol and Maximum heart rate
    """)
    return


@app.cell
def _(over_50, plt, scatter):
    fig_16, (_ax0, _ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    _scatter1 = _ax0.scatter(x=over_50['age'], y=over_50['chol'], c=over_50['target'])
    _ax0.set(ylabel='Cholestrol', title='Scatter plot of Age Vs Cholestrol')
    _ax0.legend(*scatter.legend_elements(), title='target')
    _ax0.axhline(over_50['chol'].mean(), linestyle='--')
    _scatter2 = _ax1.scatter(x=over_50['age'], y=over_50['thalach'], c=over_50['target'])
    _ax1.set(xlabel='Age', ylabel='Maximum Heart rate', title='Scatter plot of Age Vs Maximum Heart Rate')
    _ax1.legend(*scatter.legend_elements(), title='target')
    _ax1.axhline(over_50['thalach'].mean(), linestyle='--')
    fig_16.suptitle('Heart Disease Analysis', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Customising matplotlib plots
    """)
    return


@app.cell
def _(plt):
    # See the different styles availables
    plt.style.available
    return


@app.cell
def _(car_sales):
    car_sales['Price'].plot();
    return


@app.cell
def _(plt):
    plt.style.use('seaborn-whitegrid')
    return


@app.cell
def _(car_sales):
    car_sales.plot();
    return


@app.cell
def _(car_sales):
    car_sales['Odometer (KM)'].plot(kind="bar").legend().set_visible(True);
    return


@app.cell
def _(over_50, plt):
    fig_17, _ax = plt.subplots(figsize=(10, 6))
    scatter_1 = _ax.scatter(x=over_50['age'], y=over_50['chol'], c=over_50['target'], cmap='winter')
    _ax.set(title='Heart Disease and Cholestrol Levels', xlabel='Age', ylabel='Cholestrol')
    _ax.legend(*scatter_1.legend_elements(), title='target')
    _ax.axhline(over_50['chol'].mean(), linestyle='--')
    plt.show()
    return (scatter_1,)


@app.cell
def _(over_50, plt, scatter_1):
    fig_18, (_ax0, _ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    _scatter1 = _ax0.scatter(x=over_50['age'], y=over_50['chol'], c=over_50['target'], cmap='winter')
    _ax0.set(ylabel='Cholestrol', title='Scatter plot of Age Vs Cholestrol', xlim=[50, 80], ylim=[100, 600])
    _ax0.legend(*scatter_1.legend_elements(), title='target')
    _ax0.axhline(over_50['chol'].mean(), linestyle='--')
    _scatter2 = _ax1.scatter(x=over_50['age'], y=over_50['thalach'], c=over_50['target'], cmap='winter')
    _ax1.set(xlabel='Age', ylabel='Maximum Heart rate', title='Scatter plot of Age Vs Maximum Heart Rate', xlim=[50, 80], ylim=[60, 200])
    _ax1.legend(*scatter_1.legend_elements(), title='target')
    _ax1.axhline(over_50['thalach'].mean(), linestyle='--')
    fig_18.suptitle('Heart Disease Analysis', fontsize=16)
    plt.show()
    return (fig_18,)


@app.cell
def _(fig_18):
    fig_18.savefig('./heart-disease-analysis.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src='./heart-disease-analysis.png'/>
    """)
    return


if __name__ == "__main__":
    app.run()
