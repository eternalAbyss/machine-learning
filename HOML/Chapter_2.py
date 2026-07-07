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

    - The higher the norm index, the more it focuses on large values and neglects small ones.
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
def _(np):
    np.random.seed(42)
    return


@app.cell
def _(housing, shuffle_and_split_data):
    train_set_with_shuffle, test_set_with_shuffle = shuffle_and_split_data(housing, 0.2)
    return test_set_with_shuffle, train_set_with_shuffle


@app.cell
def _(test_set_with_shuffle, train_set_with_shuffle):
    print("Length of the train set :", len(train_set_with_shuffle))
    print("Length of the test set :", len(test_set_with_shuffle))
    return


@app.cell
def _(train_set_with_shuffle):
    train_set_with_shuffle.head()
    return


@app.cell
def _(np):
    from zlib import crc32

    def is_id_in_test_set(identifier, test_ratio):
        return crc32(np.int64(identifier)) < test_ratio * 2**32

    def split_data_with_id_hash(data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    return (split_data_with_id_hash,)


@app.cell
def _(housing, split_data_with_id_hash):
    housing_with_id = housing.reset_index()  # adds an `index` column
    train_set_with_index, test_set_with_index = split_data_with_id_hash(housing_with_id, 0.2, "index")
    return (housing_with_id,)


@app.cell
def _(housing, housing_with_id, split_data_with_id_hash):
    # If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset and that no row ever gets deleted.

    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set_composite, test_set_composite = split_data_with_id_hash(housing_with_id, 0.2, "id")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far the train and test splits are :
    1. `train_set_with_shuffle`, `test_set_with_shuffle` $\rightarrow$ Simple shuffle and split based on indices.
    2. `train_set_with_index`, `test_set_with_index` $\rightarrow$ add an index column and then split using creating a hash and then decided if a row belongs in the test set or not. Remains consistent across run but need to make sure that data gets appended at the end of the table each time and no row ever gets deleted.
    3. `train_set_composite`, `test_set_composite` $\rightarrow$ create a hash using the most stable features and use that to split the train and test dataset. Works well with new data but need to make sure that the hash should be a unique identifier for current and future data points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Scikit-Learn` provides a few functions to split datasets into multiple subsets in various ways.
    We will generally be using the `Sklean` splits and would almost never be writing the train, test splits manually.
    """)
    return


@app.cell
def _(housing):
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    return test_set, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Random sampling methods are only good if we have a dataset large enough otherwise we could introduce significant sampling bias. To solve this we use `stratified sampling`.

    **Stratified Sampling** : the population is divided into homogenous subgroups called *strata*, and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall population.
    """)
    return


@app.cell
def _(housing, np, pd):
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    return


@app.cell
def _(housing, plt):
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `split()` method yields the training and test indices, not the data itself. Having multiple splits can be useful if you want to better estimate the performance of your model
    """)
    return


@app.cell
def _(housing):
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])
    return


@app.cell
def _(housing, train_test_split):
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    return strat_test_set, strat_train_set


@app.cell
def _(strat_test_set):
    strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    return


@app.cell
def _(housing):
    housing["income_cat"].value_counts() / len(housing)
    return


@app.cell
def _(test_set):
    test_set["income_cat"].value_counts() / len(test_set)
    return


@app.cell
def _(strat_test_set, strat_train_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return


@app.cell
def _(strat_train_set):
    housing_strat = strat_train_set.copy()
    return (housing_strat,)


@app.cell
def _(housing, plt):
    housing.plot(kind='scatter', x='longitude', y='latitude', grid='true', alpha=0.2)
    plt.show()
    return


@app.cell
def _(housing, plt):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10, 7))
    plt.show()
    return


@app.cell
def _(housing_strat):
    housing_strat.head()
    return


@app.cell
def _(housing_strat):
    corr_matrix = housing_strat.corr(numeric_only=True)
    return (corr_matrix,)


@app.cell
def _(corr_matrix):
    corr_matrix["median_house_value"].sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another way to check for correlation between attributes is to use the Pandas `scatter_matrix()` function, which plots every numerical attribute against every other numerical attribute. Since there are now $9$ numerical attributes, you would get $9^2 = 81$ plots, which would not fit on a page - so we decide to focus on a few promising attributes that seem most correlated with the median housing value.
    """)
    return


@app.cell
def _(housing, plt):
    from pandas.plotting import scatter_matrix

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at the correlation scatterplots, it seems like the most promising attribute to predict the median house value is the median income, so we zoom in on their scatterplot
    """)
    return


@app.cell
def _(housing, plt):
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot reveals a few things. First, the correlation is indeed quite strong; you can clearly see the upward trend, and the points are not too dispersed. Second, the price cap you noticed earlier is clearly visible as a horizontal line at $500,000. But the plot also reveals other less obvious straight lines: a horizontal line around $450,000, another around $350,000, perhaps one around $280,000, and a few more below that. You may want to try removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks.
    """)
    return


@app.cell
def _(housing_strat):
    housing_strat["rooms_per_house"] = housing_strat["total_rooms"] / housing_strat["households"]
    housing_strat["bedrooms_ratio"] = housing_strat["total_bedrooms"] / housing_strat["total_rooms"]
    housing_strat["people_per_house"] = housing_strat["population"] / housing_strat["households"]
    return


@app.cell
def _(housing_strat):
    corr_matrix_with_added_cols = housing_strat.corr(numeric_only=True)
    return (corr_matrix_with_added_cols,)


@app.cell
def _(corr_matrix_with_added_cols):
    corr_matrix_with_added_cols["median_house_value"].sort_values(ascending=False)
    return


@app.cell
def _(strat_train_set):
    housing_values = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return (housing_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most machine learning algorithms cannot work with missing features, so we’ll need to take care of these.

    We have three options to fix this:
    1. Get rid of the corresponding districts.
        ```
        housing.dropna(subset=["total_bedrooms"], inplace=True)
        ```
    2. Get rid of the whole attribute.
        ```
        housing.drop("total_bedrooms", axis=1)
        ```
    3. Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
        ```
        median = housing["total_bedrooms"].median()
        housing["total_bedrooms"].fillna(median, inplace=True)
        ```
    """)
    return


@app.cell
def _():
    # median = housing_values["total_bedrooms"].median()
    # housing_values.fillna({"total_bedrooms" : median}, inplace=True)
    return


@app.cell
def _():
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    return (imputer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the median can only be computed on numerical attributes, we then need to create a copy of the data with only the numerical attributes.
    """)
    return


@app.cell
def _(housing_values, np):
    # Reset the values to initialization of housing_values before running this
    housing_num = housing_values.select_dtypes(include=[np.number])
    return (housing_num,)


@app.cell
def _(housing_num):
    housing_num.info()
    return


@app.cell
def _(housing_num, imputer):
    imputer.fit(housing_num)
    return


@app.cell
def _(housing_num):
    housing_num.median().values
    return


@app.cell
def _(housing_num, imputer):
    X = imputer.transform(housing_num)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Missing values can also be replaced with the mean value (strategy="mean"), or with the most frequent value (strategy="most_frequent"), or with a constant value (strategy="constant", fill_value=…​). The last two strategies support non-numerical
    """)
    return


@app.cell
def _(X, housing_num, pd):
    # wrap X in a DataFrame and recover the column names and index from housing_num
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)
    return


@app.cell
def _(housing_values):
    housing_cat = housing_values[["ocean_proximity"]]
    return (housing_cat,)


@app.cell
def _(housing_cat):
    from sklearn.preprocessing import OrdinalEncoder

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    return housing_cat_encoded, ordinal_encoder


@app.cell
def _(housing_cat_encoded):
    housing_cat_encoded[:8]
    return


@app.cell
def _(ordinal_encoder):
    ordinal_encoder.categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values. This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, and “excellent”), but it is obviously not the case for the `ocean_proximity` column (for example, categories 0 and 4 are clearly more similar than categories 0 and 1). To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is "<1H OCEAN" (and 0 otherwise), another attribute equal to 1 when the category is "INLAND" (and 0 otherwise), and so on. This is called **one-hot encoding**, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold). The new attributes are sometimes called *dummy attributes*.
    """)
    return


@app.cell
def _(housing_cat):
    from sklearn.preprocessing import OneHotEncoder

    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    return OneHotEncoder, housing_cat_1hot


@app.cell
def _(housing_cat_1hot):
    housing_cat_1hot.toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, you can set sparse_output=False when creating the OneHotEncoder, in which case the transform() method will return a regular (dense) NumPy array directly:
    """)
    return


@app.cell
def _(OneHotEncoder, housing_cat):
    cat_encoder = OneHotEncoder(sparse_output=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    return (housing_cat_1hot,)


@app.cell
def _():
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
