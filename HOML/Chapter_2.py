import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ML imports
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib.pyplot as plt
    import urllib.request
    import pandas as pd
    import numpy as np
    import tarfile

    # System imports
    from pathlib import Path

    return OneHotEncoder, Path, np, pd, plt, tarfile, urllib


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
    return (train_test_split,)


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
    return housing_labels, housing_values


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
    return SimpleImputer, imputer


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
def _():
    # cat_encoder = OneHotEncoder()
    # housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    return


@app.cell
def _():
    # housing_cat_1hot.toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, you can set `sparse_output=False` when creating the **OneHotEncoder**, in which case the `transform()` method will return a regular (dense) NumPy array directly.

    The advantage of OneHotEncoder is that it remembers which categories it was trained on. This is very important because once your model is in production, it should be fed exactly the same features as during training: no more, no less.
    """)
    return


@app.cell
def _(OneHotEncoder, housing_cat):
    cat_encoder = OneHotEncoder(sparse_output=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    return (housing_cat_1hot,)


@app.cell
def _(housing_cat_1hot):
    housing_cat_1hot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If a categorical attribute has a large number of possible categories (e.g., country code, profession, species), then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance. If this happens, you may want to replace the categorical input with useful numerical features related to the categories: for example, you could replace the `ocean_proximity` feature with the distance to the ocean (similarly, a country code could be replaced with the country’s population and GDP per capita). Alternatively, you can use one of the encoders provided by the `category_encoders` package on GitHub. Or, when dealing with neural networks, you can replace each category with a learnable, low-dimensional vector called an embedding. This is an example of **representation learning**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Min-max scaling** (many people call this **normalization**) is the simplest: for each attribute, the values are shifted and rescaled so that they end up ranging from 0 to 1. This is performed by subtracting the min value and dividing by the difference between the min and the max. Scikit-Learn provides a transformer called MinMaxScaler for this. It has a feature_range hyperparameter that lets you change the range if, for some reason, you don’t want 0–1 (e.g., neural networks work best with zero-mean inputs, so a range of –1 to 1 is preferable).
    """)
    return


@app.cell
def _(housing_num):
    from sklearn.preprocessing import StandardScaler

    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    return (StandardScaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before you scale the feature, you should first transform it to shrink the heavy tail, and if possible to make the distribution roughly symmetrical.

    Another approach to handle heavy-tailed features consists in bucketizing the feature. This means chopping its distribution into roughly equal-sized buckets, and replacing each feature value with the index of the bucket it belongs to.

    Bucketizing with equal-sized buckets results in a feature with an almost uniform distribution, so there’s no need for further scaling, or you can just divide by the number of buckets to force the values to the 0–1 range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The similarity measure is typically computed using a radial basis function (RBF)—any function that depends only on the distance between the input value and a fixed point. The most commonly used RBF is the Gaussian RBF, whose output value decays exponentially as the input value moves away from the fixed point.

    ### Why and When to Calculate Similarity

    Machine learning algorithms, especially linear regression models, look for straight-line relationships (e.g., "as age increases, price decreases"). However, real-world data is rarely that simple. Calculating similarity helps you translate a complex, non-linear pattern into a linear one that algorithms can easily understand.

    #### Why calculate it?

    To create "sweet spots": If houses are most valuable specifically around 35 years of age, a linear model will fail to capture that peak. By replacing the "Age" feature with a "Similarity to 35" feature, the model simply learns that a higher similarity score equals a higher price.

    To handle localized clusters: It allows the model to treat specific pockets of data differently from the rest of the dataset.

    #### When to calculate it?

    **Geographical Data:** Similarity is highly effective for coordinates (latitude/longitude). Instead of feeding raw coordinates to a model, you can calculate the similarity (distance) to major city centers, transit hubs, or employment districts.

    **Cyclical or Time Data:** When dealing with time of day or days of the week, where the "distance" between 11:00 PM and 1:00 AM is very short, despite the numbers being far apart numerically.

    **Support Vector Machines (SVMs):** Similarity features are the foundational math behind SVMs. This algorithm uses the "Kernel Trick" to mathematically project data into higher dimensions—using metrics like the RBF kernel—so that previously tangled classifications can be easily separated by a straight line.
    """)
    return


@app.cell
def _(housing):
    from sklearn.metrics.pairwise import rbf_kernel

    # Claculates the age similarity between the housing_median_age and 35
    gamma = 0.1
    age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=gamma)
    return (rbf_kernel,)


@app.cell
def _(housing, np, plt, rbf_kernel):
    # 1. Setup the figure and the primary axis (for the histogram)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot the histogram of actual housing ages
    ax1.hist(housing["housing_median_age"], bins=50, color="#1874CD")
    ax1.set_xlabel("Housing median age", fontsize=14)
    ax1.set_ylabel("Number of districts", fontsize=14)

    # 2. Setup the secondary axis (for the RBF curves) sharing the same x-axis
    ax2 = ax1.twinx()

    # Create a smooth, sequentially sorted range of ages for the bell curves
    # We go from the minimum age to the maximum age in the dataset
    ages = np.linspace(housing["housing_median_age"].min(), 
                       housing["housing_median_age"].max(), 
                       500).reshape(-1, 1)
    landmark = [[35]]

    # 3. Calculate RBF similarities for both gamma values
    simil_gamma_010 = rbf_kernel(ages, landmark, gamma=0.10)
    simil_gamma_003 = rbf_kernel(ages, landmark, gamma=0.03)

    # 4. Plot both curves on the secondary y-axis
    ax2.plot(ages, simil_gamma_010, "b-", linewidth=2, label="gamma = 0.10")
    ax2.plot(ages, simil_gamma_003, "b--", linewidth=2, label="gamma = 0.03")

    # Format the secondary axis
    ax2.set_ylabel("Age similarity", color="b", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="b")
    ax2.set_ylim([0, 1.05]) # Add a slight margin at the top
    ax2.legend(loc="upper left", fontsize=14)

    # Display the final combined graph
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far we’ve only looked at the input features, but the target values may also need to be transformed. For example, if the target distribution has a heavy tail, you may choose to replace the target with its logarithm. But if you do, the regression model will now predict the log of the median house value, not the median house value itself. You will need to compute the exponential of the model’s prediction if you want the predicted median house value.

    A simple option is to use a TransformedTargetRegressor. We just need to construct it, giving it the regression model and the label transformer, then fit it on the training set, using the original unscaled labels. It will automatically use the transformer to scale the labels and train the regression model on the resulting scaled labels. Then, when we want to make a prediction, it will call the regression model’s `predict()` method and use the scaler’s `inverse_transform()` method to produce the prediction.
    """)
    return


@app.cell
def _(StandardScaler, housing, housing_labels, housing_values):
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import TransformedTargetRegressor

    some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data
    model = TransformedTargetRegressor(LinearRegression(),
                                       transformer=StandardScaler())
    model.fit(housing_values[["median_income"]], housing_labels)
    predictions = model.predict(some_new_data)
    return (LinearRegression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For transformations that don’t require any training, you can just write a function that takes a NumPy array as input and outputs the transformed array.
    """)
    return


@app.cell
def _(housing, np):
    from sklearn.preprocessing import FunctionTransformer

    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing[["population"]])
    return (FunctionTransformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There’s no inverse function for the RBF kernel, since there are always two values at a given distance from a fixed point (except at distance 0). Also `rbf_kernel()` does not treat the features separately. If you pass it an array with two features, it will measure the 2D distance (Euclidean) to measure similarity.

    Here are a few things to note for **custom transformers**:

    - The `sklearn.utils.validation` package contains several functions we can use to validate the inputs.

    - Scikit-Learn pipelines require the `fit()` method to have two arguments `X` and `y`, which is why we need the `y=None` argument even though we don’t use `y`.

    - All Scikit-Learn estimators set `n_features_in_` in the `fit()` method, and they ensure that the data passed to `transform()` or `predict()` has this number of features.

    - The `fit()` method must return self.

    - All estimators should set `feature_names_in_` in the `fit()` method when they are passed a DataFrame. Moreover, all transformers should provide a `get_feature_names_out()` method, as well as an `inverse_transform()` method when their transformation can be reversed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A custom transformer can (and often does) use other estimators in its implementation. For example, the following code demonstrates custom transformer that uses a `KMeans` clusterer in the `fit()` method to identify the main clusters in the training data, and then uses `rbf_kernel()` in the `transform()` method to measure how similar each sample is to each cluster center:
    """)
    return


@app.cell
def _(rbf_kernel):
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.cluster import KMeans

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
            self.n_clusters = n_clusters
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y=None, sample_weight=None):
            self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
            self.kmeans_.fit(X, sample_weight=sample_weight)
            return self  # always return self!

        def transform(self, X):
            return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

        def get_feature_names_out(self, names=None):
            return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

    return (ClusterSimilarity,)


@app.cell
def _(ClusterSimilarity, housing):
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
    return (similarities,)


@app.cell
def _(similarities):
    similarities[:3].round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `Pipeline` constructor takes a list of name/estimator pairs (2-tuples) defining a sequence of steps.
    """)
    return


@app.cell
def _(SimpleImputer, StandardScaler):
    from sklearn.pipeline import make_pipeline

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return make_pipeline, num_pipeline


@app.cell
def _(housing_num, num_pipeline):
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    return (housing_num_prepared,)


@app.cell
def _(housing_num, housing_num_prepared, num_pipeline, pd):
    # For recovering a nice DataFrame use the get_feature_names_out() method
    df_housing_num_prepared = pd.DataFrame(housing_num_prepared, 
                                           columns=num_pipeline.get_feature_names_out(), 
                                           index=housing_num.index)
    return (df_housing_num_prepared,)


@app.cell
def _(df_housing_num_prepared):
    df_housing_num_prepared.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pipelines support indexing; for example, `pipeline[1]` returns the second estimator in the pipeline, and `pipeline[:-1]` returns a Pipeline object containing all but the last estimator.

    You can also access the estimators via the steps attribute, which is a list of name/estimator pairs, or via the `named_steps` dictionary attribute, which maps the names to the estimators. For example, `num_pipeline["simpleimputer"]` returns the estimator named `"simpleimputer"`.
    """)
    return


@app.cell
def _(OneHotEncoder, SimpleImputer, make_pipeline, np, num_pipeline):
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.compose import ColumnTransformer

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preprocessing_ = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )
    return (
        ColumnTransformer,
        cat_pipeline,
        make_column_selector,
        preprocessing_,
    )


@app.cell
def _(housing, preprocessing_):
    housing_prepared_ = preprocessing_.fit_transform(housing)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap of processes till now

    - Missing values in numerical features will be imputed by replacing them with the median, as most ML algorithms don’t expect missing values. In categorical features, missing values will be replaced by the most frequent category.

    - The categorical feature will be one-hot encoded, as most ML algorithms only accept numerical inputs.

    - A few ratio features will be computed and added: bedrooms_ratio, rooms_per_house, and people_per_house. Hopefully these will better correlate with the median house value, and thereby help the ML models.

    - A few cluster similarity features will also be added. These will likely be more useful to the model than latitude and longitude.

    - Features with a long tail will be replaced by their logarithm, as most models prefer features with roughly uniform or Gaussian distributions.

    - All numerical features will be standardized, as most ML algorithms prefer when all features have roughly the same scale.
    """)
    return


@app.cell
def _(
    ClusterSimilarity,
    ColumnTransformer,
    FunctionTransformer,
    SimpleImputer,
    StandardScaler,
    cat_pipeline,
    make_column_selector,
    make_pipeline,
    np,
):
    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"]  # feature names out

    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler())

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    cluster_similarity = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler())
    preprocessing = ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                   "households", "median_income"]),
            ("geo", cluster_similarity, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age
    return (preprocessing,)


@app.cell
def _(housing_values, preprocessing):
    housing_prepared = preprocessing.fit_transform(housing_values)
    housing_prepared.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select and Train a Model
    """)
    return


@app.cell
def _(
    LinearRegression,
    housing_labels,
    housing_values,
    make_pipeline,
    preprocessing,
):
    # Making a pipepline and training the model
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    print("Training in progress...")
    lin_reg.fit(housing_values, housing_labels)
    print("Model trained")
    return (lin_reg,)


@app.cell
def _(housing_values, lin_reg):
    housing_predictions = lin_reg.predict(housing_values)
    return (housing_predictions,)


@app.cell
def _(housing_labels, housing_predictions):
    from sklearn.metrics import root_mean_squared_error

    lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
    lin_rmse
    return


@app.cell
def _(housing_values, make_pipeline, preprocessing):
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fir(housing_values)
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
