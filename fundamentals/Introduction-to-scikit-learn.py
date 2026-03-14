import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Scikit Learn (Sklearn)

    This notebook shows some of the most useful functions of scikit library.

    ### What we are going to cover
    0. An end-to-end scikit learn workflow
    1. Getting the data ready
    2. Choose the right estimator/algorithm for our problems
    3. Fit the model/algorithm/estimator and use it to make predictions on our data
    4. Evaluating a model
    5. Improve a model
    6. Save and load a trained model
    7. Putting it all together!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. An end-to-end scikit learn workflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get the data ready
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    # '%matplotlib inline' command supported automatically in marimo
    return RandomForestClassifier, np, pd, plt, train_test_split


@app.cell
def _(pd):
    # Reading the data
    heart_disease = pd.read_csv('./../datasets/heart-disease.csv')
    heart_disease
    return (heart_disease,)


@app.cell
def _(heart_disease):
    # Create X (features matrix)
    X = heart_disease.drop('target', axis=1)
    X

    # Create Y (labels)
    Y = heart_disease['target']
    Y
    return X, Y


@app.cell
def _(RandomForestClassifier):
    # 2. Choose the right model and hyperparameters
    clf = RandomForestClassifier()

    # We will keep the default hyperparameters
    clf.get_params()
    return (clf,)


@app.cell
def _(X, Y, train_test_split):
    # 3. fit the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(X_train, Y_train, clf):
    clf.fit(X_train, Y_train);
    return


@app.cell
def _(X_test, clf):
    # make a prediction
    y_preds = clf.predict(X_test)
    return (y_preds,)


@app.cell
def _(y_preds):
    y_preds
    return


@app.cell
def _(X_train, Y_train, clf):
    # Evaluate the model on the training data and the test data
    clf.score(X_train, Y_train)
    return


@app.cell
def _(X_test, Y_test, clf):
    clf.score(X_test, Y_test)
    return


@app.cell
def _():
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    return accuracy_score, classification_report, confusion_matrix


@app.cell
def _(Y_test, classification_report, y_preds):
    print(classification_report(Y_test, y_preds))
    return


@app.cell
def _(Y_test, confusion_matrix, y_preds):
    confusion_matrix(Y_test, y_preds)
    return


@app.cell
def _(Y_test, accuracy_score, y_preds):
    accuracy_score(Y_test, y_preds)
    return


@app.cell
def _(RandomForestClassifier, X_test, X_train, Y_test, Y_train, np):
    # 5. Improve a model
    # Try different amount n estimators
    np.random.seed(42)
    for i in range(10, 100, 10):
        print(f'Trying model with {i} estimators...')
        clf_1 = RandomForestClassifier(n_estimators=i).fit(X_train, Y_train)
        print(f'Model accuracy on test set: {clf_1.score(X_test, Y_test) * 100:.2f}%')
        print('')
    return


@app.cell
def _(RandomForestClassifier, X_test, X_train, Y_test, Y_train):
    clf_2 = RandomForestClassifier(n_estimators=40).fit(X_train, Y_train)
    clf_2.score(X_test, Y_test)
    return (clf_2,)


@app.cell
def _(clf_2):
    # 6. Save a model and load it 
    import pickle
    pickle.dump(clf_2, open('random_frst_model1.pkl', 'wb'))
    return (pickle,)


@app.cell
def _(X_test, Y_test, pickle):
    loaded_model = pickle.load(open("random_frst_model1.pkl","rb"))
    loaded_model.score(X_test, Y_test)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore") # To remove the warnings from Jupyter notebook we can use this command
    warnings.filterwarnings("default") # To set it back to the default value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Getting the data ready to be used with machine learning

    **Three main things we have to do :**

        1. Split the data into features and labels (usually 'X' and 'y')
        2. Filling (also called imputing) or disregarding missing values
        3. Converting non-numerical values to numerical values (also called feature encoding)
    """)
    return


@app.cell
def _(heart_disease):
    heart_disease.head()
    return


@app.cell
def _(heart_disease):
    X_1 = heart_disease.drop('target', axis=1)
    y = heart_disease['target']
    return X_1, y


@app.cell
def _(RandomForestClassifier, X_1, accuracy_score, train_test_split, y):
    # Split the data into training and test set
    X_train_1, X_test_1, y_train, y_test = train_test_split(X_1, y, test_size=0.2)
    clf_3 = RandomForestClassifier().fit(X=X_train_1, y=y_train)
    y_preds_1 = clf_3.predict(X_test_1)
    accuracy_score(y_test, y_preds_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Make sure the data is all numerical
    """)
    return


@app.cell
def _(pd):
    car_sales = pd.read_csv('./../datasets/car-sales-extended.csv')
    car_sales.dtypes
    return (car_sales,)


@app.cell
def _(car_sales):
    car_sales['Make']
    return


@app.cell
def _(car_sales, train_test_split):
    # Split the data into training and test data set 
    X_2 = car_sales.drop('Price', axis=1)
    y_1 = car_sales['Price']
    X_train_2, X_test_2, y_train_1, y_test_1 = train_test_split(X_2, y_1, test_size=0.2)
    (X_train_2.shape, X_test_2.shape, y_train_1.shape, y_test_1.shape)
    return X_2, y_1


@app.cell
def _():
    # Train the model on un-prepared data
    # clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
    # y_preds = clf.predict(X_test)
    # accuracy_score(y_test, y_preds)
    return


@app.cell
def _(X_2):
    X_2.head()
    return


@app.cell
def _(X_2):
    # Turn the categories into numbers
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    categorical_features = ['Make', 'Colour', 'Doors']
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')
    transformed_X = transformer.fit_transform(X_2)
    transformed_X
    return ColumnTransformer, OneHotEncoder, transformed_X


@app.cell
def _(car_sales, pd):
    # Another method to hot encode our data
    Dummies = pd.get_dummies(car_sales[["Make","Colour","Doors"]])
    Dummies.head()
    return


@app.cell
def _(y_1):
    y_1
    return


@app.cell
def _(np, train_test_split, transformed_X, y_1):
    from sklearn.ensemble import RandomForestRegressor
    np.random.seed(42)
    # Fitting the transformed X into ML model
    X_train_3, X_test_3, y_train_2, y_test_2 = train_test_split(transformed_X, y_1, test_size=0.2)
    # 1. Splitting the data
    model = RandomForestRegressor()
    # Training the model
    model.fit(X_train_3, y_train_2)
    return RandomForestRegressor, X_test_3, model, y_test_2


@app.cell
def _(X_test_3, model, y_test_2):
    model.score(X_test_3, y_test_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 What if there were missing values?

    1. Fill them with some value (also known as imputation).
    2. Remove the samples with missing data altogether.
    """)
    return


@app.cell
def _(pd):
    # Import car sales missing data
    car_sales_missing = pd.read_csv('./../datasets/car-sales-extended-missing-data.csv')
    car_sales_missing.head()
    return (car_sales_missing,)


@app.cell
def _(car_sales_missing):
    car_sales_missing.isna().sum()
    return


@app.cell
def _(ColumnTransformer, OneHotEncoder, car_sales_missing):
    X_3 = car_sales_missing.drop('Price', axis=1)
    y_2 = car_sales_missing['Price']
    categorical_features_1 = ['Make', 'Colour', 'Doors']
    one_hot_1 = OneHotEncoder()
    transformer_1 = ColumnTransformer([('one_hot', one_hot_1, categorical_features_1)], remainder='passthrough')
    transformed_X_1 = transformer_1.fit_transform(X_3)
    transformed_X_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Option 1 : Fill missing Data with pandas
    """)
    return


@app.cell
def _(car_sales_missing):
    car_sales_missing["Make"].fillna('missing', inplace=True)
    car_sales_missing["Colour"].fillna('missing', inplace=True)
    car_sales_missing['Odometer (KM)'].fillna(car_sales_missing['Odometer (KM)'].mean(), inplace=True)
    car_sales_missing['Doors'].fillna(4, inplace=True)
    car_sales_missing.isna().sum()
    return


@app.cell
def _(car_sales_missing):
    # Remove rows with missing Price values
    car_sales_missing.dropna(inplace=True)
    car_sales_missing.isna().sum()
    return


@app.cell
def _(car_sales_missing):
    len(car_sales_missing)
    return


@app.cell
def _(ColumnTransformer, OneHotEncoder, car_sales_missing):
    categorical_features_2 = ['Make', 'Colour', 'Doors']
    one_hot_2 = OneHotEncoder()
    transformer_2 = ColumnTransformer([('one_hot', one_hot_2, categorical_features_2)], remainder='passthrough')
    Data_transformed = transformer_2.fit_transform(car_sales_missing)
    Data_transformed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Option 2: Fill missing value using sklearn
    """)
    return


@app.cell
def _(pd):
    car_sales_missing_1 = pd.read_csv('../datasets/car-sales-extended-missing-data.csv')
    car_sales_missing_1.head()
    return (car_sales_missing_1,)


@app.cell
def _(car_sales_missing_1):
    car_sales_missing_1.isna().sum()
    return


@app.cell
def _(car_sales_missing_1):
    # First we need to drop the rows which do not have label
    car_sales_missing_1.dropna(subset=['Price'], inplace=True)  # This drops the subset 'Price' which does not have value
    car_sales_missing_1['Price'].isna().sum()
    return


@app.cell
def _(car_sales_missing_1):
    # Split into X & Y
    X_4 = car_sales_missing_1.drop('Price', axis=1)
    y_3 = car_sales_missing_1['Price']
    return X_4, y_3


@app.cell
def _(ColumnTransformer, X_4, pd):
    # Fill missing values in X
    from sklearn.impute import SimpleImputer
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    door_imputer = SimpleImputer(strategy='constant', fill_value=4)
    # Fill Categorical values with 'missing' and numerical values with mean
    num_imputer = SimpleImputer(strategy='mean')
    cat_features = ['Make', 'Colour']
    door_features = ['Doors']
    num_features = ['Odometer (KM)']
    # Define Columns
    imputer = ColumnTransformer([('cat_imputer', cat_imputer, cat_features), ('door_imputer', door_imputer, door_features), ('num_imputer', num_imputer, num_features)], remainder='passthrough')
    filled_X = pd.DataFrame(imputer.fit_transform(X_4))
    # Create an imputer (something that fills the missing values)
    # Transform the data 
    filled_X
    return (filled_X,)


@app.cell
def _(ColumnTransformer, OneHotEncoder, filled_X):
    cat_features_1 = [0, 1, 2]
    one_hot_3 = OneHotEncoder()
    transformer_3 = ColumnTransformer([('one_hot', one_hot_3, cat_features_1)], remainder='passthrough')
    transformed_X_2 = transformer_3.fit_transform(filled_X)
    transformed_X_2
    return (transformed_X_2,)


@app.cell
def _(RandomForestRegressor, np, train_test_split, transformed_X_2, y_3):
    np.random.seed(42)
    X_train_4, X_test_4, y_train_3, y_test_3 = train_test_split(transformed_X_2, y_3, test_size=0.2)
    model_1 = RandomForestRegressor()
    model_1.fit(X_train_4, y_train_3)
    # Splitting the data
    # Fitting the model with new data
    model_1.score(X_test_4, y_test_3)
    return


@app.cell
def _(filled_X):
    len(filled_X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Process of filling missing values is called imputation
    * Process of  turning non-numerical values into numerical values is called feature engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Choosing the right estimator / algorithm for your machine learning problem

    Some things to note:
    * Sklearn refers machine learning models/algorithms as estimators
    * Classification Problem - predicting a category
        * Sometimes you will see `clf` (Short for classifier) used as a classification estimator
    * Regression Problem - predicting a number

    If you are working on a machine learning problem and looking to use sklearn and not sure what model you should use, refer to sklearn machine learning map : https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Picking a machine learning model for a regression problem
    Let's use california housing dataset
    """)
    return


@app.cell
def _():
    # Get California Housing dataset
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    housing
    return fetch_california_housing, housing


@app.cell
def _(housing, pd):
    housing_df = pd.DataFrame(housing["data"], columns=housing['feature_names'])
    housing_df
    return (housing_df,)


@app.cell
def _(housing, housing_df):
    housing_df['target'] = housing['target']
    housing_df
    return


@app.cell
def _():
    # housing_df = housing_df.drop('MedHouseVal', axis=1)
    return


@app.cell
def _(housing_df):
    housing_df
    return


@app.cell
def _(RandomForestRegressor, housing_df, np, train_test_split):
    # Import the model
    np.random.seed(42)
    X_5 = housing_df.drop('target', axis=1)
    # Setup random seed
    y_4 = housing_df['target']
    X_train_5, X_test_5, y_train_4, y_test_4 = train_test_split(X_5, y_4, test_size=0.2)
    # split the data into X and y
    model_2 = RandomForestRegressor(n_estimators=120)
    model_2.fit(X_train_5, y_train_4)
    # splitting the training and test data
    # Training the model
    model_2.score(X_test_5, y_test_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forests
    Random forests mitigate this problem well. A random forest is simply a collection of decision trees whose results are aggregated into one final result. Their ability to limit overfitting without substantially increasing error due to bias is why they are such powerful models.
    """)
    return


@app.cell
def _(pd):
    # Importing the data
    heart_disease_1 = pd.read_csv('./../datasets/heart-disease.csv')
    len(heart_disease_1)
    return (heart_disease_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trying the SVM model as is given in the map
    """)
    return


@app.cell
def _(heart_disease_1, np, train_test_split):
    # Setting the random seed
    np.random.seed(42)
    X_6 = heart_disease_1.drop('target', axis=1)
    # Preparing Data for X and y
    y_5 = heart_disease_1['target']
    X_train_6, X_test_6, y_train_5, y_test_5 = train_test_split(X_6, y_5, test_size=0.2)
    from sklearn import svm
    # Spliting the data in test and train dataset
    model_3 = svm.SVC()
    model_3.fit(X_train_6, y_train_5)
    #Importing the model 
    # Selecting a model
    # Fitting the model
    # Evaluating the model
    model_3.score(X_test_6, y_test_5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SVM didn't work...
    Trying the K Nearest Neighbors model as further suggested by the map
    """)
    return


@app.cell
def _(heart_disease_1, np, train_test_split):
    # Setting the random seed 
    np.random.seed(42)
    X_7 = heart_disease_1.drop('target', axis=1)
    # Preparing Data for X and y
    y_6 = heart_disease_1['target']
    X_train_7, X_test_7, y_train_6, y_test_6 = train_test_split(X_7, y_6, test_size=0.2)
    from sklearn.neighbors import KNeighborsClassifier
    # Spliting the data in test and train dataset
    model_4 = KNeighborsClassifier()
    model_4.fit(X_train_7, y_train_6)
    #Importing the model 
    # Selecting a model
    # Fitting the model
    # Evaluating the model
    model_4.score(X_test_7, y_test_6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    K Nearest Neighbors didn't work too... Trying the Random forest classifier model
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, np, train_test_split):
    # Setting the random seed 
    np.random.seed(42)
    X_8 = heart_disease_1.drop('target', axis=1)
    # Preparing Data for X and y
    y_7 = heart_disease_1['target']
    X_train_8, X_test_8, y_train_7, y_test_7 = train_test_split(X_8, y_7, test_size=0.2)
    model_5 = RandomForestClassifier()
    # Spliting the data in test and train dataset
    model_5.fit(X_train_8, y_train_7)
    #Importing the model 
    # Selecting a model
    # Fitting the model
    # Evaluating the model
    model_5.score(X_test_8, y_test_7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TidBit:

    1. If you have structured Data use ensemble methods
    2. If you have unstructured data use Deep learning or transfer learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit the model/algorithm on our data and use it to make predictions

    ### 3.1 Fitting the model to data
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, np, train_test_split):
    # Setting the random seed 
    np.random.seed(42)
    X_9 = heart_disease_1.drop('target', axis=1)
    # Preparing Data for X and y
    y_8 = heart_disease_1['target']
    X_train_9, X_test_9, y_train_8, y_test_8 = train_test_split(X_9, y_8, test_size=0.2)
    model_6 = RandomForestClassifier()
    # Spliting the data in test and train dataset
    model_6.fit(X_train_9, y_train_8)
    #Importing the model 
    # Selecting a model
    # Fit the model to the data (training the machine learning model)
    # Evaluating the model (use the pattern model has learnt)
    model_6.score(X_test_9, y_test_8)
    return X_test_9, model_6, y_test_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Make predictions using the machine learning models

    2 ways to make predictions:
    1. `predict()`
    2. `predict_proba()`
    """)
    return


@app.cell
def _(X_test_9, accuracy_score, model_6, y_test_8):
    # Use a trained model to make predictions
    y_preds_2 = model_6.predict(X_test_9)
    accuracy_score(y_test_8, y_preds_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make predictions with `predict_proba()`
    """)
    return


@app.cell
def _(X_test_9, model_6):
    # predict_proba returns probabilities of a classification label
    model_6.predict_proba(X_test_9[:5])
    return


@app.cell
def _(X_test_9, model_6):
    # Let's predict on the same data...
    model_6.predict(X_test_9[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `predict()` can also be used with regression models
    """)
    return


@app.cell
def _(fetch_california_housing):
    housing_1 = fetch_california_housing()
    housing_1
    return (housing_1,)


@app.cell
def _(housing_1, pd):
    housing_df_1 = pd.DataFrame(housing_1['data'], columns=housing_1['feature_names'])
    housing_df_1['target'] = housing_1['target']
    housing_df_1.head()
    return (housing_df_1,)


@app.cell
def _(RandomForestRegressor, housing_df_1, np, train_test_split):
    # Setting random seed
    np.random.seed(42)
    X_10 = housing_df_1.drop('target', axis=1)
    # Splitting the data into X and y
    y_9 = housing_df_1['target']
    X_train_10, X_test_10, y_train_9, y_test_9 = train_test_split(X_10, y_9, test_size=0.2)
    from sklearn.metrics import r2_score
    # splitting the data into training and test Dataset
    model_7 = RandomForestRegressor()
    model_7.fit(X_train_10, y_train_9)
    # Selecting a model
    y_preds_3 = model_7.predict(X_test_10)
    # Fitting the model into data
    # Predicting using the trained model 
    # Evaluating the model
    r2_score(y_test_9, y_preds_3)
    return (r2_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluating a machine learning model
    Three ways to evaluate scikit-learn models/estimators:

    1. Estimator's built in `score()` method
    2. The `scoring` parameter
    3. Problem specific metric functions

    You can read more about these here: https://scikit-learn.org/stable/modules/model_evaluation.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 Evaluating a model/estimator with `score` method
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, np, train_test_split):
    # Setting up random seed
    np.random.seed(42)
    X_11 = heart_disease_1.drop('target', axis=1)
    # Importing the model
    y_10 = heart_disease_1['target']
    X_train_11, X_test_11, y_train_10, y_test_10 = train_test_split(X_11, y_10, test_size=0.2)
    # Splitting the data into X & y
    model_8 = RandomForestClassifier()
    model_8.fit(X_train_11, y_train_10)
    # Spliting the data into test and train
    # Selecting a model/estimator
    # Fitting the Estimator
    # Evaluating the model/estimator
    model_8.score(X_test_11, y_test_10)
    return


@app.cell
def _(RandomForestRegressor, housing_df_1, np, train_test_split):
    # Setting random seed
    np.random.seed(42)
    X_12 = housing_df_1.drop('target', axis=1)
    # Splitting the data into X and y
    y_11 = housing_df_1['target']
    X_train_12, X_test_12, y_train_11, y_test_11 = train_test_split(X_12, y_11, test_size=0.2)
    model_9 = RandomForestRegressor(n_estimators=100)
    # splitting the data into training and test Dataset
    model_9.fit(X_train_12, y_train_11)
    # Selecting a model
    # Fitting the model into data
    # Scoring the model 
    model_9.score(X_test_12, y_test_11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The default score() evaluation metric is r_squared for regression algorithms
    Highest = 1.0 and Lowest = 0.0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Evaluating a model using the `Scoring` parameter
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, np, train_test_split):
    from sklearn.model_selection import cross_val_score
    np.random.seed(42)
    # Setting up random seed
    X_13 = heart_disease_1.drop('target', axis=1)
    y_12 = heart_disease_1['target']
    # Importing the model
    X_train_13, X_test_13, y_train_12, y_test_12 = train_test_split(X_13, y_12, test_size=0.2)
    model_10 = RandomForestClassifier()
    # Splitting the data into X & y
    # Spliting the data into test and train
    # Selecting a model/estimator
    # Fitting the Estimator
    model_10.fit(X_train_13, y_train_12)
    return X_13, X_test_13, cross_val_score, model_10, y_12, y_test_12


@app.cell
def _(X_13, cross_val_score, model_10, y_12):
    cross_val_score(model_10, X_13, y_12, cv=10)
    return


@app.cell
def _(X_test_13, model_10, y_test_12):
    # Default scoring parameter of classifier = mean accuracy
    model_10.score(X_test_13, y_test_12)
    return


@app.cell
def _(X_13, cross_val_score, model_10, y_12):
    # Scoring parameter set to None by default
    cross_val_score(model_10, X_13, y_12, scoring=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2.1 Classification model evaluation metrics

    1. Accuracy
    2. Area under ROC curve
    3. Confusion matrix
    4. Classification Report

    **Accuracy**
    """)
    return


@app.cell
def _(RandomForestClassifier, cross_val_score, heart_disease_1, np):
    np.random.seed(42)
    X_14 = heart_disease_1.drop('target', axis=1)
    y_13 = heart_disease_1['target']
    clf_4 = RandomForestClassifier()
    cv_score = cross_val_score(clf_4, X_14, y_13, cv=5)
    print(cv_score.mean())
    return X_14, clf_4, y_13


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Area under the reciever operating characteristics curve (AUC/ROC)**

    * Area under curve (AUC)
    * ROC Curve

    ROC Curves are a comparison of a model's true positive rate(tpr) versus a model's false positive rate (fpr)

    * True Positive = model predicts 1 when truth is 1
    * False Postive = model predicts 1 when truth is 0
    * True Negative = model predicts 0 when truth is 0
    * False Negative = model predicts 0 when truth is 1
    """)
    return


@app.cell
def _(X_14, clf_4, train_test_split, y_13):
    # Prepare the data
    X_train_14, X_test_14, y_train_13, y_test_13 = train_test_split(X_14, y_13, test_size=0.2)
    # Fitting the data 
    clf_4.fit(X_train_14, y_train_13)
    return X_test_14, y_test_13


@app.cell
def _(X_test_14, clf_4):
    from sklearn.metrics import roc_curve
    # Make prediction with probabilities 
    y_probs = clf_4.predict_proba(X_test_14)
    return roc_curve, y_probs


@app.cell
def _(y_probs):
    y_probs_positive = y_probs[:,1]
    y_probs_positive[:5]
    return (y_probs_positive,)


@app.cell
def _(roc_curve, y_probs_positive, y_test_13):
    # calculate fpr, tpr and thresholds
    fpr, tpr, thresholds = roc_curve(y_test_13, y_probs_positive)
    fpr
    return fpr, tpr


@app.cell
def _(fpr, plt, tpr):
    # Create a function for plotting ROC curves
    def plot_roc_curve(fpr, tpr):
    # '%matplotlib inline' command supported automatically in marimo
        """
        Plots a ROC curve given the False Positive Rate (fpr)
        and True Positive Rate (tpr) of a model
        """
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
        plt.xlabel('False Positive Rate (fpr)')  # Plot roc curve
        plt.ylabel('True Positive Rate (tpr)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()  #Plot line with no redictive power (baseline)
        plt.show()
    plot_roc_curve(fpr, tpr)  # Customize the plot
    return (plot_roc_curve,)


@app.cell
def _(y_probs_positive, y_test_13):
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_test_13, y_probs_positive)
    return (roc_auc_score,)


@app.cell
def _(plot_roc_curve, roc_curve, y_test_13):
    # plot perfect ROC curve and AUC score
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_test_13, y_test_13)
    plot_roc_curve(fpr_1, tpr_1)
    return


@app.cell
def _(roc_auc_score, y_test_13):
    roc_auc_score(y_test_13, y_test_13)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Confusion Matrix**

    A confusion matrix is a quick wy to compare the labels a model predicts and the actual labels it was supposed to predict.

    In essence, giving you an idea where the model is getting confused.
    """)
    return


@app.cell
def _(X_test_14, clf_4, confusion_matrix, y_test_13):
    y_preds_4 = clf_4.predict(X_test_14)
    confusion_matrix(y_test_13, y_preds_4)
    return (y_preds_4,)


@app.cell
def _(pd, y_preds_4, y_test_13):
    # Visualize confusion matrix using pd.crosstab
    pd.crosstab(y_test_13, y_preds_4, rownames=['Actual Labels'], colnames=['Predicted Labels'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How to install a conda package inot the current environment from Jupyter notebook
    """)
    return


@app.cell
def _(subprocess):
    import sys
    #! conda install --yes --prefix {sys.prefix} seaborn
    subprocess.call(['conda', 'install', '--yes', '--prefix', str(sys.prefix), 'seaborn'])
    return


@app.cell
def _(confusion_matrix, y_preds_4, y_test_13):
    # Make our confusion matrix more visual using seaborn's heatmap()
    import seaborn as sns
    sns.set(font_scale=1.1)
    # Set the font scale
    conf_map = confusion_matrix(y_test_13, y_preds_4)
    # Create a confusion matrix 
    # Plot it using Seaborn
    sns.heatmap(conf_map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating a Confusion Matrix using scikit learn
    To use the new methods of creating a confusion matrix with Scikit-learn
    """)
    return


@app.cell
def _(subprocess):
    import sys
    #! conda install --yes --prefix {sys.prefix} scikit-learn
    subprocess.call(['conda', 'install', '--yes', '--prefix', str(sys.prefix), 'scikit-learn'])
    return


@app.cell
def _():
    import sklearn
    sklearn.__version__
    return


@app.cell
def _(X_14, clf_4, y_13, y_preds_4, y_test_13):
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(estimator=clf_4, X=X_14, y=y_13)
    ConfusionMatrixDisplay.from_predictions(y_true=y_test_13, y_pred=y_preds_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification Report
    """)
    return


@app.cell
def _(classification_report, y_preds_4, y_test_13):
    print(classification_report(y_test_13, y_preds_4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2.3 Finally using the `scoring` parameter
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, np, train_test_split):
    np.random.seed(42)
    X_15 = heart_disease_1.drop('target', axis=1)
    y_14 = heart_disease_1['target']
    X_train_15, X_test_15, y_train_14, y_test_14 = train_test_split(X_15, y_14, test_size=0.2)
    clf_5 = RandomForestClassifier()
    clf_5.fit(X_train_15, y_train_14)
    return X_15, clf_5, y_14


@app.cell
def _(X_15, clf_5, cross_val_score, np, y_14):
    np.random.seed(42)
    cv_acc = cross_val_score(clf_5, X_15, y_14, cv=5, scoring=None)
    # Crossvalidation accuracy
    cv_acc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using different evaluation metrics using sklearn functions
    The 3rd way to evaluate scikit-learn machine learning models/estimators is by using the `sklearn.metrics` module
    """)
    return


@app.cell
def _(RandomForestClassifier, heart_disease_1, train_test_split):
    from sklearn.metrics import f1_score, precision_score, recall_score
    X_16 = heart_disease_1.drop('target', axis=1)
    y_15 = heart_disease_1['target']
    X_train_16, X_test_16, y_train_15, y_test_15 = train_test_split(X_16, y_15, test_size=0.2)
    clf_6 = RandomForestClassifier()
    clf_6.fit(X_train_16, y_train_15)
    return X_test_16, clf_6, f1_score, precision_score, recall_score, y_test_15


@app.cell
def _(
    X_test_16,
    accuracy_score,
    clf_6,
    f1_score,
    np,
    precision_score,
    r2_score,
    recall_score,
    y_test_15,
):
    np.random.seed(42)
    y_preds_5 = clf_6.predict(X_test_16)
    print(f'The Accuracy score is : {accuracy_score(y_test_15, y_preds_5) * 100:.2f}%')
    # Accuracy Score
    print(f'The f1 score is : {f1_score(y_test_15, y_preds_5) * 100:.2f}%')
    print(f'The R2 score is : {r2_score(y_test_15, y_preds_5) * 100:.2f}%')
    print(f'The Precision score is : {precision_score(y_test_15, y_preds_5) * 100:.2f}%')
    print(f'The Recall score is : {recall_score(y_test_15, y_preds_5) * 100:.2f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### More Evaluation metrics
    """)
    return


@app.cell
def _(RandomForestRegressor, housing_df_1, np, train_test_split):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    np.random.seed(42)
    X_17 = housing_df_1.drop('target', axis=1)
    y_16 = housing_df_1['target']
    X_train_17, X_test_17, y_train_16, y_test_16 = train_test_split(X_17, y_16, test_size=0.2)
    model_11 = RandomForestRegressor()
    model_11.fit(X_train_17, y_train_16)
    y_preds_6 = model_11.predict(X_test_17)
    return mean_absolute_error, mean_squared_error, y_preds_6, y_test_16


@app.cell
def _(mean_absolute_error, mean_squared_error, r2_score, y_preds_6, y_test_16):
    # R2 Score
    print(f'The r2 score is : {r2_score(y_test_16, y_preds_6)}')
    print(f'The MAE is : {mean_absolute_error(y_test_16, y_preds_6)}')
    # Mean Absolute Error
    # Mean Squared Error
    print(f'The MSE is : {mean_squared_error(y_test_16, y_preds_6)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Improving a model

    First Predictions = baseline predictions
    First model = baseline model

    **From a data perspective:**
    * Could we collect more data? (generally the more data, the better)
    * Could we improve our data?

    **From a model perspective:**
    * Is there a better model we can use?
    * Could we improve the current model?

    **Parameters vs Hyperparameters**

    * Parameters = model find these patterns in data
    * Hyperparameters = settings on a model that you can adjust to (potentially) improve its ability to find patterns

    **Three ways to adjust Hyperparameters**
    1. By hand
    2. Randomly with RandomSearchCV
    3. Exhaustively with GridSearchCV
    """)
    return


@app.cell
def _(RandomForestClassifier):
    clf_7 = RandomForestClassifier()
    return (clf_7,)


@app.cell
def _(clf_7):
    # Get the hyperparameters
    clf_7.get_params()
    return


if __name__ == "__main__":
    app.run()
