import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from sklearn.datasets import fetch_openml
    import pandas as pd

    return fetch_openml, pd


@app.cell
def _(fetch_openml):
    mnist = fetch_openml('mnist_784', as_frame=False)
    return (mnist,)


@app.cell
def _(mnist):
    X, y = mnist.data, mnist.target
    return X, y


@app.cell
def _(X):
    import matplotlib.pyplot as plt

    def plot_digit(image_data):
        image = image_data.reshape(28,28)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

    some_digit = X[0]
    plot_digit(some_digit)
    plt.show()
    return plot_digit, plt, some_digit


@app.cell
def _(y):
    print(y[0])
    return


@app.cell
def _(X, y):
    div_ratio = 60000
    X_train, X_test, y_train, y_test = X[:div_ratio], X[div_ratio:], y[:div_ratio], y[div_ratio:]
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training a binary Classifier

    Training a model to distinguish between two classes, 5 and non-5.
    """)
    return


@app.cell
def _(y_test, y_train):
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')
    return (y_train_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SGDClassifier
    Stochastic Gradient Descent (SGD) is capable of handling very large datasets efficiently. SGD deals with training instances independently, one at a time, which also makes SGD well suited for online learning.
    """)
    return


@app.cell
def _(X_train, y_train_5):
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    return SGDClassifier, sgd_clf


@app.cell
def _(sgd_clf, some_digit):
    sgd_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Performance Measures
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-Validation
    """)
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_score

    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    return (cross_val_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though this gives a really high score as the accuracy, it is not trustable because there's a huge chance that it's just saying that none of the images are actually 5. This would mean that 90% of the time it is accurate, it is correct. Even when it's just saying that an image of 5 is not actually 5, the accuracy just drops down a bit because it's just 10% of the images which are actually 5. This is not a good metric to evaluate the performance of the model.
    Accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets. A much better way to evaluate the performance of a classifier is to look at the **confusion matrix**.
    """)
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3) # add shuffle=True if the dataset is not already shuffled

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrices

    The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B, for all A/B pairs. For example, to know the number of times the classifier confused images of 8s with 0s, you would look at row #8, column #0 of the confusion matrix.
    """)
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_predict

    # Makes K-folds and returns the predictions made on each test fold
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) 
    return cross_val_predict, y_train_pred


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_train_5, y_train_pred)
    cm
    return cm, confusion_matrix


@app.cell
def _(pd):
    def convert_conf_matrix_to_df(matrix):
        matrix = matrix.tolist()
        matrix_dict = {
            ">" : ["Actual Negative", "Actual Positive"],
            "Predicted Negative" : [matrix[0][0], matrix[1][0]],
            "Predicted Positive" : [matrix[0][1], matrix[1][1]]
        }
        matrix_df = pd.DataFrame(matrix_dict)
        return matrix_df

    return (convert_conf_matrix_to_df,)


@app.cell
def _(cm, convert_conf_matrix_to_df):
    convert_conf_matrix_to_df(cm)
    return


@app.cell
def _(confusion_matrix, convert_conf_matrix_to_df, y_train_5):
    y_train_perfect_conditions = y_train_5 # Pretend we reached perfection
    convert_conf_matrix_to_df(confusion_matrix(y_train_5, y_train_perfect_conditions))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ precision = \frac{TP}{TP+FP} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ recall = \frac{TP}{TP+FN}$$
    """)
    return


@app.cell
def _(y_train_5, y_train_pred):
    # Implementing precision and Recall

    from sklearn.metrics import precision_score, recall_score

    print("Precision Score :", precision_score(y_train_5, y_train_pred))
    print("Recall Score :", recall_score(y_train_5, y_train_pred))
    return precision_score, recall_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is often convenient to combine precision and recall into a single metric called the **F1 score**, especially when you need a single metric to compare two classifiers. The F1 score is the harmonic mean of precision and recall. Harmonic mean gives much more weight to low values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ F_1 = 2 \times \left[\frac{precision \times recall}{precision + recall}\right] $$
    """)
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import f1_score
    f1_score(y_train_5, y_train_pred)
    return (f1_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Precision/Recall Trade-Off

    Increasing precision reduces recall, and vice versa.

    Instead of calling the classifier’s predict() method, you can call its decision_function() method, which returns a score for each instance. You can then use any threshold you want to make predictions based on those scores.
    """)
    return


@app.cell
def _(X_train, cross_val_predict, sgd_clf, y_train_5):
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    return (y_scores,)


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    return precision_recall_curve, precisions, recalls, thresholds


@app.cell
def _(plt, precisions, recalls, thresholds):
    import numpy as np

    # Find the values at the chosen threshold (e.g., threshold = 3000 or your chosen threshold)
    threshold = 3000  # replace with your specific threshold variable if different
    idx = np.argmax(thresholds >= threshold)
    prec_at_thresh = precisions[idx]
    rec_at_thresh = recalls[idx]

    plt.figure(figsize=(8, 4))

    # 1. Curves and vertical threshold line
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

    # 2. Points (circles) at the threshold intersection
    plt.plot(threshold, prec_at_thresh, "bo")
    plt.plot(threshold, rec_at_thresh, "go")

    # Optional horizontal indicator lines to the y-axis
    plt.hlines(prec_at_thresh, -50000, threshold, "b", "dotted")
    plt.hlines(rec_at_thresh, -50000, threshold, "g", "dotted")

    # 3. Beautification: axis bounds, labels, grid, and legend
    plt.axis([-50000, 50000, 0, 1.0])
    plt.xlabel("Threshold")
    plt.grid(True)
    plt.legend(loc="center right")

    plt.show()
    return np, prec_at_thresh, rec_at_thresh


@app.cell
def _(plt, prec_at_thresh, precisions, rec_at_thresh, recalls):
    plt.figure(figsize=(6, 5))

    # 2. Main precision-recall curve
    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

    # 3. Dotted indicator lines and point at threshold 3,000
    plt.plot([rec_at_thresh, rec_at_thresh], [0.0, prec_at_thresh], "k:")
    plt.plot([0.0, rec_at_thresh], [prec_at_thresh, prec_at_thresh], "k:")
    plt.plot([rec_at_thresh], [prec_at_thresh], "ko", label="Point at threshold 3,000")

    # 4. Curved arrow and "Higher threshold" annotation
    plt.annotate(
        "Higher\nthreshold",
        xy=(rec_at_thresh + 0.1, prec_at_thresh - 0.1),
        xytext=(rec_at_thresh + 0.05, prec_at_thresh - 0.25),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=0.3",
            linewidth=2,
            color="black",
        ),
        fontsize=11,
        ha="left",
    )

    # 5. Beautify: axes bounds, labels, grid, and legend
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=11)

    plt.show()
    return


@app.cell
def _(precisions, thresholds):
    # Search for the lowest threshold that gives at least 90% precision
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[idx_for_90_precision]
    print(threshold_for_90_precision)
    return (threshold_for_90_precision,)


@app.cell
def _(threshold_for_90_precision, y_scores):
    # Make predictions on the training set
    y_train_pred_90 = (y_scores >= threshold_for_90_precision)
    return (y_train_pred_90,)


@app.cell
def _(precision_score, recall_score, y_train_5, y_train_pred_90):
    print("Precision score : ", precision_score(y_train_5, y_train_pred_90))
    print("Recall score :", recall_score(y_train_5, y_train_pred_90))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The FixedThresholdClassifier class lets you wrap a binary classifier and set the desired threshold manually. If the underlying classifier has a predict_proba() method, then the threshold should be a value between 0 and 1 (the default is 0.5). Otherwise, it should be a decision score, comparable to the output of the model’s decision_function() (the default is 0).

    - The TunedThresholdClassifierCV class uses k-fold cross-validation to automatically find the optimal threshold for a given metric. By default, it tries to find the threshold that maximizes the model’s balanced accuracy: that’s the average of each class’s recall. However, you can select another metric to optimize for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The ROC Curve

    The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate (FPR). The FPR (also called the fall-out) is the ratio of negative instances that are incorrectly classified as positive. It is equal to 1 – the true negative rate (TNR), which is the ratio of negative instances that are correctly classified as negative. The TNR is also called specificity. Hence, the ROC curve plots sensitivity (recall) versus 1 – specificity.
    """)
    return


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds_roc = roc_curve(y_train_5, y_scores)
    return fpr, tpr


@app.cell
def _(fpr, plt, threshold_for_90_precision, thresholds, tpr):
    idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

    plt.figure(figsize=(6, 5))

    # 1. Base plots from your snippet
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k:", label="Random classifier's ROC curve")
    plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

    # 2. Dotted reference lines to the axes
    plt.plot([fpr_90, fpr_90], [0.0, tpr_90], "k:")
    plt.plot([0.0, fpr_90], [tpr_90, tpr_90], "k:")

    # 3. Arrow pointing to the 90% precision threshold point
    plt.annotate(
        "Minimum\nthreshold",
        xy=(fpr_90, tpr_90),
        xytext=(fpr_90 + 0.15, tpr_90 - 0.25),
        arrowprops=dict(
            facecolor="black",
            shrink=0.08,
            width=1.5,
            headwidth=8,
        ),
        fontsize=11,
        ha="center",
    )

    # 4. Beautify: axis bounds, labels, grid, and legend
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=10)

    plt.show()
    return


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores) # Closer to 1 means a better score
    return (roc_auc_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives. Otherwise, use the ROC curve.
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)
    return (forest_clf,)


@app.cell
def _(X_train, cross_val_predict, forest_clf, y_train_5):
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    return (y_probas_forest,)


@app.cell
def _(y_probas_forest):
    y_probas_forest[:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The CalibratedClassifierCV class from the sklearn.calibration package can calibrate the estimated probabilities using cross-validation, making them much closer to actual probabilities
    """)
    return


@app.cell
def _(precision_recall_curve, y_probas_forest, y_train_5):
    y_scores_forest = y_probas_forest[:, 1]
    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
    return precisions_forest, recalls_forest, y_scores_forest


@app.cell
def _(X_train, confusion_matrix, cross_val_predict, forest_clf, y_train_5):
    y_train_predict_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    cm_rf = confusion_matrix(y_train_5, y_train_predict_forest)
    return cm_rf, y_train_predict_forest


@app.cell
def _(cm_rf, convert_conf_matrix_to_df):
    convert_conf_matrix_to_df(cm_rf)
    return


@app.cell
def _(f1_score, y_train_5, y_train_predict_forest):
    f1_score(y_train_5, y_train_predict_forest)
    return


@app.cell
def _(roc_auc_score, y_scores_forest, y_train_5):
    roc_auc_score(y_train_5, y_scores_forest)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The AUC for roc curve is way more for `random_forest` than for the `SGD` classifier. Since the higher score is much better we can say that `random_forest` is better here. The caveat of claibration is valid and can be of issue under certain circumstances.
    """)
    return


@app.cell
def _(plt, precisions, precisions_forest, recalls, recalls_forest):
    plt.figure(figsize=(6, 5))

    # 1. Base plots from your snippet
    plt.plot(
        recalls_forest,
        precisions_forest,
        "b-",
        linewidth=2,
        label="Random Forest",
    )  #[cite: 3]
    plt.plot(
        recalls, precisions, "--", linewidth=2, label="SGD"
    )  #[cite: 3]

    # 2. Beautify the figure: labels, grid, legend, and axis limits
    plt.xlabel("Recall", fontsize=12)  #[cite: 3]
    plt.ylabel("Precision", fontsize=12)  #[cite: 3]
    plt.axis([0, 1, 0, 1])  #[cite: 3]
    plt.grid(True)  #[cite: 3]
    plt.legend(loc="lower left", fontsize=11)  #[cite: 3]

    plt.show()  #[cite: 3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiclass Classification

    ## Using binary classifiers to perform multiclass classification

    - One way to create a system that can classify the digit images into 10 classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a 1-detector, a 2-detector, and so on). Then when you want to classify an image, you get the decision score from each classifier for that image and you select the class whose classifier outputs the highest score. This is called the one-versus-the-rest (OvR) strategy, or sometimes one-versus-all (OvA).

    - Another strategy is to train a binary classifier for every pair of digits: one to distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. This is called the one-versus-one (OvO) strategy. If there are N classes, you need to train N × (N – 1) / 2 classifiers.
    """)
    return


@app.cell
def _(X_train):
    X_train.shape
    return


@app.cell
def _(X_train, y_train):
    from sklearn.svm import SVC

    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train[:20000], y_train[:20000]) # Training on the first 20000 images only to save time
    return SVC, svm_clf


@app.cell
def _(some_digit, svm_clf):
    svm_clf.predict([some_digit])
    return


@app.cell
def _(some_digit, svm_clf):
    some_digit_scores = svm_clf.decision_function([some_digit])
    some_digit_scores.round(2)
    return (some_digit_scores,)


@app.cell
def _(some_digit_scores, svm_clf):
    class_id = some_digit_scores.argmax()
    print(svm_clf.classes_[class_id])
    return


@app.cell
def _(SVC, X_train, y_train):
    from sklearn.multiclass import OneVsRestClassifier

    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(X_train[:2000], y_train[:2000])
    return (ovr_clf,)


@app.cell
def _(ovr_clf, some_digit):
    ovr_clf.predict([some_digit])
    return


@app.cell
def _(ovr_clf):
    len(ovr_clf.estimators_)
    return


@app.cell
def _(X_train, cross_val_predict, plt, svm_clf, y_train):
    from sklearn.metrics import ConfusionMatrixDisplay

    y_train_pred_svm = cross_val_predict(svm_clf, X_train, y_train, cv=3)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_svm)
    plt.show()
    return (ConfusionMatrixDisplay,)


@app.cell
def _(SGDClassifier, X_train, cross_val_score, sgd_clf, y_train):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

    sgd_clf_multi = SGDClassifier(random_state=42)
    sgd_clf_multi.fit(X_train, y_train)

    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return X_train_scaled, sgd_clf_multi


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_train_scaled,
    cross_val_predict,
    plt,
    sgd_clf_multi,
    y_train,
):
    y_train_pred_sgd = cross_val_predict(sgd_clf_multi, X_train_scaled, y_train, cv=3)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_sgd)
    plt.show()
    return (y_train_pred_sgd,)


@app.cell
def _(ConfusionMatrixDisplay, plt, y_train, y_train_pred_sgd):
    sample_weight = (y_train_pred_sgd != y_train)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_sgd,
                                            sample_weight=sample_weight,
                                           normalize="true", values_format=".0%")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 36% in row #7, column #9 in the left grid does not mean that 36% of all images of 7s were misclassified as 9s. It means that 36% of the errors the model made on images of 7s were misclassifications as 9s.
    It is also possible to normalize the confusion matrix by column rather than by row: if you set normalize="pred". For example, you can see that 56% of misclassified 7s are actually 9s.

    A much simpler approach to improve the model consists of augmenting the training set with slightly shifted and rotated variants of the training images. This will force the model to learn to be more tolerant to such variations. This is called **data augmentation**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multilabel Classification

    Consider a face-recognition classifier: what should it do if it recognizes several people in the same picture? It should attach one tag per person it recognizes. Say the classifier has been trained to recognize three faces: Alice, Bob, and Charlie. Then when the classifier is shown a picture of Alice and Charlie, it should output [True, False, True] (meaning “Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple binary tags is called a multilabel classification system.
    """)
    return


@app.cell
def _(X_train, np, y_train):
    # Create a y_multilabel array containing two target labels for each digit image: 
    # the first indicates whether the digit is large (7, 8, or 9), and the second indicates whether it is odd. 
    # Then the code creates a KNeighborsClassifier instance, which supports multilabel classification, and 
    # trains this model using the multiple targets array.

    from sklearn.neighbors import KNeighborsClassifier

    y_train_large = (y_train >= '7')
    y_train_odd = (y_train.astype('int8') % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    return KNeighborsClassifier, knn_clf, y_multilabel


@app.cell
def _(knn_clf, some_digit):
    knn_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating Multiclass model
    """)
    return


@app.cell
def _(X_train, cross_val_predict, f1_score, knn_clf, y_multilabel):
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    f1_score(y_multilabel, y_train_knn_pred, average="macro")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This approach assumes that all labels are equally important, which may not be the case. One simple option is to give each label a weight equal to its support (i.e., the number of instances with that target label). To do this, simply set average="weighted" when calling the `f1_score()` function.⁠
    """)
    return


@app.cell
def _(SVC, X_train, y_multilabel):
    # Creating a chain of classifiers (one feeds the decision of other)

    from sklearn.multioutput import ClassifierChain

    chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
    chain_clf.fit(X_train[:2000], y_multilabel[:2000])
    return (chain_clf,)


@app.cell
def _(chain_clf, some_digit):
    chain_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multioutput Classification

    It is a generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values).
    """)
    return


@app.cell
def _(X_test, X_train, np):
    # Adding noise to the MNIST dataset

    rng = np.random.default_rng(seed=42)
    noise_train = rng.integers(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise_train
    noise_test = rng.integers(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise_test
    y_train_mod = X_train
    y_test_mod = X_test
    return X_test_mod, X_train_mod, y_train_mod


@app.cell
def _(KNeighborsClassifier, X_train_mod, y_train_mod):
    knn_clf_mod = KNeighborsClassifier()
    knn_clf_mod.fit(X_train_mod, y_train_mod)
    return (knn_clf_mod,)


@app.cell
def _(X_test_mod, knn_clf_mod, plot_digit, plt):
    clean_digit = knn_clf_mod.predict([X_test_mod[0]])
    plot_digit(clean_digit)
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
