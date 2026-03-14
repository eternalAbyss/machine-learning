# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A personal machine learning learning repository containing Jupyter notebooks that cover NumPy, pandas, Matplotlib, and scikit-learn. The focus is on hands-on learning through notebooks and exercises rather than production code.

## Working with Notebooks

Launch Jupyter to edit notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

Run a specific notebook non-interactively:
```bash
jupyter nbconvert --to notebook --execute fundamentals/Introduction-to-scikit-learn.ipynb
```

## Structure

- `fundamentals/` — Main learning notebooks (numpy, pandas, matplotlib, scikit-learn)
- `exercises/` — Practice notebooks for numpy and pandas
- `datasets/` — CSV files used across notebooks (`heart-disease.csv`, `car-sales*.csv`)
- `resources/` — Reference outlines and topic lists

## Scikit-learn Notebook Flow

The `Introduction-to-scikit-learn.ipynb` follows a 7-step ML workflow:
1. Get data ready (train/test split, feature/target separation)
2. Pick a model (RandomForestClassifier / RandomForestRegressor)
3. Fit model and make predictions
4. Evaluate (accuracy, ROC/AUC, confusion matrix, R², MAE, MSE)
5. Improve via hyperparameter tuning (RandomizedSearchCV, GridSearchCV)
6. Save/reload models (pickle, joblib)
7. End-to-end pipelines (missing data, encoding, preprocessing)

Saved model artifacts (`.pkl` files) live in `fundamentals/`.
