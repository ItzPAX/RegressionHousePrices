# California Housing Price Prediction

This is a simple Python script that demonstrates a complete machine learning workflow for predicting median housing values in California.

It uses the [California Housing Prices dataset from Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and builds a preprocessing pipeline with `scikit-learn` to train and evaluate multiple regression models.

## Features

* **Kaggle Dataset:** Automatically downloads the dataset using the `kagglehub` library.
* **Preprocessing Pipeline:** Uses `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer` to create a robust preprocessing workflow that handles:
    * Log-transforming and scaling skewed numerical features.
    * Scaling standard numerical features.
    * One-hot encoding categorical features.
* **Custom Transformer:** Includes a custom `FeatureEngineer` to create new features as part of the pipeline.
* **Model Training:**
    1.  Trains and evaluates a `LinearRegression` model.
    2.  Trains and evaluates a `RandomForestRegressor` model.
* **Hyperparameter Tuning:** Uses `GridSearchCV` to find the best hyperparameters for the `RandomForestRegressor` in a way that avoids data leakage.
