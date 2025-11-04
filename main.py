import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

def hyperparam_pipeline_eval(pipeline, params):
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
    grid_search.fit(X_train, Y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, Y_test)
    print(f"Test set score: {test_score}")

def build_eval_pipeline(preprocessor, regressor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

def eval_pipeline(pipeline):
    print("Training model...")
    pipeline.fit(X_train, Y_train)
    score = pipeline.score(X_test, Y_test)
    print(f"Model R^2 Score: {score}")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('Initialising transformer')
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        bedroom_ratio = X['total_bedrooms'] / X['total_rooms']
        household_rooms = X['total_rooms'] / X['households']
        return np.c_[bedroom_ratio, household_rooms]

if __name__ == "__main__":
    path = kagglehub.dataset_download("camnugent/california-housing-prices")
    print("Path to dataset files:", path)

    csv_path = os.path.join(path, "housing.csv")

    data = pd.read_csv(csv_path)
    data.dropna(inplace=True)

    X = data.drop(['median_house_value'], axis=1)
    Y = data['median_house_value']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    log_pipeline = Pipeline([
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('scale', StandardScaler())
    ])

    engineer_pipeline = Pipeline([
        ('engineer', FeatureEngineer()),
        ('scale', StandardScaler())
    ])

    scale_pipeline = Pipeline([
        ('scale', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('log_transform', log_pipeline, ['total_rooms', 'total_bedrooms', 'population', 'households']),
            ('scale_transform', scale_pipeline, ['longitude', 'latitude', 'housing_median_age', 'median_income']),
            ('categorical_transform', cat_pipeline, ['ocean_proximity']),
            ('engineer_transform', engineer_pipeline, ['total_rooms', 'total_bedrooms', 'households'])
        ],
        remainder='drop'
    )

    lr_pipeline = build_eval_pipeline(preprocessor, LinearRegression())
    rfr_pipeline = build_eval_pipeline(preprocessor, RandomForestRegressor())

    eval_pipeline(lr_pipeline)
    eval_pipeline(rfr_pipeline)

    param_grid = {
        'n_estimators': [50, 80, 100],
    }
    hyperparam_pipeline_eval(rfr_pipeline, param_grid)