from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def encode_categorical_columns(df):
    df_encoded = df.copy()
    object_columns = df_encoded.select_dtypes(include=["object"]).columns
    for column in object_columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
    return df_encoded


def best_k(df, target_column, min_k=2, max_k=15):
    le = LabelEncoder()
    df_encoded = df.copy()
    object_columns = df_encoded.select_dtypes(include=['object']).columns
    for column in object_columns:
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(
        df_encoded), columns=df_encoded.columns)
    X = df_imputed.drop(target_column, axis=1)
    y = df_imputed[target_column]
    pipeline = Pipeline(steps=[('model', KNeighborsRegressor(n_neighbors=3))])
    params = {'model__n_neighbors': [3, 5, 7],
              'model__weights': ['uniform', 'distance']}
    best_k = 0
    best_score = -np.inf
    for k in range(min_k, max_k+1):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, params, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_k = k
    return best_k


def impute_missing_values_with_knn(df, n_neighbors=10):
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_columns = df_imputed.select_dtypes(include=['int64', 'float64']).columns
    df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
    return df_imputed
