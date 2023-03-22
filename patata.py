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
