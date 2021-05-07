import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .custom_scaler import CustomStandardScaler
from src.entities.feature_params import FeatureParams


def build_numerical_pipeline_w_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("custom_scaler", CustomStandardScaler()),
        ]
    )
    return num_pipeline


def build_numerical_pipeline_wo_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))
        ]
    )
    return num_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="if_binary")),
        ]
    )
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:

    if params.normalize_numerical:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_w_scaler(),
                    params.numerical_features,
                ),
            ]
        )
    else:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_wo_scaler(),
                    params.numerical_features,
                ),
            ]
        )

    return transformer


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]

