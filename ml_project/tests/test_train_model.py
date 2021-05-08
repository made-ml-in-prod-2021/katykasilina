from typing import NoReturn, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.entities import LogRegParams, RandomForestParams
from src.models import train_model


def test_lr_train_model(
    lr_training_params: LogRegParams,
    prepared_dataframe: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_dataset = prepared_dataframe
    model = train_model(transformed_dataset, target, lr_training_params)

    check_is_fitted(model)
    assert isinstance(model, LogisticRegression)


def test_rf_train_model(
    rf_training_params: RandomForestParams,
    prepared_dataframe: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_dataset = prepared_dataframe
    model = train_model(transformed_dataset, target, rf_training_params)

    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)
