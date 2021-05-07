from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.entities.train_params import LogRegParams, RandomForestParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: Union[LogRegParams, RandomForestParams],
                ) -> SklearnClassifierModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            max_depth=train_params.max_depth,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty=train_params.penalty,
            tol=train_params.tol,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model

