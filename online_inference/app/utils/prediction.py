from typing import Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def predict(data: pd.DataFrame,
        transformer: ColumnTransformer,
        model: Union[LogisticRegression, RandomForestClassifier]):

    transformed_data = pd.DataFrame(transformer.transform(data))
    predicts = model.predict(transformed_data)

    return predicts
