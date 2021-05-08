import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import read_data, load_pkl_file


def test_read_data(synthetic_data_path: str):

    df = read_data(synthetic_data_path)

    assert isinstance(df, pd.DataFrame)
    assert (200, 15) == df.shape


def test_load_pkl_file(load_model_path: str):
    model = load_pkl_file(load_model_path)
    assert isinstance(model, RandomForestClassifier) or isinstance(model, LogisticRegression)




