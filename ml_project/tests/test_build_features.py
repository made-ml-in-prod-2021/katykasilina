from typing import NoReturn

import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.features import get_target, build_transformer

from src.entities import FeatureParams
from src.features import CustomStandardScaler


def test_extract_target(synthetic_data: pd.DataFrame, feature_params_w_norm: FeatureParams) -> NoReturn:
    target_df = get_target(synthetic_data, feature_params_w_norm)

    assert len(target_df) == len(synthetic_data)
    assert synthetic_data[feature_params_w_norm.target_col].equals(target_df)


def test_custom_transformer(synthetic_data: pd.DataFrame, feature_params_w_norm: FeatureParams) -> NoReturn:
    synth_data_np = synthetic_data[feature_params_w_norm.numerical_features].to_numpy()
    correct_synth_np = (synth_data_np - synth_data_np.mean(axis=0)) / synth_data_np.std(axis=0)

    scaler = CustomStandardScaler()
    scaler.fit(synth_data_np)

    custom_scaled_data = scaler.transform(synth_data_np)

    assert custom_scaled_data.shape == correct_synth_np.shape
    assert np.allclose(custom_scaled_data, correct_synth_np)


def test_build_features_pipeline_norm(
        synthetic_data: pd.DataFrame, feature_params_w_norm: FeatureParams
) -> NoReturn:
    transformer = build_transformer(feature_params_w_norm)

    transformer.fit(synthetic_data)
    check_is_fitted(transformer)

    transformed_data = transformer.transform(synthetic_data)

    synth_data_np = synthetic_data[feature_params_w_norm.numerical_features].to_numpy()
    correct_synth_np = (synth_data_np - synth_data_np.mean(axis=0)) / synth_data_np.std(axis=0)

    num_features = len(feature_params_w_norm.numerical_features)
    transformed_cols = transformed_data[:, -num_features:]

    assert np.allclose(transformed_cols, correct_synth_np)
    assert not pd.isnull(transformed_data).any().any()
    assert (synthetic_data.shape[0], 27) == transformed_data.shape


def test_build_features_pipeline_wo_norm(
        synthetic_data: pd.DataFrame, feature_params_wo_norm: FeatureParams
) -> NoReturn:
    transformer = build_transformer(feature_params_wo_norm)

    transformer.fit(synthetic_data)
    check_is_fitted(transformer)

    transformed_data = transformer.transform(synthetic_data)

    synth_data_np = synthetic_data[feature_params_wo_norm.numerical_features].to_numpy()

    num_features = len(feature_params_wo_norm.numerical_features)
    transformed_cols = transformed_data[:, -num_features:]

    assert np.allclose(transformed_cols, synth_data_np)
    assert not pd.isnull(transformed_data).any().any()
    assert (synthetic_data.shape[0], 27) == transformed_data.shape
