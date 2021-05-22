from typing import NoReturn, List, Tuple

import pytest
import pandas as pd
from faker import Faker

from src.entities import FeatureParams, LogRegParams, RandomForestParams, \
    TrainingPipelineParams, SplittingParams, PathParams, PredictingPipelineParams
from src.features.build_features import get_target, build_transformer
from src.train_pipeline import train_pipeline

ROW_NUMS = 200


@pytest.fixture(scope="session")
def synthetic_data_path() -> str:
    return "tests/synthetic_data.csv"


@pytest.fixture(scope="session")
def output_predictions_path() -> str:
    return "tests/test_predictions.csv"


@pytest.fixture(scope="session")
def load_model_path() -> str:
    return "tests/test_model.pkl"


@pytest.fixture(scope="session")
def metric_path() -> str:
    return "tests/test_metrics.json"


@pytest.fixture(scope="session")
def load_transformer_path() -> str:
    return "tests/test_transformer.pkl"


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"


@pytest.fixture(scope="session")
def normalize_numerical_true() -> bool:
    return True


@pytest.fixture(scope="session")
def normalize_numerical_false() -> bool:
    return False


@pytest.fixture(scope="session")
def feature_params_w_norm(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        normalize_numerical_true: bool
) -> FeatureParams:
    fp = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        normalize_numerical=normalize_numerical_true
    )
    return fp


@pytest.fixture(scope="session")
def feature_params_wo_norm(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        normalize_numerical_false: bool
) -> FeatureParams:
    fp = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        normalize_numerical=normalize_numerical_false
    )
    return fp


@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(ROW_NUMS)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(ROW_NUMS)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(ROW_NUMS)],
        "chol": [fake.pyint(min_value=126, max_value=555) for _ in range(ROW_NUMS)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(ROW_NUMS)],
        "thalach": [fake.pyint(min_value=71, max_value=202) for _ in range(ROW_NUMS)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(ROW_NUMS)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(ROW_NUMS)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(ROW_NUMS)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(ROW_NUMS)],
        "target": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)]
    }

    return pd.DataFrame(data=df)


@pytest.fixture(scope="package")
def lr_training_params() -> LogRegParams:
    model = LogRegParams(
        model_type="LogisticRegression",
        penalty="l2",
        tol=1e-4,
        random_state=21
    )
    return model


@pytest.fixture(scope="package")
def rf_training_params() -> RandomForestParams:
    model = RandomForestParams(
        model_type="RandomForestClassifier",
        n_estimators=20,
        max_depth=4,
        random_state=21
    )
    return model


@pytest.fixture(scope="package")
def prepared_dataframe(
        synthetic_data: pd.DataFrame, feature_params_w_norm: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    transformer = build_transformer(feature_params_w_norm)
    transformer.fit(synthetic_data)

    transformed_features = transformer.transform(synthetic_data)
    target = get_target(synthetic_data, feature_params_w_norm)
    return target, transformed_features


@pytest.fixture(scope="package")
def train_pipeline_params(
    synthetic_data_path: str,
    load_model_path: str,
    metric_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    normalize_numerical_true: bool,
    target_col: str,
    load_transformer_path: str,
    lr_training_params: LogRegParams
) -> TrainingPipelineParams:

    train_pipeline_parms = TrainingPipelineParams(
        path_config=PathParams(
            input_data_path=synthetic_data_path,
            metric_path=metric_path,
            output_model_path=load_model_path,
            output_transformer_path=load_transformer_path,
        ),

        splitting_params=SplittingParams(val_size=0.2, random_state=21),

        feature_params=FeatureParams(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target_col=target_col,
            normalize_numerical=normalize_numerical_true
        ),

        train_params=lr_training_params
    )
    return train_pipeline_parms


@pytest.fixture(scope="package")
def predict_pipeline_params(
    synthetic_data_path: str,
    load_model_path: str,
    output_predictions_path: str,
    load_transformer_path: str,
) -> PredictingPipelineParams:

    pred_pipeline_params = PredictingPipelineParams(
        input_data_path=synthetic_data_path,
        output_data_path=output_predictions_path,
        pipeline_path=load_transformer_path,
        model_path=load_model_path,
    )
    return pred_pipeline_params


@pytest.fixture(scope="package")
def train_synthetic(train_pipeline_params: TrainingPipelineParams) -> NoReturn:
    train_pipeline(train_pipeline_params)


