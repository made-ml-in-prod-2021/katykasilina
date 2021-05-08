import os

from src.entities import TrainingPipelineParams
from src.train_pipeline import train_pipeline


def test_train_pipeline(
    train_pipeline_params: TrainingPipelineParams,
    metric_path: str,
    load_model_path: str,
    load_transformer_path: str
):

    metrics = train_pipeline(train_pipeline_params)
    assert 0 < metrics["roc_auc_score"] <= 1
    assert 0 < metrics["accuracy_score"] <= 1
    assert 0 < metrics["f1_score"] <= 1

    assert os.path.exists(load_transformer_path)
    assert os.path.exists(metric_path)
    assert os.path.exists(load_model_path)