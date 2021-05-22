import os

from src.entities import PredictingPipelineParams
from src.predict_pipeline import predict_pipeline


def test_eval_pipeline(predict_pipeline_params: PredictingPipelineParams,
                       output_predictions_path: str,
                       train_synthetic):

    predictions = predict_pipeline(predict_pipeline_params)

    assert os.path.exists(output_predictions_path)
    assert 200 == predictions.shape[0]
    assert {0, 1} == set(predictions.iloc[:, 0])