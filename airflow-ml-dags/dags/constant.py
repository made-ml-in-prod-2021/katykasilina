import os
import datetime
from airflow.models import Variable


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["nonothanx@yandex.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

DATA_RAW_DIR = "/data/raw/{{ ds }}"
DATA_VOLUME_DIR = Variable.get("data_path")

DATASET_RAW_DATA_FILE_NAME = "data.csv"
DATASET_RAW_TARGET_FILE_NAME = "target.csv"

DATA_PROCESSED_DIR = "data/processed/{{ ds }}"

MODEL_FILE_NAME = Variable.get("model_name_train")
MODELS_DIR_DEFAULT = Variable.get("models_path_train")

MODEL_FILE_NAME_PREDICTION = Variable.get("model_name_prediction")
MODELS_PATH_PREDICTION = Variable.get("models_path_prediction")

DATASET_PREDICTION_FILE_NAME = "data.csv"

DATA_PREDICTION_DIR = "data/predictions/{{ ds }}"