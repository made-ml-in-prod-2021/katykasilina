import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATASET_RAW_DATA_FILE_NAME, MODELS_PATH_PREDICTION, \
    MODEL_FILE_NAME_PREDICTION, DATA_VOLUME_DIR, DATA_PREDICTION_DIR, DATASET_PREDICTION_FILE_NAME

with DAG(
        "predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(2),
) as dag:


    model_path_str = MODELS_PATH_PREDICTION + '/' + MODEL_FILE_NAME_PREDICTION

    wait_for_model = FileSensor(
        task_id='wait_for_model',
        poke_interval=30,
        filepath=model_path_str
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {DATA_RAW_DIR} --input-model-dir {MODELS_PATH_PREDICTION} --out_dir {DATA_PREDICTION_DIR} --model_name {MODEL_FILE_NAME_PREDICTION} --data_file_name {DATASET_PREDICTION_FILE_NAME}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"]
    )

    wait_for_model >> predict