import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR, DATASET_RAW_DATA_FILE_NAME, \
    DATASET_RAW_TARGET_FILE_NAME, DATA_PROCESSED_DIR, MODELS_DIR_DEFAULT


with DAG(
        "train",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:


    data_path_str = DATA_RAW_DIR[1:] + '/' + DATASET_RAW_DATA_FILE_NAME
    target_path_str = DATA_RAW_DIR[1:] + '/' + DATASET_RAW_TARGET_FILE_NAME

    wait_for_train_data = FileSensor(
        task_id='wait_for_train_data',
        poke_interval=30,
        filepath=data_path_str
    )

    wait_for_target = FileSensor(
        task_id='wait_for_target_data',
        poke_interval=30,
        filepath=target_path_str #'data/raw/{{ ds }}/target.csv'
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir {DATA_RAW_DIR} --out_dir {DATA_PROCESSED_DIR} ",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"]
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input-dir {DATA_PROCESSED_DIR}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input-dir {DATA_PROCESSED_DIR} --out_dir {MODELS_DIR_DEFAULT}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input-dir {DATA_PROCESSED_DIR} --input-model-dir {MODELS_DIR_DEFAULT}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"]
    )

    [wait_for_train_data, wait_for_target] >> preprocess >> split >> train >> validate