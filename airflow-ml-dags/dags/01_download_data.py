from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constant import DEFAULT_ARGS, DATA_RAW_DIR, DATA_VOLUME_DIR, \
    DATASET_RAW_DATA_FILE_NAME, DATASET_RAW_TARGET_FILE_NAME


with DAG(
    "data_download",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0, 2),
) as dag:

    DATA_OUT_DATA = DATA_RAW_DIR + '/' + DATASET_RAW_DATA_FILE_NAME
    DATA_OUT_TARGET = DATA_RAW_DIR + '/' + DATASET_RAW_TARGET_FILE_NAME

    data_download = DockerOperator(
        image="airflow-download",
        command=f"--out_dir {DATA_RAW_DIR} --out_dir_data {DATA_OUT_DATA} --out_dir_target {DATA_OUT_TARGET}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )