import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")

@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dag_bag_import(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}


def test_dag_generate_data_load(dag_bag):
    assert "data_download" in dag_bag.dags
    assert len(dag_bag.dags["data_download"].tasks) == 1


def test_dag_train_pipeline_load(dag_bag):
    assert "train" in dag_bag.dags
    assert len(dag_bag.dags["train"].tasks) == 6


def test_dag_predict_pipeline_load(dag_bag):
    assert "predict" in dag_bag.dags
    assert len(dag_bag.dags["predict"].tasks) == 2


def test_dag_download_structure(dag_bag):
    dag_flow = {
        "docker-airflow-download": [],
    }
    dag = dag_bag.dags["data_download"]
    for name, task in dag.task_dict.items():
        assert set(dag_flow[name]) == task.downstream_task_ids


def test_dag_train_structure(dag_bag):
    dag_flow = {
        "wait_for_train_data": ["docker-airflow-preprocess"],
        "wait_for_target_data": ["docker-airflow-preprocess"],
        "docker-airflow-preprocess": ["docker-airflow-split"],
        "docker-airflow-split": ["docker-airflow-train"],
        "docker-airflow-train": ["docker-airflow-validate"],
        "docker-airflow-validate": []
    }
    dag = dag_bag.dags["train"]
    for name, task in dag.task_dict.items():
        assert set(dag_flow[name]) == task.downstream_task_ids


def test_dag_predict_structure(dag_bag):
    dag_flow = {
        "wait_for_model": ["predict"],
        "predict": [],
    }
    dag = dag_bag.dags["predict"]
    for name, task in dag.task_dict.items():
        assert set(dag_flow[name]) == task.downstream_task_ids
