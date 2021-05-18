import pytest
from fastapi.testclient import TestClient

from make_prediction import *
from utils.prediction import predict
from utils.datamodels_cheker import InputDataModel
from utils.utils import load_pkl_file

SOURCE_DIR = "sources"


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_data():
    features = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 'fbs': 0,
        'restecg': 1, 'thalach': 182, 'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    label = predict(
        data=InputDataModel(**features).convert_to_pandas(),
        transformer=load_pkl_file(os.path.join(SOURCE_DIR, "transformer.pkl")),
        model=load_pkl_file(os.path.join(SOURCE_DIR, "model.pkl")),
    )
    return features, label


def test_load_on_startup(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() is True


@pytest.mark.parametrize(
    ["feature", "wrong_val", "expected_status_code"],
    [
        pytest.param("trestbps", 600, 400),
        pytest.param("sex", 6, 400),
        pytest.param("cp", 23, 400),
        pytest.param("age", 180, 400),
    ]
)
def test_predict_raise_400(
        feature, wrong_val, expected_status_code,
        test_data, client
):
    broken_data = test_data[0].copy()
    broken_data[feature] = wrong_val

    resp = client.get("/predict", json=broken_data)
    assert resp.status_code == expected_status_code


def test_predict_label(test_data, client):
    expected_status_code = 200
    expected_value = {"label": test_data[1]}
    resp = client.get("/predict", json=test_data[0])

    assert resp.status_code == expected_status_code
    assert resp.json() == expected_value