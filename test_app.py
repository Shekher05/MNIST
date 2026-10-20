import numpy as np
import pytest

from app import app
from src.data import load_data_mnist_format


@pytest.fixture(scope="module")
def client():
    return app.test_client()


def test_index_serves_drawing_canvas(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<canvas" in r.data


def test_predict_returns_digit_and_confidence(client):
    r = client.post("/predict", json={"pixels": [0.0] * 784})
    assert r.status_code == 200
    assert r.json["digit"] in range(10)
    assert 0.0 <= r.json["confidence"] <= 1.0


@pytest.mark.parametrize("pixels", [[0.0] * 783, [0.0] * 785, "abc", None])
def test_predict_rejects_bad_payload(client, pixels):
    assert client.post("/predict", json={"pixels": pixels}).status_code == 400


def test_predict_matches_mnist_labels(client):
    _, (X_test, y_test) = load_data_mnist_format()
    hits = sum(
        client.post("/predict", json={"pixels": X_test[i].tolist()}).json["digit"] == y_test[i]
        for i in range(20)
    )
    assert hits >= 17
