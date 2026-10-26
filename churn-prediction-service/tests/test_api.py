"""API tests. Uses a tiny model trained on synthetic data, so the real dataset is not needed."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn import api
from churn.data import clean
from churn.train import make_pipeline

EXAMPLE = api.Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        r = dict(EXAMPLE)
        r.update(tenure=int(rng.integers(0, 72)), MonthlyCharges=float(rng.uniform(20, 110)),
                 Contract=str(rng.choice(["Month-to-month", "One year", "Two year"])))
        r["TotalCharges"] = r["tenure"] * r["MonthlyCharges"]
        rows.append(r)
    X = clean(pd.DataFrame(rows))
    y = (X["Contract"] == "Month-to-month").astype(int)
    pipe = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"pipeline": pipe, "meta": {"model": "test", "trained_at": "now", "threshold": 0.3}}, path)
    api.MODEL_PATH = path
    api.load_model.cache_clear()
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_example(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["contact"] == (body["churn_probability"] >= 0.3)


def test_batch(client):
    r = client.post("/predict/batch", json=[EXAMPLE, {**EXAMPLE, "Contract": "Two year"}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_rejects_invalid_input(client):
    assert client.post("/predict", json={**EXAMPLE, "Contract": "Weekly"}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "tenure": -1}).status_code == 422
