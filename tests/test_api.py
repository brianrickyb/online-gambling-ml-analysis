from contextlib import asynccontextmanager

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api as api
from src.model import build_pipeline
from src.monitor import summarize
from tests.test_model import HAM, SPAM


@pytest.fixture
def client(tmp_path, monkeypatch):
    X = pd.Series((SPAM + HAM) * 5)
    y = pd.Series(([1] * len(SPAM) + [0] * len(HAM)) * 5)
    toy_model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)

    monkeypatch.setattr(api, "PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")
    # Skip the startup that loads the real model from disk.
    api.app.router.lifespan_context = _no_lifespan
    api.app.dependency_overrides[api.get_model] = lambda: toy_model
    api.app.dependency_overrides[api.get_model_info] = lambda: {"version": "test", "threshold": 0.5}
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@asynccontextmanager
async def _no_lifespan(app):
    yield


def test_predict_endpoint(client, tmp_path):
    r = client.post("/predict", json={"texts": ["gacor di alexis17", "mobilnya keren bang"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test"
    assert [p["label"] for p in body["predictions"]] == ["judol", "normal"]
    log_lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(log_lines) == 2
    assert "alexis" not in log_lines[0]  # raw text must not be logged


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Judol" in r.text


def test_monitor_flags_drift():
    log = pd.DataFrame({
        "label": ["judol"] * 8 + ["normal"] * 2,
        "score": [0.5] * 10,
        "model_version": ["v1"] * 10,
    })
    summary, warnings = summarize(log, {"judol_rate": 0.33})
    assert summary["judol_rate"] == 0.8
    assert len(warnings) == 2
