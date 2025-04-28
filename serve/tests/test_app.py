from typing import Any
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Import your FastAPI app
from src.app import app, services

# Dummy classes to replace the real ones
class DummyRegistry:
    def check_mlflow_health(self):
        return "dummy-ok"

class DummyModel:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return a DataFrame with the same index
        return pd.DataFrame({
            "ds": df["ds"],
            "yhat": [99] * len(df)
        })

class DummyLoader:
    def model_exists(self, store_id: str) -> bool:
        # Only store "1" exists
        return store_id == "1"
    def get_production_model(self, store_id: str) -> Any:
        return DummyModel()

@pytest.fixture(autouse=True)
def mock_services(monkeypatch):
    """
    Before each test, patch the MlflowRegistryClient and LocalModelLoader
    used in our app's startup, so no real external calls are made.
    """
    # Patch the classes in the module namespace
    monkeypatch.setattr("src.app.MlflowRegistryClient", lambda: DummyRegistry())
    monkeypatch.setattr("src.app.LocalModelLoader", lambda: DummyLoader())
    yield
    services.clear()  # clean up between tests

@pytest.fixture
def client():
    # Use TestClient as context manager to trigger lifespan events
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body

def test_health_check(client):
    r = client.get("/health/")
    assert r.status_code == 200
    body = r.json()
    assert body["serviceStatus"] == "OK"
    assert body["modelTrackingHealth"] == "dummy-ok"

def test_forecast_days_success(client):
    payload = {"store_id": "1", "days": 2}
    r = client.post("/forecast_days/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 1
    rec = data[0]
    assert rec["request"] == payload
    assert len(rec["forecast"]) == 2
    assert all(d["value"] == 99 for d in rec["forecast"])

def test_forecast_days_not_found(client):
    payload = {"store_id": "2", "days": 2}
    r = client.post("/forecast_days/", json=payload)
    assert r.status_code == 404

def test_forecast_interval_success(client):
    payload = [
        {"store_id": "1", "begin_date": "2023-01-01", "end_date": "2023-01-02"}
    ]
    r = client.post("/forecast_interval/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 1
    rec = data[0]
    assert rec["request"] == payload[0]
    assert len(rec["forecast"]) == 2
    assert all(d["value"] == 99 for d in rec["forecast"])

def test_forecast_interval_not_found(client):
    payload = [
        {"store_id": "3", "begin_date": "2023-01-01", "end_date": "2023-01-02"}
    ]
    r = client.post("/forecast_interval/", json=payload)
    assert r.status_code == 404
