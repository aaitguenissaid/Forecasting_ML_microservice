import os
import pytest
from pathlib import Path
from config.config import (
    get_model_id,
    get_artifact_path,
    get_model_name,
    get_model_path,
    get_model_uri,
    get_model_versions,
    get_latest_model_version,
    MODEL_ID_SIZE,
    MODEL_BASE_NAME,
    MODEL_DIR,
)

@pytest.fixture
def mock_model_dir(tmp_path):
    """Fixture to create a temporary model directory for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model-0001").mkdir()
    (model_dir / "model-0001" / "1").mkdir()
    (model_dir / "model-0001" / "2").mkdir()
    return model_dir

def test_get_model_id():
    assert get_model_id("1") == "0001"
    assert get_model_id("123") == "0123"
    assert get_model_id("1234") == "1234"

def test_get_artifact_path():
    assert get_artifact_path("1") == "model-0001"
    assert get_artifact_path("123") == "model-0123"

def test_get_model_name():
    assert get_model_name("1") == f"{MODEL_BASE_NAME}-0001"
    assert get_model_name("123") == f"{MODEL_BASE_NAME}-0123"

def test_get_model_path(mock_model_dir):
    model_name = "model-0001"
    model_version = "1"
    expected_path = os.path.join(mock_model_dir, model_name, model_version)
    assert get_model_path(model_dir=mock_model_dir, model_name=model_name, model_version=model_version) == expected_path

def test_get_model_uri():
    run_id = "12345"
    artifact_path = "model-0001"
    expected_uri = "runs:/12345/model-0001"
    assert get_model_uri(run_id, artifact_path) == expected_uri

def test_get_model_versions(mock_model_dir):
    model_name = "model-0001"
    versions = get_model_versions(mock_model_dir, model_name)
    assert versions == ["1", "2"]

def test_get_latest_model_version(mock_model_dir):
    model_name = "model-0001"
    latest_version = get_latest_model_version(mock_model_dir, model_name)
    assert latest_version == "2"

def test_get_model_versions_no_model(mock_model_dir):
    with pytest.raises(FileNotFoundError, match="No model found at:"):
        get_model_versions(mock_model_dir, "non_existent_model")

def test_get_model_versions_no_versions(mock_model_dir):
    model_name = "model-0002"
    (mock_model_dir / model_name).mkdir()
    with pytest.raises(ValueError, match="No versions found for model"):
        get_model_versions(mock_model_dir, model_name)