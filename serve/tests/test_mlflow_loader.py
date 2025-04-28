import pytest
from prophet.forecaster import Prophet
from registry.mlflow.loader import LocalModelLoader
from config.config import MODEL_DIR

@pytest.fixture
def loader():
    return LocalModelLoader(MODEL_DIR)

def _get_non_existant_ids(ids) -> list[str]:
    list_ids = list(map(int, ids))
    max_id, min_id = max(list_ids), min(list_ids)
    amplitude = 10
    res = []
    [res.append(str(n_id)) for n_id in range(min_id-amplitude, min_id)]
    [res.append(str(n_id)) for n_id in range(max_id+1, max_id+amplitude)]
    return res

def test_list_and_exists(loader):
    ids = loader.list_store_ids()
    non_existant_ids = _get_non_existant_ids(ids)
    ids = loader.list_store_ids()
    assert all(loader.model_exists(id) for id in ids)
    non_existant_ids = _get_non_existant_ids(ids)
    assert not all(loader.model_exists(n_id) for n_id in non_existant_ids)

def test_scan_empty(loader):
    assert loader.list_store_ids() != []
    assert loader.model_exists("1")

def test_get_production_model_success(loader):
    id = "1"
    assert id in loader.list_store_ids()
    model = loader.get_production_model(id)
    assert isinstance(model, Prophet)

def test_get_production_model_not_found(loader):
    non_existant_ids = _get_non_existant_ids(loader.list_store_ids())
    failed_ids = []  # Collect IDs that did not raise the exception
    for n_id in non_existant_ids:
        try:
            with pytest.raises(FileNotFoundError):
                loader.get_production_model(n_id)
        except AssertionError:
            failed_ids.append(n_id)
    # Assert that no IDs failed to raise the exception
    assert not failed_ids