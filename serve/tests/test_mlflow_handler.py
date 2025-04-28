import pytest
from unittest.mock import MagicMock, patch
from registry.mlflow.handler import MlflowRegistryClient

def test_check_mlflow_health_success():
    client_mock = MagicMock()
    client_mock.search_experiments.return_value = [
        {'experiment_id': '1', 'name': 'Test Experiment'}
    ]
    with patch('registry.mlflow.handler.MlflowClient', return_value=client_mock):
        registry_client = MlflowRegistryClient()
        result = registry_client.check_mlflow_health()
        assert result == 'Service returning experiments'

def test_check_mlflow_health_failure():
    client_mock = MagicMock()
    client_mock.search_experiments.side_effect = Exception("Connection error")
    with patch('registry.mlflow.handler.MlflowClient', return_value=client_mock):
        registry_client = MlflowRegistryClient()
        result = registry_client.check_mlflow_health()
        assert result == 'Error calling MLFlow'