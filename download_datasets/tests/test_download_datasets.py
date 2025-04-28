import pytest
from pathlib import Path
from unittest.mock import patch
from download_datasets.main import download_kaggle_dataset, main

@pytest.fixture
def tmp_data_dir(tmp_path):
    # tmp_path is a built-in pytest fixture
    return tmp_path

@patch("download_datasets.main.kaggle.api.dataset_download_files")
def test_all_files_present(mock_download, tmp_data_dir):
    for fname in ["train.csv", "test.csv", "store.csv"]:
        (tmp_data_dir / fname).touch()
    download_kaggle_dataset(tmp_data_dir)
    mock_download.assert_not_called()

@patch("download_datasets.main.kaggle.api.dataset_download_files")
def test_partial_files_trigger_download(mock_download, tmp_data_dir):
    (tmp_data_dir / "train.csv").touch()  # only one file exists
    download_kaggle_dataset(tmp_data_dir)
    mock_download.assert_called_once()

@patch("download_datasets.main.kaggle.api.dataset_download_files")
def test_no_files_present_triggers_download(mock_download, tmp_data_dir):
    download_kaggle_dataset(tmp_data_dir)
    mock_download.assert_called_once()

@patch("download_datasets.main.download_kaggle_dataset")
def test_main(mock_download_kaggle_dataset):
    mock_download_kaggle_dataset.return_value = None
    main()
    mock_download_kaggle_dataset.assert_called_once()