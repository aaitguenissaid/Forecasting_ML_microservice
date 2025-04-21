import logging
from pathlib import Path
import kaggle
from config.config import DATA_DIR, LOG_FORMAT

def download_kaggle_dataset(data_dir: str, kaggle_dataset: str = "pratyushakar/rossmann-store-sales") -> None:
    data_path = Path(data_dir)
    required_files = ["train.csv", "test.csv", "store.csv"]
    missing_files = [f for f in required_files if not (data_path / f).is_file()]

    if missing_files:
        if len(missing_files) == len(required_files):
            logging.info("No dataset files found. Downloading full dataset...")
        else:
            logging.info(f"Partial dataset found. Missing files: {missing_files}. Downloading full dataset...")

        kaggle.api.dataset_download_files(
            kaggle_dataset, path=str(data_path), unzip=True, quiet=False
        )
        logging.info("Download complete.")
    else:
        logging.info("All required dataset files are present. Skipping download.")


def main():
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    download_kaggle_dataset(DATA_DIR, kaggle_dataset = "pratyushakar/rossmann-store-sales")


if __name__ == "__main__":
    main()
