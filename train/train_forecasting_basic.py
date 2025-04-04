import os
import logging
import kaggle
import pandas as pd

# config kaggle json and download the dataset.
def download_kaggle_dataset(kaggle_dataset: str = "pratyushakar/rossmann-store-sales") -> None:
    kaggle.api.dataset_download_files(kaggle_dataset, path='./data', unzip=True, quiet=False)


def main():
    data_path = "./data/"
    train_file = "train.csv"
    file_path = os.path.join(data_path, train_file)
    if os.path.exists(file_path):
        logging.info("Dataset already exists!")
    else:
        logging.info("Dataset not found, Downloading ...")
        download_kaggle_dataset()

    df = pd.read_csv(file_path)
    print(df)

if __name__ == "__main__":
    main()
