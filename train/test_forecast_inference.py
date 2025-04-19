import os
import datetime
import pandas as pd
import mlflow
from mlflow.pyfunc import PyFuncModel
from mlflow.client import MlflowClient
from config.paths import get_model_name


# tracking_uri = os.getenv(["MLFLOW_TRACKING_URI"])


tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri) 

def create_forecast_index(begin_date: str = None, end_date: str = None) -> pd.DataFrame:
    if begin_date == None:
        begin_date = datetime.datetime.now().replace(tzinfo=None)
    else:
        begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
    
    if end_date == None:
        end_date = begin_date + datetime.timedelta(days=7)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
    
    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')

    return pd.DataFrame({'ds': forecast_index})



def get_latest_model_version_from_folder(model_registry_path, model_name) -> pd.DataFrame:
    model_path = os.path.join(model_registry_path, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at: {model_path}")

    # List subfolders which represent version numbers
    version_folders = [
        name for name in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, name)) and name.isdigit()
    ]

    if not version_folders:
        raise ValueError(f"No versions found for model '{model_name}'.")

    # Convert folder names to integers and get the max
    latest_version = max(int(v) for v in version_folders)
    return latest_version


def get_production_model(store_id : str) -> PyFuncModel:
    model_name = get_model_name(store_id)
    # Retrieve the latest version of the model
    latest_versions = get_latest_model_version_from_folder("models", model_name=model_name)
    print("latest_versions", latest_versions)
    print("model_name", model_name)
    model = mlflow.prophet.load_model(model_uri=f"models/{model_name}/{latest_versions}")
    return model


for store_id in ["1", "5", "4", "10"]:
    model = get_production_model(store_id)
    forecast_index = create_forecast_index(begin_date="2023-03-01T00:00:00Z", end_date="2023-03-07T00:00:00Z")
    print(model.predict(forecast_index))
    