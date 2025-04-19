import os
from pydantic import BaseModel
import datetime
import pandas as pd

 # REQUEST UTILITIES
class ForecastRequest(BaseModel):
    store_id: str
    begin_date: str | None = None
    end_date: str | None = None

def create_forecast_index(begin_date: str = None, end_date: str = None) -> pd.DataFrame:
    # Convert forecast begin date
    if begin_date == None:
        begin_date = datetime.datetime.now().replace(tzinfo=None)
    else:
        begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
        
    # Convert forecast end date
    if end_date == None: 
        end_date = begin_date + datetime.timedelta(days=7)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)

    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')
    # Format for Prophet to consume
    return pd.DataFrame({'ds': forecast_index})

    

def create_forecast_index_days(days: str = None) -> pd.DataFrame:
    if days == None: 
        days = 7
    else:
        days = int(days)
    
    begin_date = datetime.datetime.now().replace(tzinfo=None)    
    end_date = begin_date + datetime.timedelta(days=days)

    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')

    # Format for Prophet to consume
    return pd.DataFrame({'ds': forecast_index})


def get_latest_model_version_from_folder(model_registry_path, model_name) -> str:
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
    return str(latest_version)
