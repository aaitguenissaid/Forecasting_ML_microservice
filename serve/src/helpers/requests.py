import os
from typing import Optional
from pydantic import BaseModel, Field
import datetime
from datetime import date
import pandas as pd

 # REQUEST UTILITIES
class ForecastRequestInterval(BaseModel):
    store_id:  str           = Field(..., example="16")
    begin_date: Optional[date]= Field(
        None, example=date.today() + datetime.timedelta(days=1)
    )
    end_date:   Optional[date]= Field(
        None, example=date.today() + datetime.timedelta(days=7)
    )

class ForecastRequestDays(BaseModel):
    store_id:  str           = Field(..., example="16")
    days:      Optional[int]  = Field(None, example=7)

def gen_date(given_date: date = None, days: int = 0) -> datetime:
    if given_date == None:
        given_date = date.today()
    given_date += datetime.timedelta(days=days)
    return given_date

def create_forecast_index_interval(begin_date: date = None, end_date: date = None) -> pd.DataFrame:
    if begin_date == None and end_date == None:
        begin_date = gen_date(days = 1)
        end_date = gen_date(days = 7)
    else:
        begin_date = gen_date(begin_date)
        end_date = gen_date(end_date)
    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')
    return pd.DataFrame({'ds': forecast_index})

def create_forecast_index_days(days: int = None) -> pd.DataFrame:
    if days == None:
        begin_date = gen_date(days = 1)
        end_date = gen_date(days = 7)
    else:
        begin_date = gen_date(days = 1)
        end_date = gen_date(begin_date, days = days-1)
    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')
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
