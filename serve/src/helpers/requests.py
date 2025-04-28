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
