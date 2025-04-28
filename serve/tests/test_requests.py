import pytest
from datetime import date, timedelta
import pandas as pd
from helpers.requests import (
    gen_date,
    create_forecast_index_interval,
    create_forecast_index_days,
)

def tomorrow() -> date:
    return date.today() + timedelta(days=1)

def test_gen_date():
    today = date.today()
    assert gen_date() == today
    assert gen_date(today, 5) == today + timedelta(days=5)

def test_create_forecast_index_interval():
    begin_date = tomorrow()
    end_date = date.today() + timedelta(days=7)
    result = create_forecast_index_interval(begin_date, end_date)
    days = (end_date - begin_date).days + 1
    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]['ds'].date() == begin_date
    assert result.iloc[-1]['ds'].date() == end_date
    assert len(result) == days
    assert 'ds' in result.columns

def test_create_forecast_index_days():
    days = 7
    result = create_forecast_index_days(days)
    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]['ds'].date() == tomorrow()
    assert result.iloc[-1]['ds'].date() == date.today() + timedelta(days=days)
    assert len(result) == days
    assert 'ds' in result.columns