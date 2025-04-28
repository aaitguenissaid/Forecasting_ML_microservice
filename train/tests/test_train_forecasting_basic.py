import pytest
import pandas as pd
from prophet import Prophet
import ray
from train_forecasting_basic import (
    prep_store_data,
    rename_cols,
    train_test_split,
    define_model,
    fit_model,
    predict,
    prep_train_predict,
)

@pytest.fixture
def sample_dataframe():
    data = {
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Store": [4, 4, 4],
        "Open": [1, 1, 1],
        "Sales": [100, 200, 300]
    }
    return pd.DataFrame(data)

def test_prep_store_data(sample_dataframe):
    result = prep_store_data(sample_dataframe, store_id=4, store_open=1)
    assert not result.empty
    assert result["Store"].iloc[0] == 4

def test_rename_cols(sample_dataframe):
    rename_cols(sample_dataframe, "Sales")
    assert "ds" in sample_dataframe.columns
    assert "y" in sample_dataframe.columns

def test_train_test_split(sample_dataframe):
    rename_cols(sample_dataframe, "Sales")
    train, test, train_index = train_test_split(sample_dataframe, train_fraction=0.67)
    assert len(train) == 2
    assert len(test) == 1

def test_define_model():
    seasonality = {"yearly": True, "weekly": True, "daily": False}
    model = define_model(seasonality)
    assert isinstance(model, Prophet)

def test_fit_model(sample_dataframe):
    rename_cols(sample_dataframe, "Sales")
    model = define_model({"yearly": True, "weekly": True, "daily": False})
    trained_model = fit_model(model, sample_dataframe)
    assert isinstance(trained_model, Prophet)

def test_predict(sample_dataframe):
    rename_cols(sample_dataframe, "Sales")
    model = define_model({"yearly": True, "weekly": True, "daily": False})
    trained_model = fit_model(model, sample_dataframe)
    future = trained_model.make_future_dataframe(periods=1)
    predictions = predict(trained_model, future)
    assert "yhat" in predictions.columns

def test_prep_train_predict(sample_dataframe):
    ray.init(ignore_reinit_error=True)  # Initialize Ray
    seasonality = {"yearly": True, "weekly": True, "daily": False}
    result = prep_train_predict.remote(
        sample_dataframe, store_id=4, store_open=1, train_fraction=0.67, seasonality=seasonality
    )
    model, df_predicted, df_train, df_test, train_index = ray.get(result)
    assert isinstance(model, Prophet)
    assert not df_predicted.empty
    assert not df_train.empty
    assert not df_test.empty
    ray.shutdown()  # Shutdown Ray