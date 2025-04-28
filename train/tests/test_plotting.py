import pytest
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from tools.plotting import plot_forecast, plot_results, plot_seasonal_decompose, plot_df
from prophet import Prophet

@pytest.fixture
def sample_dataframe():
    data = {
        "ds": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "y": [100, 200, 300]
    }
    return pd.DataFrame(data)

def test_plot_forecast(sample_dataframe, tmp_path):
    sample_dataframe["yhat"] = [110, 210, 310]
    sample_dataframe["yhat_upper"] = [120, 220, 320]
    sample_dataframe["yhat_lower"] = [90, 190, 290]
    results_path = tmp_path / "results"
    results_path.mkdir()
    plot_forecast(sample_dataframe, sample_dataframe, sample_dataframe, 2, str(results_path))
    assert (results_path / "store_data_forecast.png").exists()

def test_plot_results(sample_dataframe):
    model = Prophet()
    model.fit(sample_dataframe)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    plot_results(model, forecast, sample_dataframe, 1, 4)

def test_plot_seasonal_decompose(sample_dataframe):
    sample_dataframe.set_index("ds", inplace=True)
    decompose = seasonal_decompose(sample_dataframe["y"], model="additive", period=1)
    plot_seasonal_decompose(decompose, plot=True)

def test_plot_df(sample_dataframe):
    plot_df(sample_dataframe)