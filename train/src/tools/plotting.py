import os
import logging
from matplotlib import pyplot as plt
import pandas as pd
from prophet import Prophet, serialize
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

FIGSIZE = (15, 7)

def plot_forecast(df_train: pd.DataFrame, df_test: pd.DataFrame, predicted: pd.DataFrame, train_index: int, results_path: str) -> None:
    df_test["ds"] = df_test["ds"].astype(str)
    predicted["ds"] = predicted["ds"].astype(str)
    df_train["ds"] = df_train["ds"].astype(str)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    df_test.plot(
        x="ds",
        y="y",
        ax=ax,
        label="Truth",
        linewidth=1,
        markersize=5,
        color="tab:blue",
        alpha=0.9,
        marker="o"
    )
    predicted.plot(
        x="ds",
        y="yhat",
        ax=ax,
        label="Prediction + 95% CI",
        linewidth=2,
        markersize=5,
        color="red"
    )
    ax.fill_between(
        x=predicted["ds"],
        y1=predicted["yhat_upper"],
        y2=predicted["yhat_lower"],
        alpha=0.15,
        color="red"
    )
    df_train.iloc[train_index-100:].plot(
        x="ds",
        y="y",
        ax=ax,
        color='tab:blue',
        label="_nolegend_",
        alpha=0.5,
        marker="o",
    )
    
    yticks = plt.gca().get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels(["{:,.0f}".format(x) for x in yticks])
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "store_data_forecast.png"))
    plt.close()

def plot_results(model: Prophet, forecast_prophet: pd.DataFrame, df_test: pd.DataFrame, future_period: int, store_id: int) -> None:
    # Convert 'ds' column to Python datetime objects for compatibility with matplotlib
    df_test["ds"] = pd.to_datetime(df_test["ds"])
    forecast_prophet["ds"] = pd.to_datetime(forecast_prophet["ds"])

    forecast_plot = model.plot(forecast_prophet, figsize=FIGSIZE)

    axes = forecast_plot.gca()
    last_training_date = forecast_prophet['ds'].iloc[-future_period]
    axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')
    plt.title(f"Daily sales of store id: {store_id}")
    plt.plot(df_test['ds'], df_test['y'], 'ro', markersize=3, label='True Test Data')
    plt.legend()
    plt.show()

def plot_seasonal_decompose(decompose : DecomposeResult, plot: bool = True) -> None:
    if(plot):
        fig = decompose.plot()
        fig.set_size_inches(FIGSIZE)
        fig.tight_layout()
        plt.show()
    else:
        logging.INFO("plot_seasonal_decompose deactivated")

def plot_df(df: pd.DataFrame) -> None:
    ax = df.set_index('ds').plot(figsize=FIGSIZE)
    ax.set_xlabel('Date')
    plt.show()


