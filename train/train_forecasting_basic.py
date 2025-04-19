import os
import logging
from pathlib import Path
import kaggle
import pandas as pd
from prophet import Prophet, serialize
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error, 
                             median_absolute_error)
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import ray 
from tqdm import tqdm
from config.paths import RESULTS_DIR, MODEL_DIR, DATA_DIR, TRAIN_FILE, TRACKING_URI, get_model_name, get_model_path, get_artifact_path, get_model_uri

# Constants
FIGSIZE = (15, 7)

# config kaggle json and download the dataset.
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

def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df_store = df[(df["Store"] == store_id ) & (df["Open"] == store_open)].reset_index(drop=True)
    return df_store.sort_values("Date", ascending=True)

def rename_cols(df: pd.DataFrame, feature: str) -> None:
    df.rename(columns = {"Date": "ds", feature: "y"}, inplace=True)

def train_test_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_index = int(train_fraction*df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]
    return df_train, df_test, train_index

def get_var_name(var, namespace) -> str:
    return [name for name in namespace if namespace[name] is var][0]


def df_date_range_and_period(df: pd.DataFrame) -> int:
    var_name = get_var_name(df, locals())
    logging.info(f"{var_name}:")
    logging.info(f"Data range: {df['ds'].min().strftime('%Y-%m-%d')} -> {df['ds'].max().strftime('%Y-%m-%d')}")
    period = (df['ds'].max() - df['ds'].min()).days + 1
    logging.info(f"Total days: {period}\n")
    return period

def plot_forecast(df_train: pd.DataFrame, df_test: pd.DataFrame, predicted: pd.DataFrame, train_index: int, results_path: str) -> None:
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

def plot_results(model: Prophet, forecast_prophet: pd.DataFrame, df_test: pd.DataFrame, future_period: int , store_id: int) -> None:
    # plot the time series 
    forecast_plot = model.plot(forecast_prophet, figsize=FIGSIZE)

    # add a vertical line at the end of the training period
    axes = forecast_plot.gca()
    last_training_date = forecast_prophet['ds'].iloc[-future_period]
    axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')
    plt.title(f"Daily sales of store id: {store_id}")
    # plot true test data for the period after the red line
    plt.plot(df_test['ds'], df_test['y'],'ro', markersize=3, label='True Test Data')

    # show the legend to distinguish between the lines
    plt.legend()

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


def extract_params(model) -> dict:
    return {attr: getattr(model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

def define_model(seasonality: dict, interval_width: float = 0.95) -> Prophet:
    model = Prophet(
        yearly_seasonality = seasonality["yearly"],
        weekly_seasonality = seasonality["weekly"],
        daily_seasonality = seasonality["daily"],
        interval_width = interval_width
    )
    return model

def fit_model(model: Prophet, df_train:pd.DataFrame) -> Prophet:
    model.fit(df_train)
    return model

def predict(model: Prophet, df_test:pd.DataFrame) -> pd.DataFrame:
    df_predicted = model.predict(df_test)
    return df_predicted

@ray.remote(num_returns=5)
def prep_train_predict(
    df: pd.DataFrame,
    store_id: int,
    store_open: int=1,
    train_fraction: float=2/3,
    seasonality: dict={'yearly': True, 'weekly': True, 'daily': False}
) -> tuple[Prophet, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    df = prep_store_data(df, store_id=store_id, store_open=store_open) # maxi_val= 942
    rename_cols(df, "Sales")
    df_train, df_test, train_index = train_test_split(df, train_fraction=train_fraction)
    model = define_model(seasonality, interval_width = 0.95)
    model = fit_model(model, df_train)
    df_predicted = predict(model, df_test)
    return model, df_predicted, df_train, df_test, train_index


def assign_alias_to_stage(client, model_name, alias, version):
    """
    Assign an alias to the latest version of a registered model within a specified stage.

    :param model_name: The name of the registered model.
    :param stage: The stage of the model version for which the alias is to be assigned. Can be
                "Production", "Staging", "Archived", or "None".
    :param alias: The alias to assign to the model version.
    :return: None
    """
    # latest_mv = client.get_latest_versions(model_name, stages=[stage])[0]
    client.set_registered_model_alias(model_name, alias, version)


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    kaggle_dataset = "pratyushakar/rossmann-store-sales"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("prophet_models_14042025")
    mlflow.autolog()
    client = MlflowClient(tracking_uri=TRACKING_URI)
    logging.info("Defined MLflowClient and set tracking URI.")

    download_kaggle_dataset(DATA_DIR, kaggle_dataset)
    df = pd.read_csv(TRAIN_FILE)

    store_ids = df['Store'].unique()

    ray.init(num_cpus=4, include_dashboard=True)
    df_id = ray.put(df)

    seasonality = {
        "yearly": True,
        "weekly": True,
        "daily": False,
    }

    with mlflow.start_run(run_name=f"prophet-stores") as run:
        run_id = run.info.run_id

        obj_refs = [prep_train_predict.remote(df_id, store_id) for store_id in tqdm(store_ids)]
        model_obj_refs, pred_obj_refs, train_obj_refs, test_obj_refs, train_index_obj_refs = map(
            list, zip(*obj_refs)
        )

        ray_results = {
            "models": ray.get(model_obj_refs),
            "predictions": ray.get(pred_obj_refs),
            "train_data": ray.get(train_obj_refs),
            "test_data": ray.get(test_obj_refs),
            "train_indices": ray.get(train_index_obj_refs)
        }

        for model, df_train, df_test, df_pred, store_id in zip(
            ray_results['models'],
            ray_results['train_data'],
            ray_results['test_data'],
            ray_results['predictions'],
            store_ids
        ):
            df_y = df_test['y']
            df_yhat = df_pred['yhat']
            model_name = get_model_name(store_id)
            ARTIFACT_PATH = get_artifact_path(store_id)
            with mlflow.start_run(run_name=f"m-store-{store_id:04}", nested=True) as sub_run:
                mlflow.prophet.log_model(
                    model, 
                    artifact_path=ARTIFACT_PATH, 
                    # registered_model_name=model_name,   
                    signature=False, 
                    input_example=df_train.head()
                ) # or provide signature without df_train
                mlflow.log_params(seasonality)
                mlflow.log_metrics({
                    'rmse': mean_squared_error(y_true=df_y, y_pred=df_yhat),
                    'mean_abs_perc_error': mean_absolute_percentage_error(y_true=df_y, y_pred=df_yhat),
                    'mean_abs_error': mean_absolute_error(y_true=df_y, y_pred=df_yhat),
                    'median_abs_error': median_absolute_error(y_true=df_y, y_pred=df_yhat)
                })
                model_uri = get_model_uri(run_id, ARTIFACT_PATH)
                model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

                logging.info(f"Model registered for store {store_id}")

                # create "champion" alias for version 1 of model "example-model"
                mlflow.prophet.save_model(
                    pr_model=model,
                    path=get_model_path(MODEL_DIR, model_name, model_details.version), # This is the local folder where model is saved
                    # signature=signature,
                    # input_example=df
                ) # conda_env=None, code_paths=None, mlflow_model=None, signature: mlflow.models.signature.ModelSignature = None, input_example: Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix, str, bytes, tuple] = None, pip_requirements=None, extra_pip_requirements=None, metadata=None)

                # TODO set a condition to promote a model to production. 
                # It would be great to look at the metrics anc compare the production model with the current model, 
                # and update it in case the current one is better.

                # TODO: use prophet metrics to evaluate models. and cross_validation.
                 
                logging.info(f"Model transitioned to prod stage: {model_details.name}, {'Production'}, {model_details.version}")

    ray.shutdown()

if __name__ == "__main__":
    main()
