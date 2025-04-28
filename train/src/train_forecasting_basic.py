import os
import logging
from pathlib import Path
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error, 
                             median_absolute_error)
import mlflow
from mlflow import MlflowClient
import ray 
from tqdm import tqdm
from config.config import RESULTS_DIR, MODEL_DIR, DATA_DIR, TRAIN_FILE, TRACKING_URI, LOG_FORMAT, get_model_name, get_model_path, get_artifact_path, get_model_uri

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


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("prophet_models_14042025")
    mlflow.autolog()
    client = MlflowClient(tracking_uri=TRACKING_URI)
    logging.info("Defined MLflowClient and set tracking URI.")

    df = pd.read_csv(TRAIN_FILE)

    store_ids = df['Store'].unique()[:10]

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
