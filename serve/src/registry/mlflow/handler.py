import mlflow
from mlflow.client import MlflowClient
from pprint import pprint

from tqdm import tqdm
from config.config import TRACKING_URI

class MlflowRegistryClient:
    def __init__(self) -> None:
        self.client = MlflowClient(tracking_uri=TRACKING_URI)
        mlflow.set_tracking_uri(TRACKING_URI) 

    # TODO review Healthcheck
    # 2025-04-22 18:28:19,085 - urllib3.connectionpool - WARNING - Retrying (Retry(total=6, connect=6, read=7, redirect=7, status=7)) after connection bro
    # ken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at 0
    # x7f8a7efc9e50>: Failed to establish a new connection: [Errno 111] Connecti
    # on refused')': /api/2.0/mlflow/experiments/search
    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()
            for rm in tqdm(experiments):
                pprint(dict(rm), indent=4)
            return 'Service returning experiments'
        except:
            return 'Error calling MLFlow'

    # # TODO this function does not need to use the mlflow server it is independent from it. though the class is dependent.
    # # TODO: set the function to return the production model instead of the last trained model
    # def get_production_model(self, store_id : str) -> PyFuncModel:
    #     model_name = get_model_name(store_id)
    #     latest_versions = get_latest_model_version(MODEL_DIR, model_name=model_name)        
    #     model = mlflow.prophet.load_model(model_uri=get_model_path(MODEL_DIR, model_name, latest_versions))
    #     return model
