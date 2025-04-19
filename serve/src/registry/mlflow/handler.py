import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel
from pprint import pprint
from helpers.requests import get_latest_model_version_from_folder
from helpers.paths import get_model_name, get_model_path, MODEL_DIR, TRACKING_URI
class MLFlowHandler:
    def __init__(self) -> None:
        self.client = MlflowClient(tracking_uri=TRACKING_URI)
        mlflow.set_tracking_uri(TRACKING_URI) 

    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
            return 'Service returning experiment'
        except:
            return 'Error calling MLFlow'

    # TODO: set the function to return the production model instead of the last trained model
    def get_production_model(self, store_id : str) -> PyFuncModel:
        model_name = get_model_name(store_id)
        latest_versions = get_latest_model_version_from_folder(MODEL_DIR, model_name=model_name)        
        model = mlflow.prophet.load_model(model_uri=get_model_path(MODEL_DIR, model_name, latest_versions))
        return model

