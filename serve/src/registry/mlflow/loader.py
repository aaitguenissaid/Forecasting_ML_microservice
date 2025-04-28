from typing import Set
import mlflow
from mlflow.pyfunc import PyFuncModel
from config.config import (
    MODEL_DIR,
    get_latest_model_version,
    get_model_name,
    get_model_path,
)
from prophet.forecaster import Prophet

class LocalModelLoader:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._model_ids: Set[str] = self._scan_models()

    def _scan_models(self) -> Set[str]:   
        model_ids = set()
        if not self.model_dir.is_dir():
            return model_ids

        for folder in self.model_dir.iterdir():
            if not folder.is_dir():
                continue
            try:
                # expect folder names like "model-name-0005"
                store_id = str(int(folder.name.rsplit("-", 1)[-1]))
                # verify that there's at least one version inside
                if any(child.is_dir() for child in folder.iterdir()):
                    model_ids.add(store_id)
            except Exception:
                continue
        return model_ids
    
    def list_store_ids(self) -> list[str]:
        return sorted(self._model_ids)

    def model_exists(self, store_id: str) -> bool:
        return store_id in self._model_ids
    
    def get_production_model(self, store_id: str) -> Prophet:
        """
        Always load the latest version of the store’s model from disk.
        """
        if store_id not in self._model_ids:
            raise FileNotFoundError(f"No model found for store_id={store_id} in {self.model_dir!r}")
        
        model_name = get_model_name(store_id)
        version = get_latest_model_version(self.model_dir, model_name=model_name)
        uri = get_model_path(self.model_dir, model_name, version)
        return mlflow.prophet.load_model(model_uri=uri)
