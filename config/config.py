import os
from pathlib import Path

# Constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TRACKING_URI = os.getenv("TRACKING_URI", "http://0.0.0.0:5000")
MODEL_BASE_NAME = f"prophet-retail-forecaster-store"
MODEL_ID_SIZE = 4

# TODO use only os.path, and set all dirs to strs.
#   
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.csv" 
MLRUNS_DIR = PROJECT_ROOT / "train" / "mlruns"
RESULTS_DIR = PROJECT_ROOT / "train" / "results"

def get_model_id(store_id: str) -> str:
    return store_id.zfill(MODEL_ID_SIZE)

def get_artifact_path(store_id: str) -> str:
    return f"model-{get_model_id(store_id)}"

def get_model_name(store_id : str) -> str:
    return f"{MODEL_BASE_NAME}-{get_model_id(store_id)}"

def get_model_path( model_dir : str = MODEL_DIR, model_name : str = None, model_version : str = None) -> str:
    return os.path.join(model_dir, model_name, model_version)

def get_model_uri( run_id : str, artifact_path : str) -> str:
    return str(Path("runs:/", run_id, artifact_path))

def get_model_versions(model_registry_path: str, model_name: str) -> list[str]:
    model_path = os.path.join(model_registry_path, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at: {model_path}")

    # List subfolders which represent version numbers
    model_versions = [
        name for name in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, name)) and name.isdigit()
    ]

    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'.")

    return sorted(model_versions)

def get_latest_model_version(model_registry_path: str, model_name: str) -> str:
    model_versions = get_model_versions(model_registry_path, model_name)
    latest_version = max(int(v) for v in model_versions)
    return str(latest_version)
