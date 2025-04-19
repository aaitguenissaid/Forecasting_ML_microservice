from pathlib import Path, PurePath

# Automatically resolve the project root (assuming 2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
# Key folders relative to root
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.csv" 
MLRUNS_DIR = PROJECT_ROOT / "train" / "mlruns"
RESULTS_DIR = PROJECT_ROOT / "train" / "results"
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_BASE_NAME = f"prophet-retail-forecaster-store"

def get_artifact_path(store_id:int) -> str:
    return f"model-{store_id:04}"

def get_model_name(store_id : str) -> str:
    return f"{MODEL_BASE_NAME}-{int(store_id):04}"

def get_model_path( model_dir : str = MODEL_DIR, model_name : str = None, model_version : str = None) -> str:
    return str(Path(model_dir, model_name, model_version))

def get_model_uri( run_id : str, artifact_path : str) -> str:
    return str(Path("runs:/", run_id, artifact_path))

