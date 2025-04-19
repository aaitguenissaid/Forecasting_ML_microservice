from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
    
#TODO: set path to production models!
MODEL_DIR = PROJECT_ROOT / "models"
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_BASE_NAME = f"prophet-retail-forecaster-store"

def get_model_name(store_id : str) -> str:
    return f"{MODEL_BASE_NAME}-{int(store_id):04}"

def get_model_path( model_dir : str = MODEL_DIR, model_name : str = None, model_version : str = None) -> str:
    return str(Path(model_dir, model_name, model_version))
