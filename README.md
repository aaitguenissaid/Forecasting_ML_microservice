 # Setup
git clone git@github.com:aaitguenissaid/Forecasting_ML_microservice.git
cd Forecasting_ML_microservice/train/
uv sync
uv venv
source .venv/bin/activate
mlflow ui

## in  a separate terminal  
source .venv/bin/activate
uv run train_forecasting_basic.py 
