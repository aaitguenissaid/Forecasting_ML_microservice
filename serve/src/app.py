import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from registry.mlflow.handler import MLFlowHandler
from mlflow.pyfunc import PyFuncModel
from helpers.requests import ForecastRequest, create_forecast_index
from helpers.paths import get_model_name
from typing import List
import os


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

#TODO replace global variables with redis or memcache to avoid race conditions.
handlers = {}
models = {}

async def get_service_handlers() -> dict:
    mlflow_handler = MLFlowHandler()
    global handlers
    handlers['mlflow'] = mlflow_handler
    logging.info(f"Retrieving mlflow handler {mlflow_handler}")
    return handlers

@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_service_handlers()
    logging.info("Service handlers initialized on startup.")
    yield
    logging.info("Shutting down application...")
    handlers.clear()
    models.clear()
    logging.info("Cleared in-memory caches.")
    # TODO enhance cleanup if needed

app = FastAPI(lifespan=lifespan)

@app.get("/", status_code=200)
def read_root() -> dict:
    return {"message": f"Welcome to the Forecasting Microservice API. at {os.getcwd()}"}

@app.get("/health/", status_code=200)
async def health_check() -> dict:
    global handlers
    logging.info("Got handlers in healthcheck.")
    mlflow_health = handlers.get('mlflow').check_mlflow_health() if 'mlflow' in handlers else 'Unknown'
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": mlflow_health
    }

async def get_model(store_id: str) -> PyFuncModel:
    global handlers
    global models
    store_id_int = int(store_id)
    model_name = get_model_name(store_id)
    if model_name not in models:
        models[model_name] = handlers['mlflow'].get_production_model(store_id=store_id)
        logging.info(f"Loaded model for store {store_id}")
    return models[model_name]

@app.post("/forecast/", status_code=200)
async def return_forecast(forecast_request: List[ForecastRequest]) -> List[dict]:
    forecasts = []
    for item in forecast_request:
        model = await get_model(item.store_id)
        forecast_input = create_forecast_index(begin_date=item.begin_date, end_date=item.end_date)
        prediction = model.predict(forecast_input)[['ds', 'yhat']]
        prediction = prediction.rename(columns={'ds': 'timestamp', 'yhat': 'value'})
        prediction['value'] = prediction['value'].astype(int)
        forecasts.append({
            'request': item.model_dump(),
            'forecast': prediction.to_dict('records')
        })
    return forecasts