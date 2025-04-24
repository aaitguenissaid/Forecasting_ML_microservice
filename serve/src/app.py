import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from config.config import LOG_FORMAT
from registry.mlflow.handler import MLFlowHandler
from mlflow.pyfunc import PyFuncModel
from helpers.requests import ForecastRequestDays, ForecastRequestInterval, create_forecast_index_days, create_forecast_index_interval
from helpers.paths import get_model_name
from typing import Any, Callable, List
import os

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

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
    model_name = get_model_name(store_id)
    if model_name not in models:
        models[model_name] = handlers['mlflow'].get_production_model(store_id=store_id)
        logging.info(f"Loaded model for store {store_id}")
    return models[model_name]

async def _generate_forecast(items: List[BaseModel], create_index: Callable[[Any], pd.DataFrame]) -> List[dict]:
    forecasts = []
    print()
    for item in items:
        model = await get_model(item.store_id)
        forecast_input = create_index(item)
        prediction = model.predict(forecast_input)[['ds', 'yhat']]
        prediction = prediction.rename(columns={'ds': 'timestamp', 'yhat': 'value'})
        prediction['value'] = prediction['value'].astype(int)
        forecasts.append({
            'request': item.model_dump(),
            'forecast': prediction.to_dict('records')
        })
    return forecasts

@app.post("/forecast_interval/", status_code=200)
async def return_forecast_interval(forecast_request: List[ForecastRequestInterval]) -> List[dict]:
    return await _generate_forecast(
        forecast_request,
        lambda item: create_forecast_index_interval(begin_date=item.begin_date, end_date=item.end_date)
    )

@app.post("/forecast_days/", status_code=200)
async def return_forecast_days(forecast_request: ForecastRequestDays) -> List[dict]:
    return await _generate_forecast(
        [forecast_request],
        lambda item: create_forecast_index_days(days=item.days)
    )
