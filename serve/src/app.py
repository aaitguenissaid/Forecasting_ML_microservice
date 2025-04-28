import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config.config import LOG_FORMAT
from registry.mlflow.handler import MlflowRegistryClient
from registry.mlflow.loader import LocalModelLoader
from helpers.requests import (
    ForecastRequestDays,
    ForecastRequestInterval,
    create_forecast_index_days,
    create_forecast_index_interval,
)

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    services["registry"] = MlflowRegistryClient()
    logging.info("MLflowRegistryClient initialized.")
    services["loader"] = LocalModelLoader()
    logging.info("LocalModelLoader initialized.")
    yield
    services.clear()
    logging.info("Cleared in-memory caches.")
    logging.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

@app.get("/", status_code=200)
def read_root() -> dict:
    return {"message": f"Welcome to the Forecasting Microservice API. at {os.getcwd()}"}

@app.get("/health/", status_code=200)
async def health_check() -> dict:
    global services
    registry: MlflowRegistryClient = services["registry"]
    logging.info("Got registry in healthcheck.")
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": registry.check_mlflow_health()
    }

async def _generate_forecast(items: List[BaseModel], create_index: Callable[[Any], pd.DataFrame]) -> List[dict]:
    loader: LocalModelLoader = services["loader"]
    forecasts = []
    for item in items:
        store_id = item.store_id
        if not loader.model_exists(store_id):
            raise HTTPException(404, f"No model foind for store {store_id}")
        model = loader.get_production_model(store_id)
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
    return await _generate_forecast(forecast_request, lambda item: create_forecast_index_interval(item.begin_date, item.end_date))

@app.post("/forecast_days/", status_code=200)
async def return_forecast_days(forecast_request: ForecastRequestDays) -> List[dict]:
    return await _generate_forecast([forecast_request], lambda item: create_forecast_index_days(item.days))
