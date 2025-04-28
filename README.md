# Forecasting ML Microservice

![Build Status](https://github.com/aaitguenissaid/Forecasting_ML_microservice/actions/workflows/ci.yml/badge.svg)

## Overview
This project is a machine learning microservice for time-series forecasting. It uses Prophet models to forecast retail data and provides a REST API for serving predictions. The project supports training, serving, and deployment using Docker and Kubernetes.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Setup
```bash
git clone git@github.com:aaitguenissaid/Forecasting_ML_microservice.git
cd Forecasting_ML_microservice/train/
uv sync
uv venv
source .venv/bin/activate
mlflow ui
```

## Usage
### Training
```bash
# In a separate terminal
cd train
source .venv/bin/activate
uv run train_forecasting_basic.py
```

### Serving the Model
```bash
export PYTHONPATH=$(pwd)/serve/src:$(pwd)
cd serve
uv sync
uv venv
source .venv/bin/activate
uvicorn src.app:app --reload
```

### Running with Docker
```bash
sudo docker build -t custom-forecast-service:latest -f serve/Dockerfile .
sudo docker run -e TRACKING_URI=http://172.17.0.1:5000 -p 8000:8000 -v "$(pwd)/models:/app/models" custom-forecast-service:latest
```

### Running with Minikube
```bash
docker build -t custom-forecast-service:latest -f serve/Dockerfile .

minikube start --driver=docker
minikube image load custom-forecast-service:latest
minikube mount ./models:/app/models &
minikube mount train/mlartifacts/:/mlartifacts &
minikube mount train/mlruns/:/mlruns &
kubectl apply -f serve/direct_kube_deploy.yaml
minikube tunnel

minikube service mlflow-service --url
kubectl get svc
kubectl get pods
kubectl get deployments

watch -n 0.5 kubectl --tail=20 logs <POD_NAME>
```

### Shutting Down Minikube
```bash
kubectl delete service fast-api-service mlflow-service
kubectl delete deployment fast-api-deployment mlflow-deployment

pkill -f "minikube tunnel"
minikube stop
minikube delete
```

## API Documentation

The Forecasting ML Microservice provides an interactive API documentation interface powered by FastAPI's OpenAPI support. You can access it at the following endpoints:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to explore the available endpoints, their request/response schemas, and test the API directly.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
