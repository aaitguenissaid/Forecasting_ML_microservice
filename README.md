# Forecasting ML Microservice

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
cd Forecasting_ML_microservice/
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Usage
### Download Datasets
```bash
cd download_datasets
uv sync
uv venv
source .venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000 &
python main.py
```

### Training
```bash
# In a separate terminal
cd train
source .venv/bin/activate
uv run train_forecasting_basic.py
```

### Serving the Model
```bash
cd serve
uv sync
uv venv
source .venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000 &
python main.py  #or: cd src && uvicorn app:app --reload
```

### Serving the Model with Docker
```bash
cd train 
mlflow ui --host 0.0.0.0 --port 5000 &
cd ..
sudo docker build -t custom-forecast-service:latest -f serve/Dockerfile .
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway -e TRACKING_URI=http://host.docker.internal:5000 -v "$(pwd)/models:/app/models" custom-forecast-service:latest
```

### Serving the Model with Podman
```bash
cd train 
mlflow ui --host 0.0.0.0 --port 5000 &
cd ..
podman build -t custom-forecast-service:latest -f serve/Dockerfile .
podman run -p 8000:8000 -e TRACKING_URI=http://host.containers.internal:5000 -v "$(pwd)/models:/app/models" custom-forecast-service:latest
```

### Serving the Model with Docker Compose
```bash
docker compose -f serve/docker-compose.yml up --build -d
docker compose -f serve/docker-compose.yml down
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
minikube service fast-api-service --url
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
