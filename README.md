 # Setup
git clone git@github.com:aaitguenissaid/Forecasting_ML_microservice.git
cd Forecasting_ML_microservice/train/
uv sync
uv venv
source .venv/bin/activate
mlflow ui

## in  a separate terminal
cd train 
source .venv/bin/activate
uv run train_forecasting_basic.py 

## to serve the model:
export PYTHONPATH=$(pwd)/serve/src:$(pwd)
cd serve
uv sync
uv venv
source .venv/bin/activate
uvicorn src.app:app (this tag --reload can be used for developement.)

## to run Docker:
sudo docker build -t custom-forecast-service:latest -f serve/Dockerfile .
sudo docker run -e TRACKING_URI=http://172.17.0.1:5000 -p 8000:8000 -v "$(pwd)/models:/app/models" custom-forecast-service:latest 
or 
http://host.docker.internal:5000

## to run minikube
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

### To shutdown minikube
kubectl delete service fast-api-service mlflow-service
kubectl delete deployment fast-api-deployment mlflow-deployment

pkill -f "minikube tunnel"
minikube stop
minikube delete
