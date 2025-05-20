from app import app

def main():
    import uvicorn
    print("Starting the Forecasting ML Microservice...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
if __name__ == "__main__":
    main()

# TODO: remove main and useless files from the docker file. 
# TODO: resolve the mlflow not showing the experiments issue in kube 