from app import app

def main():
    import uvicorn
    print("Starting the Forecasting ML Microservice...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
if __name__ == "__main__":
    main()
