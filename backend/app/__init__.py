from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="Deepfake Detector API")
    return app
