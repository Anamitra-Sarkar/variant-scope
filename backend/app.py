import os
import threading
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
load_dotenv()

from firebase_auth import init_firebase
from models.loader import ModelLoader
from config import CORS_ORIGINS, CORS_ORIGIN_REGEX

model_loader = ModelLoader()
sae_model = None


def _warmup():
    global sae_model
    try:
        print("Warmup: loading ESM-2...")
        model_loader.load_esm()
        print("Warmup: loading DNABERT-2...")
        model_loader.load_dnabert()
        print("Warmup: loading SAE...")
        sae_model = model_loader.load_sae()
        print("Models loaded and warmed up successfully")
    except Exception as e:
        print(f"Warmup warning (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    threading.Thread(target=_warmup, daemon=True).start()
    print("VariantScope API started")
    yield
    print("Shutting down")


app = FastAPI(
    title="VariantScope API",
    description="Dual-Engine Transformer Framework for Genomic and Proteomic Variant Effect Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.health import router as health_router
from routes.prediction import router as prediction_router
from routes.agent import router as agent_router
from routes.history import router as history_router
from routes.models import router as models_router

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(agent_router)
app.include_router(history_router)
app.include_router(models_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
