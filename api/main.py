from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from training.train_pipeline import train_and_evaluate_all
from config.settings import BEST_MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure baseline trained model exists on API startup."""
    if not BEST_MODEL_PATH.exists():
        print("Initial model artifacts not detected. Running startup model training...")
        train_and_evaluate_all()
    yield

app = FastAPI(
    title="AI Student Performance Prediction REST API",
    description="Production Machine Learning & Risk Diagnostics API for Student Academic Performance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


