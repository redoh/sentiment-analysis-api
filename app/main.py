import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import sentiment_service
from app.api.routes import router
from app.config import settings
from app.middleware import APIKeyMiddleware, RequestLoggingMiddleware, metrics

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sentiment_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading sentiment model: %s", settings.model_name)
    try:
        sentiment_service.load_model()
        logger.info("Model loaded successfully on device=%s", sentiment_service.device)
    except Exception:
        logger.exception("Failed to load model")
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware (order matters — outermost first)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def root_health() -> dict:
    return {
        "status": "healthy" if sentiment_service.is_loaded else "degraded",
        "model_loaded": sentiment_service.is_loaded,
        "version": settings.app_version,
        "device": sentiment_service.device,
    }


@app.get("/metrics")
async def root_metrics() -> dict:
    return metrics.to_dict()
