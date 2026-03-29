import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import sentiment_service
from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading sentiment model: %s", settings.model_name)
    sentiment_service.load_model()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


# Root-level health endpoint for convenience
@app.get("/health")
async def root_health() -> dict:
    return {
        "status": "healthy" if sentiment_service.is_loaded else "degraded",
        "model_loaded": sentiment_service.is_loaded,
        "version": settings.app_version,
    }
