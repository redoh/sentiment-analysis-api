import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_sentiment_service
from app.config import settings
from app.middleware import metrics
from app.models.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.services.sentiment import SentimentService

logger = logging.getLogger("sentiment_api")

router = APIRouter()


@router.post(
    "/analyze",
    response_model=SentimentResponse,
    responses={503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def analyze_sentiment(
    request: SentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResponse:
    """Analyze sentiment of a single text."""
    try:
        result = await service.analyze_async(request.text)
        return SentimentResponse(**result)
    except TimeoutError as e:
        logger.warning("Inference timeout: %s", e)
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        logger.error("Service unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail="Internal inference error")


@router.post(
    "/analyze/batch",
    response_model=BatchSentimentResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def analyze_sentiment_batch(
    request: BatchSentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> BatchSentimentResponse:
    """Analyze sentiment of multiple texts."""
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}",
        )
    try:
        results = await service.analyze_batch_async(request.texts)
        return BatchSentimentResponse(results=[SentimentResponse(**r) for r in results])
    except TimeoutError as e:
        logger.warning("Batch inference timeout: %s", e)
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        logger.error("Service unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during batch analysis")
        raise HTTPException(status_code=500, detail="Internal inference error")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: SentimentService = Depends(get_sentiment_service),
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        version=settings.app_version,
        device=service.device,
        cache_stats=service.cache_stats,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Application metrics endpoint."""
    return MetricsResponse(**metrics.to_dict())
