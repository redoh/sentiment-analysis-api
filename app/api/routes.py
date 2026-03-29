from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_sentiment_service
from app.config import settings
from app.models.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.services.sentiment import SentimentService

router = APIRouter()


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResponse:
    """Analyze sentiment of a single text."""
    try:
        result = service.analyze(request.text)
        return SentimentResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/analyze/batch", response_model=BatchSentimentResponse)
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
        results = service.analyze_batch(request.texts)
        return BatchSentimentResponse(results=[SentimentResponse(**r) for r in results])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: SentimentService = Depends(get_sentiment_service),
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        version=settings.app_version,
    )
