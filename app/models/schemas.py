from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (auto-detected if not provided)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "I absolutely love this product! Best purchase ever."},
                {"text": "Bu restoran berbattı, bir daha gelmem.", "language": "tr"},
            ]
        }
    }


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=32, description="List of texts to analyze"
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (auto-detected if not provided)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "Great service, very happy!",
                        "Terrible experience, never again.",
                        "It was okay, nothing special.",
                    ]
                }
            ]
        }
    }


class SentimentScore(BaseModel):
    label: str = Field(..., description="Sentiment label: positive, negative, or neutral")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class SentimentResponse(BaseModel):
    text: str
    sentiment: SentimentScore
    scores: dict[str, float] = Field(..., description="Scores for all sentiment categories")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I love this product!",
                    "sentiment": {"label": "positive", "score": 0.92},
                    "scores": {"negative": 0.03, "neutral": 0.05, "positive": 0.92},
                }
            ]
        }
    }


class BatchSentimentResponse(BaseModel):
    results: list[SentimentResponse]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str
    device: str = "cpu"
    cache_stats: dict | None = None


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    avg_latency_ms: float
    inference_count: int
    avg_inference_ms: float


class ErrorResponse(BaseModel):
    detail: str
