from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (auto-detected if not provided)",
    )


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=32, description="List of texts to analyze"
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (auto-detected if not provided)",
    )


class SentimentScore(BaseModel):
    label: str = Field(..., description="Sentiment label: positive, negative, or neutral")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class SentimentResponse(BaseModel):
    text: str
    sentiment: SentimentScore
    scores: dict[str, float] = Field(..., description="Scores for all sentiment categories")


class BatchSentimentResponse(BaseModel):
    results: list[SentimentResponse]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str
