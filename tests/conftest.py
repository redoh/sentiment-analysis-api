from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_sentiment_service
from app.main import app
from app.services.sentiment import SentimentService


def _make_mock_service() -> SentimentService:
    """Create a mock sentiment service that returns predictable results."""
    service = MagicMock(spec=SentimentService)
    service.is_loaded = True

    service.analyze.return_value = {
        "text": "I love this product!",
        "sentiment": {"label": "positive", "score": 0.95},
        "scores": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
    }

    service.analyze_batch.return_value = [
        {
            "text": "I love this!",
            "sentiment": {"label": "positive", "score": 0.95},
            "scores": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
        },
        {
            "text": "I hate this!",
            "sentiment": {"label": "negative", "score": 0.90},
            "scores": {"negative": 0.90, "neutral": 0.05, "positive": 0.05},
        },
    ]

    return service


@pytest.fixture
def mock_service() -> SentimentService:
    return _make_mock_service()


@pytest_asyncio.fixture
async def client(mock_service: SentimentService):
    app.dependency_overrides[get_sentiment_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
