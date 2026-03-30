"""Tests for middleware components."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch

from app.api.dependencies import get_sentiment_service
from app.main import app
from app.services.sentiment import SentimentService


def _mock_service() -> SentimentService:
    service = MagicMock(spec=SentimentService)
    service.is_loaded = True
    service.device = "cpu"
    service.cache_stats = None
    service.analyze_async.return_value = {
        "text": "test",
        "sentiment": {"label": "positive", "score": 0.9},
        "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
    }
    return service


@pytest_asyncio.fixture
async def auth_client():
    """Client that tests API key auth."""
    mock = _mock_service()
    app.dependency_overrides[get_sentiment_service] = lambda: mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
@patch("app.middleware.settings")
async def test_api_key_required_rejects_no_key(mock_settings, auth_client: AsyncClient) -> None:
    mock_settings.api_key = "secret-key-123"
    response = await auth_client.post(
        "/api/v1/analyze",
        json={"text": "hello"},
    )
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


@pytest.mark.asyncio
@patch("app.middleware.settings")
async def test_api_key_required_accepts_valid_key(mock_settings, auth_client: AsyncClient) -> None:
    mock_settings.api_key = "secret-key-123"
    response = await auth_client.post(
        "/api/v1/analyze",
        json={"text": "hello"},
        headers={"X-API-Key": "secret-key-123"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
@patch("app.middleware.settings")
async def test_api_key_skips_health(mock_settings, auth_client: AsyncClient) -> None:
    mock_settings.api_key = "secret-key-123"
    response = await auth_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
@patch("app.middleware.settings")
async def test_api_key_skips_docs(mock_settings, auth_client: AsyncClient) -> None:
    mock_settings.api_key = "secret-key-123"
    response = await auth_client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_no_api_key_configured_allows_all(auth_client: AsyncClient) -> None:
    """When api_key is empty, all requests should pass."""
    response = await auth_client.post(
        "/api/v1/analyze",
        json={"text": "hello"},
    )
    assert response.status_code == 200
