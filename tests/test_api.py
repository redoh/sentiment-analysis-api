import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_analyze_sentiment(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "I love this product!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"]["label"] == "positive"
    assert 0 <= data["sentiment"]["score"] <= 1
    assert "scores" in data
    assert set(data["scores"].keys()) == {"negative", "neutral", "positive"}


@pytest.mark.asyncio
async def test_analyze_batch(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze/batch",
        json={"texts": ["I love this!", "I hate this!"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"]["label"] == "positive"
    assert data["results"][1]["sentiment"]["label"] == "negative"


@pytest.mark.asyncio
async def test_analyze_empty_text(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_missing_text(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_api_v1_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"
