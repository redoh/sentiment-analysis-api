import pytest
from httpx import AsyncClient


# --- Single Analysis ---


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
async def test_analyze_with_language(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "Bu harika!", "language": "tr"},
    )
    assert response.status_code == 200


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
async def test_analyze_long_text(client: AsyncClient) -> None:
    """Text at max boundary (5000 chars) should succeed."""
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "a" * 5000},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_analyze_too_long_text(client: AsyncClient) -> None:
    """Text exceeding 5000 chars should fail validation."""
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "a" * 5001},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_unicode_text(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "这个产品非常好！我很喜欢 🎉"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_analyze_special_characters(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "<script>alert('xss')</script> & \"quotes\" 'single'"},
    )
    assert response.status_code == 200


# --- Batch Analysis ---


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
async def test_analyze_batch_single_item(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze/batch",
        json={"texts": ["Single text"]},
    )
    assert response.status_code == 200
    assert "results" in response.json()


@pytest.mark.asyncio
async def test_analyze_batch_empty_list(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze/batch",
        json={"texts": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_batch_missing_texts(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze/batch",
        json={},
    )
    assert response.status_code == 422


# --- Health & Metrics ---


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data


@pytest.mark.asyncio
async def test_api_v1_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"
    assert data["device"] == "cpu"
    assert data["cache_stats"] is not None


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient) -> None:
    response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data
    assert "error_count" in data
    assert "avg_latency_ms" in data
    assert "inference_count" in data


@pytest.mark.asyncio
async def test_root_metrics(client: AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data


# --- Response Headers ---


@pytest.mark.asyncio
async def test_response_has_request_id(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_response_has_timing(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "x-response-time-ms" in response.headers


# --- Malformed Requests ---


@pytest.mark.asyncio
async def test_invalid_json(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_wrong_field_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze",
        json={"text": 12345},
    )
    # FastAPI/Pydantic will try to coerce int to str
    assert response.status_code in (200, 422)


@pytest.mark.asyncio
async def test_batch_wrong_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyze/batch",
        json={"texts": "not a list"},
    )
    assert response.status_code == 422
