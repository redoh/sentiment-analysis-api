# CLAUDE.md — Sentiment Analysis API

## Project Overview

A production-ready REST API for real-time sentiment analysis with multi-language support. Accepts text input and returns sentiment classification (positive, negative, neutral) with confidence scores.

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (async, high-performance)
- **NLP/ML**: Hugging Face Transformers (`nlptown/bert-base-multilingual-uncased-sentiment` for multi-language sentiment)
- **HTTP Server**: Uvicorn (ASGI)
- **Validation**: Pydantic v2
- **Testing**: pytest + pytest-asyncio + httpx (async test client)
- **Linting/Formatting**: ruff
- **Dependency Management**: pip + requirements.txt
- **Containerization**: Docker + docker-compose
- **CI/CD**: GitHub Actions

## Project Structure

```
sentiment-analysis-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point with lifespan, CORS, middleware
│   ├── config.py             # Settings via pydantic-settings (env vars)
│   ├── middleware.py          # Request logging, API key auth, metrics collector
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Request/response Pydantic models with OpenAPI examples
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py         # API route definitions with error handling
│   │   └── dependencies.py   # Shared dependencies (model loader)
│   └── services/
│       ├── __init__.py
│       └── sentiment.py      # Sentiment analysis service with caching, GPU, timeout
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures and mock service
│   ├── test_api.py           # API endpoint tests (21 tests)
│   ├── test_sentiment.py     # Service, cache, metrics tests (19 tests)
│   └── test_middleware.py    # Auth middleware tests (5 tests)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml
├── CLAUDE.md
└── README.md
```

## Commands

- **Run dev server**: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- **Run tests**: `python -m pytest -v`
- **Run single test**: `python -m pytest tests/test_api.py::test_name -v`
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Lint fix**: `ruff check --fix .`
- **Docker build**: `docker build -t sentiment-api .`
- **Docker run**: `docker-compose up`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/analyze` | Analyze sentiment of a single text |
| POST | `/api/v1/analyze/batch` | Analyze sentiment of multiple texts |
| GET | `/api/v1/health` | Health check with model/cache/device info |
| GET | `/api/v1/metrics` | Application metrics (request count, latency) |
| GET | `/health` | Root-level health check |
| GET | `/metrics` | Root-level metrics |
| GET | `/docs` | Swagger UI (auto-generated) |
| GET | `/redoc` | ReDoc UI |

## Environment Variables

All prefixed with `SENTIMENT_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_DEBUG` | `false` | Enable debug mode |
| `SENTIMENT_MODEL_NAME` | `nlptown/bert-base-multilingual-uncased-sentiment` | HuggingFace model |
| `SENTIMENT_MODEL_REVISION` | `main` | Model revision/tag |
| `SENTIMENT_DEVICE` | `auto` | Device: auto, cpu, cuda |
| `SENTIMENT_USE_FP16` | `false` | Enable FP16 inference (CUDA only) |
| `SENTIMENT_API_KEY` | `` | API key (empty = no auth) |
| `SENTIMENT_ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins |
| `SENTIMENT_RATE_LIMIT` | `60/minute` | Rate limit per client |
| `SENTIMENT_INFERENCE_TIMEOUT_SECONDS` | `30.0` | Max inference time |
| `SENTIMENT_CACHE_ENABLED` | `true` | Enable LRU result cache |
| `SENTIMENT_CACHE_MAX_SIZE` | `1024` | Max cached results |
| `SENTIMENT_LOG_LEVEL` | `INFO` | Logging level |

## Code Conventions

- Use type hints everywhere
- Async endpoints by default
- Pydantic models for all request/response schemas
- Service layer pattern: routes delegate to services
- Load ML model once at startup via FastAPI lifespan
- Environment config via pydantic-settings (`.env` support)
- Tests use httpx AsyncClient with dependency overrides
- Keep functions small and focused
- Structured logging with request IDs
