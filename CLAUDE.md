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
│   ├── main.py              # FastAPI app entry point
│   ├── config.py             # Settings via pydantic-settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Request/response Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py         # API route definitions
│   │   └── dependencies.py   # Shared dependencies (model loader)
│   └── services/
│       ├── __init__.py
│       └── sentiment.py      # Sentiment analysis service
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_sentiment.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── CLAUDE.md
└── README.md
```

## Commands

- **Run dev server**: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- **Run tests**: `pytest -v`
- **Run single test**: `pytest tests/test_api.py::test_name -v`
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
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (auto-generated) |

## Code Conventions

- Use type hints everywhere
- Async endpoints by default
- Pydantic models for all request/response schemas
- Service layer pattern: routes delegate to services
- Load ML model once at startup via FastAPI lifespan
- Environment config via pydantic-settings (`.env` support)
- Tests use httpx AsyncClient with FastAPI TestClient
- Keep functions small and focused
