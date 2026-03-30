import logging
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger("sentiment_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request/response details with timing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.monotonic()
        request_id = f"{time.time_ns()}"

        logger.info(
            "Request started | method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.exception(
                "Request failed | method=%s path=%s request_id=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                request_id,
                duration_ms,
            )
            raise

        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Request completed | method=%s path=%s status=%d duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key from X-API-Key header. Skip if no key configured."""

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.api_key:
            return await call_next(request)

        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


# Simple in-memory metrics collector
class MetricsCollector:
    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0
        self.inference_count: int = 0
        self.inference_latency_ms: float = 0.0

    def record_request(self, duration_ms: float, error: bool = False) -> None:
        self.request_count += 1
        self.total_latency_ms += duration_ms
        if error:
            self.error_count += 1

    def record_inference(self, duration_ms: float) -> None:
        self.inference_count += 1
        self.inference_latency_ms += duration_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def avg_inference_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.inference_latency_ms / self.inference_count

    def to_dict(self) -> dict:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "inference_count": self.inference_count,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
        }


metrics = MetricsCollector()
