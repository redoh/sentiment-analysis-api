import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings
from app.middleware import metrics

logger = logging.getLogger("sentiment_api")

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


class LRUCache:
    """Simple thread-safe LRU cache for sentiment results."""

    def __init__(self, max_size: int = 1024) -> None:
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> dict | None:
        key = self._key(text)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, result: dict) -> None:
        key = self._key(text)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class SentimentService:
    def __init__(self) -> None:
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        self._loaded = False
        self._device: torch.device | None = None
        self._cache = LRUCache(max_size=settings.cache_max_size) if settings.cache_enabled else None

    def _resolve_device(self) -> torch.device:
        if settings.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(settings.device)

    def load_model(self) -> None:
        self._device = self._resolve_device()
        logger.info("Using device: %s", self._device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name, revision=settings.model_revision
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_name, revision=settings.model_revision
        )
        self.model.eval()
        self.model.to(self._device)

        if settings.use_fp16 and self._device.type == "cuda":
            self.model = self.model.half()
            logger.info("Model converted to FP16")

        self._loaded = True
        logger.info("Model loaded: %s (revision=%s)", settings.model_name, settings.model_revision)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return str(self._device) if self._device else "none"

    @property
    def cache_stats(self) -> dict | None:
        return self._cache.stats if self._cache else None

    def _aggregate_scores(self, probabilities: list[float]) -> dict[str, float]:
        """Aggregate 5-star probabilities into 3 sentiment categories."""
        negative = probabilities[0] + probabilities[1]
        neutral = probabilities[2]
        positive = probabilities[3] + probabilities[4]
        return {
            "negative": round(negative, 4),
            "neutral": round(neutral, 4),
            "positive": round(positive, 4),
        }

    def _build_result(self, text: str, probabilities: list[float]) -> dict:
        scores = self._aggregate_scores(probabilities)
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        return {
            "text": text,
            "sentiment": {"label": best_label, "score": round(best_score, 4)},
            "scores": scores,
        }

    def _run_inference(self, texts: list[str]) -> list[list[float]]:
        """Run model inference and return probabilities for each text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.monotonic()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_text_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        all_probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

        duration_ms = (time.monotonic() - start) * 1000
        metrics.record_inference(duration_ms)
        logger.debug("Inference completed in %.1fms for %d texts", duration_ms, len(texts))

        return all_probs

    def analyze(self, text: str) -> dict:
        if self._cache:
            cached = self._cache.get(text)
            if cached:
                return cached

        probs = self._run_inference([text])
        result = self._build_result(text, probs[0])

        if self._cache:
            self._cache.put(text, result)

        return result

    async def analyze_async(self, text: str) -> dict:
        """Async wrapper with timeout for single text analysis."""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.analyze, text),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Inference timed out after {settings.inference_timeout_seconds}s")

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        # Check cache for each text, only infer uncached ones
        results: dict[int, dict] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text)
                if cached:
                    results[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            all_probs = self._run_inference(uncached_texts)
            for idx, (text, probs) in zip(uncached_indices, zip(uncached_texts, all_probs)):
                result = self._build_result(text, probs)
                results[idx] = result
                if self._cache:
                    self._cache.put(text, result)

        return [results[i] for i in range(len(texts))]

    async def analyze_batch_async(self, texts: list[str]) -> list[dict]:
        """Async wrapper with timeout for batch analysis."""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.analyze_batch, texts),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Batch inference timed out after {settings.inference_timeout_seconds}s"
            )
