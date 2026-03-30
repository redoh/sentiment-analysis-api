from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services.sentiment import LRUCache, SentimentService


class TestSentimentService:
    def test_initial_state(self) -> None:
        service = SentimentService()
        assert service.is_loaded is False
        assert service.device == "none"

    def test_analyze_raises_when_not_loaded(self) -> None:
        service = SentimentService()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.analyze("test text")

    def test_analyze_batch_raises_when_not_loaded(self) -> None:
        service = SentimentService()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.analyze_batch(["test text"])

    def test_aggregate_scores(self) -> None:
        service = SentimentService()
        # 5-star probabilities: [1star, 2star, 3star, 4star, 5star]
        scores = service._aggregate_scores([0.1, 0.1, 0.2, 0.3, 0.3])
        assert scores["negative"] == pytest.approx(0.2, abs=0.001)
        assert scores["neutral"] == pytest.approx(0.2, abs=0.001)
        assert scores["positive"] == pytest.approx(0.6, abs=0.001)

    def test_aggregate_scores_all_negative(self) -> None:
        service = SentimentService()
        scores = service._aggregate_scores([0.5, 0.4, 0.05, 0.03, 0.02])
        assert scores["negative"] > scores["positive"]
        assert scores["negative"] > scores["neutral"]

    def test_aggregate_scores_all_neutral(self) -> None:
        service = SentimentService()
        scores = service._aggregate_scores([0.05, 0.05, 0.8, 0.05, 0.05])
        assert scores["neutral"] > scores["positive"]
        assert scores["neutral"] > scores["negative"]

    def test_build_result(self) -> None:
        service = SentimentService()
        result = service._build_result("test", [0.1, 0.1, 0.2, 0.3, 0.3])
        assert result["text"] == "test"
        assert result["sentiment"]["label"] == "positive"
        assert 0 <= result["sentiment"]["score"] <= 1
        assert set(result["scores"].keys()) == {"negative", "neutral", "positive"}

    @patch("app.services.sentiment.AutoTokenizer")
    @patch("app.services.sentiment.AutoModelForSequenceClassification")
    def test_load_model(self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock) -> None:
        service = SentimentService()
        service.load_model()

        assert service.is_loaded is True
        assert service.device in ("cpu", "cuda")
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()

    @patch("app.services.sentiment.AutoTokenizer")
    @patch("app.services.sentiment.AutoModelForSequenceClassification")
    def test_analyze_returns_valid_result(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        mock_logits = torch.tensor([[0.1, 0.1, 0.2, 0.3, 0.3]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        service = SentimentService()
        service.load_model()
        result = service.analyze("Great product!")

        assert result["text"] == "Great product!"
        assert result["sentiment"]["label"] in ("positive", "negative", "neutral")
        assert 0 <= result["sentiment"]["score"] <= 1
        assert set(result["scores"].keys()) == {"negative", "neutral", "positive"}

    @patch("app.services.sentiment.AutoTokenizer")
    @patch("app.services.sentiment.AutoModelForSequenceClassification")
    def test_analyze_batch_returns_valid_results(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        mock_logits = torch.tensor(
            [
                [0.1, 0.1, 0.2, 0.3, 0.3],
                [0.4, 0.4, 0.1, 0.05, 0.05],
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = mock_logits

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        service = SentimentService()
        service.load_model()
        results = service.analyze_batch(["Great!", "Terrible!"])

        assert len(results) == 2
        assert results[0]["sentiment"]["label"] == "positive"
        assert results[1]["sentiment"]["label"] == "negative"


class TestLRUCache:
    def test_basic_put_get(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("hello", {"result": "positive"})
        assert cache.get("hello") == {"result": "positive"}

    def test_cache_miss(self) -> None:
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_eviction(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("a", {"r": 1})
        cache.put("b", {"r": 2})
        cache.put("c", {"r": 3})  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == {"r": 2}
        assert cache.get("c") == {"r": 3}

    def test_lru_order(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("a", {"r": 1})
        cache.put("b", {"r": 2})
        cache.get("a")  # access "a", making "b" the least recently used
        cache.put("c", {"r": 3})  # should evict "b"
        assert cache.get("a") == {"r": 1}
        assert cache.get("b") is None
        assert cache.get("c") == {"r": 3}

    def test_stats(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("a", {"r": 1})
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_overwrite(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("a", {"r": 1})
        cache.put("a", {"r": 2})
        assert cache.get("a") == {"r": 2}
        assert cache.stats["size"] == 1


class TestMetricsCollector:
    def test_metrics_initial(self) -> None:
        from app.middleware import MetricsCollector

        m = MetricsCollector()
        assert m.request_count == 0
        assert m.avg_latency_ms == 0.0
        assert m.avg_inference_ms == 0.0

    def test_record_request(self) -> None:
        from app.middleware import MetricsCollector

        m = MetricsCollector()
        m.record_request(100.0)
        m.record_request(200.0, error=True)
        assert m.request_count == 2
        assert m.error_count == 1
        assert m.avg_latency_ms == pytest.approx(150.0)

    def test_record_inference(self) -> None:
        from app.middleware import MetricsCollector

        m = MetricsCollector()
        m.record_inference(50.0)
        m.record_inference(150.0)
        assert m.inference_count == 2
        assert m.avg_inference_ms == pytest.approx(100.0)

    def test_to_dict(self) -> None:
        from app.middleware import MetricsCollector

        m = MetricsCollector()
        m.record_request(100.0)
        d = m.to_dict()
        assert "request_count" in d
        assert "error_count" in d
        assert "avg_latency_ms" in d
        assert "inference_count" in d
        assert "avg_inference_ms" in d
