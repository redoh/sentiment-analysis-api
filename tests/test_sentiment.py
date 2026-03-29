from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services.sentiment import SentimentService


class TestSentimentService:
    def test_initial_state(self) -> None:
        service = SentimentService()
        assert service.is_loaded is False

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

    @patch("app.services.sentiment.AutoTokenizer")
    @patch("app.services.sentiment.AutoModelForSequenceClassification")
    def test_load_model(self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock) -> None:
        service = SentimentService()
        service.load_model()

        assert service.is_loaded is True
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()

    @patch("app.services.sentiment.AutoTokenizer")
    @patch("app.services.sentiment.AutoModelForSequenceClassification")
    def test_analyze_returns_valid_result(
        self, mock_model_cls: MagicMock, mock_tokenizer_cls: MagicMock
    ) -> None:
        # Setup mock model output
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
