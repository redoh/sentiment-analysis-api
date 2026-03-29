import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings

# Map 1-5 star ratings to sentiment labels
STAR_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


class SentimentService:
    def __init__(self) -> None:
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        self._loaded = False

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_name)
        self.model.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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

    def analyze(self, text: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_text_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        scores = self._aggregate_scores(probabilities)

        # Determine dominant sentiment
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        return {
            "text": text,
            "sentiment": {"label": best_label, "score": round(best_score, 4)},
            "scores": scores,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_text_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        all_probs = torch.softmax(outputs.logits, dim=-1).tolist()

        results = []
        for text, probabilities in zip(texts, all_probs):
            scores = self._aggregate_scores(probabilities)
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            results.append(
                {
                    "text": text,
                    "sentiment": {"label": best_label, "score": round(best_score, 4)},
                    "scores": scores,
                }
            )

        return results
