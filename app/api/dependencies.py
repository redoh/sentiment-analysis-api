from app.services.sentiment import SentimentService

sentiment_service = SentimentService()


def get_sentiment_service() -> SentimentService:
    return sentiment_service
