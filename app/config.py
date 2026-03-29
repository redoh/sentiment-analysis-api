from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    debug: bool = False
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    max_batch_size: int = 32
    max_text_length: int = 512

    model_config = {"env_prefix": "SENTIMENT_", "env_file": ".env"}


settings = Settings()
