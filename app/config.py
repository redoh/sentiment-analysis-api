from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model settings
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    model_revision: str = "main"
    max_batch_size: int = 32
    max_text_length: int = 512
    use_fp16: bool = False
    device: str = "auto"  # "auto", "cpu", "cuda"

    # Security
    api_key: str = ""  # Empty = no auth required
    allowed_origins: list[str] = ["*"]

    # Rate limiting
    rate_limit: str = "60/minute"

    # Timeouts
    inference_timeout_seconds: float = 30.0

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 1024

    # Logging
    log_level: str = "INFO"

    model_config = {"env_prefix": "SENTIMENT_", "env_file": ".env"}


settings = Settings()
