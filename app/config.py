from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: str = Field("models", alias="MODELS_DIR")

    prediction_length: int = Field(1, alias="PREDICTION_LENGTH")
    training_time_limit: int = Field(120, alias="TRAINING_TIME_LIMIT")
    min_candles_train: int = Field(100, alias="MIN_CANDLES_TRAIN")
    min_candles_predict: int = Field(30, alias="MIN_CANDLES_PREDICT")
    target: str = Field("close", alias="TARGET")

    ma_windows: list[int] = Field([5, 10, 20], alias="MA_WINDOWS")
    rsi_period: int = Field(14, alias="RSI_PERIOD")

    retrain_holdout_size: int = Field(20, ge=1, alias="RETRAIN_HOLDOUT_SIZE")
    retrain_mape_tolerance: float = Field(0.0, ge=0.0, alias="RETRAIN_MAPE_TOLERANCE")

    retrain_cron: str = Field("0 3 * * *", alias="RETRAIN_CRON")
    retrain_timezone: str = Field("Europe/Moscow", alias="RETRAIN_TIMEZONE")

    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("market_data", alias="POSTGRES_DB")
    postgres_user: str = Field("postgres", alias="POSTGRES_USER")
    postgres_password: str = Field("postgres", alias="POSTGRES_PASSWORD")

    mlflow_uri: str = Field("http://market_mlflow:5000", alias="MLFLOW_URI")

    loki_url: str = Field("localhost:3100", alias="LOKI_URL")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
