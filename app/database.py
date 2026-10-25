from datetime import UTC, datetime

import structlog
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings

logger = structlog.get_logger(__name__)

engine = create_async_engine(settings.postgres_dsn, echo=False, pool_size=10)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ModelTrainingRun(Base):
    __tablename__ = "model_training_runs"
    __table_args__ = (
        Index("idx_mtr_instrument_interval_started", "instrument_uid", "interval", "started_at"),
        Index("idx_mtr_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    instrument_uid: Mapped[str] = mapped_column(String(64), nullable=False)
    interval: Mapped[str] = mapped_column(String(32), nullable=False)
    model_key: Mapped[str] = mapped_column(String(128), nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    time_limit_sec: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    autogluon_preset: Mapped[str] = mapped_column(Text, nullable=False, default="medium_quality")

    training_samples: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    candles_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    candles_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    best_model: Mapped[str | None] = mapped_column(String(256), nullable=True, default=None)
    mape: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True, default=None)
    leaderboard: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )


class ModelPrediction(Base):
    """A forecast of the serving model, later matched with the real close price."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_uid",
            "interval",
            "model_version",
            "last_candle_time",
            name="uq_model_prediction",
        ),
        Index("idx_mp_instrument_interval_version", "instrument_uid", "interval", "model_version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    instrument_uid: Mapped[str] = mapped_column(String(64), nullable=False)
    interval: Mapped[str] = mapped_column(String(32), nullable=False)
    # trained_at of the model that made the forecast
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    last_candle_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_close: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    predicted_close: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    actual_close: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True, default=None)
    abs_pct_error: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Table model_training_runs is created")


async def close_engine() -> None:
    await engine.dispose()
    logger.info("SQLAlchemy engine disposed")


def get_session() -> AsyncSession:
    return AsyncSessionLocal()
