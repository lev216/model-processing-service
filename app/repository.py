from datetime import UTC, datetime

import pandas as pd
import structlog
from sqlalchemy import desc, select, text
from sqlalchemy.dialects.postgresql import insert

from app.database import AsyncSessionLocal, ModelPrediction, ModelTrainingRun

logger = structlog.get_logger(__name__)


async def fetch_candles(
    instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
    interval: str = "1DAY",
    limit: int | None = None,
) -> pd.DataFrame:
    if limit is not None:
        sql = text("""
            SELECT time,
                   open::float,
                   high::float,
                   low::float,
                   close::float,
                   volume::float
            FROM (
                SELECT time, open, high, low, close, volume
                FROM candles
                WHERE instrument_uid = :uid
                  AND interval        = :interval
                  AND is_complete     = TRUE
                ORDER BY time DESC
                LIMIT :limit
            ) sub
            ORDER BY time ASC
            """)
        params = {"uid": instrument_uid, "interval": interval, "limit": limit}
    else:
        sql = text("""
            SELECT time,
                   open::float,
                   high::float,
                   low::float,
                   close::float,
                   volume::float
            FROM candles
            WHERE instrument_uid = :uid
              AND interval        = :interval
              AND is_complete     = TRUE
            ORDER BY time ASC
            """)
        params = {"uid": instrument_uid, "interval": interval}

    async with AsyncSessionLocal() as session:
        result = await session.execute(sql, params)
        rows = result.fetchall()

    if not rows:
        raise ValueError(f"No result: instrument_uid={instrument_uid!r}, interval={interval!r}. ")

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)

    logger.info(
        f"Loaded {len(df)} candles: instrument_uid={instrument_uid!r}, "
        f"interval={interval!r}, range: {df['time'].iloc[0].date()} — {df['time'].iloc[-1].date()}"
    )
    return df


async def fetch_available_instruments() -> list[dict]:
    sql = text("""
        SELECT
            instrument_uid,
            figi,
            interval,
            COUNT(*)  AS candle_count,
            MIN(time) AS first_candle,
            MAX(time) AS last_candle
        FROM candles
        WHERE is_complete = TRUE
        GROUP BY instrument_uid, figi, interval
        ORDER BY candle_count DESC
    """)
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql)
        rows = result.fetchall()

    return [
        {
            "instrument_uid": r.instrument_uid,
            "figi": r.figi,
            "interval": r.interval,
            "candle_count": r.candle_count,
            "first_candle": r.first_candle.isoformat() if r.first_candle else None,
            "last_candle": r.last_candle.isoformat() if r.last_candle else None,
        }
        for r in rows
    ]


async def audit_start(
    instrument_uid: str,
    interval: str,
    model_key: str,
    time_limit_sec: int | None,
    autogluon_preset: str = "medium_quality",
) -> int:
    run = ModelTrainingRun(
        instrument_uid=instrument_uid,
        interval=interval,
        model_key=model_key,
        status="running",
        time_limit_sec=time_limit_sec,
        autogluon_preset=autogluon_preset,
        started_at=datetime.now(UTC),
    )
    async with AsyncSessionLocal() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run.id


async def audit_done(
    run_id: int,
    training_samples: int,
    candles_from: str,
    candles_to: str,
    best_model: str,
    mape: float,
    leaderboard: list[dict],
    status: str = "done",
    error_message: str | None = None,
) -> None:
    """Finish a run that trained a model; status "rejected" means the serving model was kept."""
    async with AsyncSessionLocal() as session:
        run = await session.get(ModelTrainingRun, run_id)
        if run is None:
            logger.error("audit_done: run not found", run_id=run_id)
            return
        run.status = status
        run.error_message = error_message
        run.training_samples = training_samples
        run.candles_from = datetime.fromisoformat(candles_from)
        run.candles_to = datetime.fromisoformat(candles_to)
        run.best_model = best_model
        run.mape = mape
        run.leaderboard = leaderboard
        run.finished_at = datetime.now(UTC)
        await session.commit()


async def audit_error(run_id: int, error_message: str) -> None:
    async with AsyncSessionLocal() as session:
        run = await session.get(ModelTrainingRun, run_id)
        if run is None:
            logger.error("audit_error: run not found", run_id=run_id)
            return
        run.status = "error"
        run.error_message = error_message[:2000]
        run.finished_at = datetime.now(UTC)
        await session.commit()


async def fetch_training_runs(
    instrument_uid: str | None = None,
    interval: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    stmt = select(ModelTrainingRun).order_by(desc(ModelTrainingRun.started_at))

    if instrument_uid:
        stmt = stmt.where(ModelTrainingRun.instrument_uid == instrument_uid)
    if interval:
        stmt = stmt.where(ModelTrainingRun.interval == interval)
    if status:
        stmt = stmt.where(ModelTrainingRun.status == status)

    stmt = stmt.limit(limit).offset(offset)

    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        runs = result.scalars().all()

    return [_run_to_dict(r) for r in runs]


async def fetch_training_run_by_id(run_id: int) -> dict | None:
    async with AsyncSessionLocal() as session:
        run = await session.get(ModelTrainingRun, run_id)
    return _run_to_dict(run) if run else None


async def fetch_active_run(instrument_uid: str, interval: str) -> int | None:
    stmt = (
        select(ModelTrainingRun.id)
        .where(
            ModelTrainingRun.instrument_uid == instrument_uid,
            ModelTrainingRun.interval == interval,
            ModelTrainingRun.status == "running",
        )
        .limit(1)
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
    return row


async def fetch_best_models() -> list[dict]:
    sql = text("""
        SELECT DISTINCT ON (instrument_uid, interval)
            id, instrument_uid, interval, model_key,
            best_model, mape, training_samples,
            candles_from, candles_to,
            started_at, finished_at,
            EXTRACT(EPOCH FROM (finished_at - started_at))::INT AS duration_sec
        FROM model_training_runs
        WHERE status = 'done'
        ORDER BY instrument_uid, interval, started_at DESC
    """)
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql)
        rows = result.mappings().fetchall()

    out = []
    for r in rows:
        d = dict(r)
        for col in ("candles_from", "candles_to", "started_at", "finished_at"):
            if d[col] is not None:
                d[col] = d[col].isoformat()
        out.append(d)
    return out


def _run_to_dict(run: ModelTrainingRun) -> dict:
    def _iso(v):
        return v.isoformat() if v else None

    duration = None
    if run.started_at and run.finished_at:
        duration = int((run.finished_at - run.started_at).total_seconds())

    return {
        "id": run.id,
        "instrument_uid": run.instrument_uid,
        "interval": run.interval,
        "model_key": run.model_key,
        "status": run.status,
        "error_message": run.error_message,
        "time_limit_sec": run.time_limit_sec,
        "autogluon_preset": run.autogluon_preset,
        "training_samples": run.training_samples,
        "candles_from": _iso(run.candles_from),
        "candles_to": _iso(run.candles_to),
        "best_model": run.best_model,
        "mape": float(run.mape) if run.mape is not None else None,
        "leaderboard": run.leaderboard,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "duration_sec": duration,
    }


async def fetch_last_run_started_at(instrument_uid: str, interval: str) -> datetime | None:
    stmt = (
        select(ModelTrainingRun.started_at)
        .where(
            ModelTrainingRun.instrument_uid == instrument_uid,
            ModelTrainingRun.interval == interval,
        )
        .order_by(desc(ModelTrainingRun.started_at))
        .limit(1)
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def has_candles_since(instrument_uid: str, interval: str, since: datetime) -> bool:
    sql = text("""
        SELECT EXISTS (
            SELECT 1
            FROM candles
            WHERE instrument_uid = :uid
              AND interval        = :interval
              AND is_complete     = TRUE
              AND time > :since
        )
    """)
    params = {"uid": instrument_uid, "interval": interval, "since": since}
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql, params)
        return bool(result.scalar_one())


async def save_prediction(result: dict) -> None:
    """Store a forecast; repeated forecasts from the same candle and model overwrite each other."""
    last_candle_time = datetime.fromisoformat(result["last_candle_time"])
    if last_candle_time.tzinfo is None:
        last_candle_time = last_candle_time.replace(tzinfo=UTC)
    stmt = (
        insert(ModelPrediction)
        .values(
            instrument_uid=result["instrument_uid"],
            interval=result["interval"],
            model_version=result["model_version"],
            last_candle_time=last_candle_time,
            last_close=result["last_close"],
            predicted_close=result["predicted_close"],
            predicted_at=datetime.now(UTC),
        )
        .on_conflict_do_update(
            constraint="uq_model_prediction",
            set_={"predicted_close": result["predicted_close"], "predicted_at": datetime.now(UTC)},
        )
    )
    async with AsyncSessionLocal() as session:
        await session.execute(stmt)
        await session.commit()


async def fill_prediction_actuals(instrument_uid: str, interval: str) -> int:
    """Match open forecasts with the first complete candle after the one they were made from."""
    sql = text("""
        UPDATE model_predictions p
        SET actual_close  = nxt.close,
            abs_pct_error = ABS(p.predicted_close - nxt.close) / NULLIF(nxt.close, 0)
        FROM (
            SELECT p2.id,
                   (SELECT c.close
                    FROM candles c
                    WHERE c.instrument_uid = p2.instrument_uid
                      AND c.interval        = p2.interval
                      AND c.is_complete     = TRUE
                      AND c.time > p2.last_candle_time
                    ORDER BY c.time
                    LIMIT 1) AS close
            FROM model_predictions p2
            WHERE p2.instrument_uid = :uid
              AND p2.interval        = :interval
              AND p2.actual_close IS NULL
        ) nxt
        WHERE p.id = nxt.id
          AND nxt.close IS NOT NULL
    """)
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql, {"uid": instrument_uid, "interval": interval})
        await session.commit()
        return result.rowcount


async def fetch_realized_error(
    instrument_uid: str, interval: str, model_version: str, window: int
) -> tuple[float | None, int]:
    """MAPE of the last `window` matched forecasts of a model and how many there were."""
    sql = text("""
        SELECT AVG(abs_pct_error) AS mape, COUNT(*) AS points
        FROM (
            SELECT abs_pct_error
            FROM model_predictions
            WHERE instrument_uid = :uid
              AND interval        = :interval
              AND model_version   = :version
              AND abs_pct_error IS NOT NULL
            ORDER BY last_candle_time DESC
            LIMIT :window
        ) recent
    """)
    params = {
        "uid": instrument_uid,
        "interval": interval,
        "version": model_version,
        "window": window,
    }
    async with AsyncSessionLocal() as session:
        row = (await session.execute(sql, params)).one()
    return (float(row.mape) if row.mape is not None else None), int(row.points)
