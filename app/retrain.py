"""Retraining runs: on request and on the nightly schedule."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.metrics import MODEL_REALIZED_MAPE
from app.ml_service import _PREDICTOR_CACHE, _model_path, retrain_model
from app.repository import (
    audit_done,
    audit_error,
    audit_start,
    fetch_active_run,
    fetch_candles,
    fetch_last_run_started_at,
    fetch_realized_error,
    fill_prediction_actuals,
    has_candles_since,
    save_prediction,
)
from app.scheduler import scheduler

logger = structlog.get_logger(__name__)

RETRAIN_JOB_ID = "retrain_all_models"
ERROR_CHECK_JOB_ID = "check_realized_error"

# the nightly job, the error check and the API share one event loop: the lock keeps them from
# starting two trainings of the same pair before the first one is recorded as running
_LOCKS: dict[str, asyncio.Lock] = {}


class TrainingInProgress(Exception):
    pass


class NoCandles(ValueError):
    pass


async def run_retrain(instrument_uid: str, interval: str, time_limit: int | None = None) -> dict:
    """Retrain one pair, record the run and switch predictions to the new model if published."""
    lock = _LOCKS.setdefault(f"{instrument_uid}__{interval}", asyncio.Lock())
    if lock.locked() or await fetch_active_run(instrument_uid, interval):
        raise TrainingInProgress("Training already running")
    async with lock:
        return await _retrain(instrument_uid, interval, time_limit)


async def _retrain(instrument_uid: str, interval: str, time_limit: int | None) -> dict:
    try:
        df = await fetch_candles(instrument_uid, interval)
    except ValueError as e:
        raise NoCandles(str(e)) from e

    run_id = await audit_start(
        instrument_uid=instrument_uid,
        interval=interval,
        model_key=f"{instrument_uid}__{interval}",
        time_limit_sec=time_limit,
    )

    try:
        result = await asyncio.to_thread(
            retrain_model,
            instrument_uid=instrument_uid,
            interval=interval,
            df=df,
            time_limit=time_limit,
        )
    except Exception as e:
        await audit_error(run_id=run_id, error_message=str(e))
        raise

    await audit_done(
        run_id=run_id,
        training_samples=result["training_samples"],
        candles_from=result["candles_from"],
        candles_to=result["candles_to"],
        best_model=result["best_model"],
        mape=result["mape"],
        leaderboard=result["leaderboard"],
        status=result["status"],
        error_message=result.get("reason"),
    )

    if result["status"] == "done":
        _PREDICTOR_CACHE.pop(f"{instrument_uid}_{interval}", None)

    return {
        "run_id": run_id,
        "status": result["status"],
        "best_model": result["best_model"],
        "mape": result["mape"],
        "comparison": result["comparison"],
        "reason": result.get("reason"),
    }


def trained_pairs() -> list[tuple[str, str]]:
    """Instrument/interval pairs that have a serving model on disk."""
    pairs = []
    models_dir = Path(settings.models_dir)
    if not models_dir.exists():
        return pairs
    for path in sorted(models_dir.iterdir()):
        meta_path = path / "meta.json"
        if path.name.endswith(("__candidate", "__previous")) or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("instrument_uid") and meta.get("interval"):
            pairs.append((meta["instrument_uid"], meta["interval"]))
    return pairs


async def retrain_all_models() -> None:
    """Nightly job: retrain every trained pair that got new candles since its last run."""
    for instrument_uid, interval in trained_pairs():
        log = logger.bind(instrument_uid=instrument_uid, interval=interval)
        try:
            last_run = await fetch_last_run_started_at(instrument_uid, interval)
            if last_run and not await has_candles_since(instrument_uid, interval, last_run):
                log.info("Scheduled retrain skipped: no new candles since the last run")
                continue
            result = await run_retrain(instrument_uid, interval)
            log.info("Scheduled retrain finished", **result)
        except TrainingInProgress:
            log.info("Scheduled retrain skipped: training already running")
        except Exception as e:
            # one failed pair must not stop the others; the run audit keeps the error
            log.exception("Scheduled retrain failed", error=str(e))


async def record_prediction(result: dict) -> None:
    """Keep the forecast to compare it with the real price later; never fails the prediction."""
    try:
        await save_prediction(result)
    except Exception as e:
        logger.warning("Prediction was not saved", error=str(e))


def _serving_model_version(instrument_uid: str, interval: str) -> str | None:
    meta_path = _model_path(instrument_uid, interval) / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("trained_at")


async def check_realized_error(instrument_uid: str, interval: str) -> bool:
    """Retrain the pair if the real MAPE of its serving model is above the limit.

    Returns True if a retraining was started.
    """
    log = logger.bind(instrument_uid=instrument_uid, interval=interval)
    await fill_prediction_actuals(instrument_uid, interval)

    version = _serving_model_version(instrument_uid, interval)
    if version is None:
        return False
    mape, points = await fetch_realized_error(
        instrument_uid, interval, version, settings.realized_error_window
    )
    if mape is None:
        return False
    MODEL_REALIZED_MAPE.labels(instrument_uid=instrument_uid, interval=interval).set(mape)

    if points < settings.realized_error_min_points or mape <= settings.retrain_max_realized_mape:
        return False

    # a rejected retrain leaves the same model and the same error: wait before trying again
    last_run = await fetch_last_run_started_at(instrument_uid, interval)
    cooldown = timedelta(hours=settings.retrain_cooldown_hours)
    if last_run and datetime.now(UTC) - last_run < cooldown:
        log.info("Real error is high, retrain postponed by cooldown", realized_mape=mape)
        return False

    log.warning(
        "Real error is above the limit, retraining",
        realized_mape=mape,
        points=points,
        limit=settings.retrain_max_realized_mape,
    )
    result = await run_retrain(instrument_uid, interval)
    log.info("Retrain after error growth finished", **result)
    return True


async def check_realized_errors() -> None:
    """Hourly job: check the real error of every trained pair."""
    for instrument_uid, interval in trained_pairs():
        try:
            await check_realized_error(instrument_uid, interval)
        except TrainingInProgress:
            logger.info(
                "Error check skipped: training already running", instrument_uid=instrument_uid
            )
        except Exception as e:
            logger.exception(
                "Real error check failed",
                instrument_uid=instrument_uid,
                interval=interval,
                error=str(e),
            )


def _add_cron_job(job_id: str, func, cron: str) -> None:
    scheduler.add_job(
        func,
        trigger=CronTrigger.from_crontab(cron, timezone=settings.retrain_timezone),
        id=job_id,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    logger.info(
        "Retrain job registered", job_id=job_id, cron=cron, timezone=settings.retrain_timezone
    )


def schedule_retrain() -> None:
    """Register the nightly retrain and the hourly real error check; empty crons disable them."""
    if settings.retrain_cron:
        _add_cron_job(RETRAIN_JOB_ID, retrain_all_models, settings.retrain_cron)
    else:
        logger.info("Scheduled retrain is disabled")
    if settings.retrain_error_check_cron:
        _add_cron_job(ERROR_CHECK_JOB_ID, check_realized_errors, settings.retrain_error_check_cron)
    else:
        logger.info("Real error check is disabled")
