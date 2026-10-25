"""Retraining runs: on request and on the nightly schedule."""

import asyncio
import json
from pathlib import Path

import structlog
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.ml_service import _PREDICTOR_CACHE, retrain_model
from app.repository import (
    audit_done,
    audit_error,
    audit_start,
    fetch_active_run,
    fetch_candles,
    fetch_last_run_started_at,
    has_candles_since,
)
from app.scheduler import scheduler

logger = structlog.get_logger(__name__)

RETRAIN_JOB_ID = "retrain_all_models"


class TrainingInProgress(Exception):
    pass


class NoCandles(ValueError):
    pass


async def run_retrain(instrument_uid: str, interval: str, time_limit: int | None = None) -> dict:
    """Retrain one pair, record the run and switch predictions to the new model if published."""
    if await fetch_active_run(instrument_uid, interval):
        raise TrainingInProgress("Training already running")

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


def schedule_retrain() -> None:
    if not settings.retrain_cron:
        logger.info("Scheduled retrain is disabled")
        return
    trigger = CronTrigger.from_crontab(settings.retrain_cron, timezone=settings.retrain_timezone)
    scheduler.add_job(
        retrain_all_models,
        trigger=trigger,
        id=RETRAIN_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    logger.info(
        "Scheduled retrain registered",
        cron=settings.retrain_cron,
        timezone=settings.retrain_timezone,
    )
