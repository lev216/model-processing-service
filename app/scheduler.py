import asyncio
import time
from datetime import datetime

import mlflow
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.metrics import (
    SCHEDULER_JOB_DURATION_SECONDS,
    SCHEDULER_JOB_RUNS_TOTAL,
    SCHEDULER_JOBS_ACTIVE,
)
from app.ml_service import predict_next_close
from app.repository import fetch_candles, save_prediction

logger = structlog.get_logger(__name__)

scheduler = AsyncIOScheduler()

_MODEL_CACHE = {}


async def execute_scheduled_inference(instrument_uid: str, interval: str, run_id: str):
    logger.info("Starting scheduled inference", instrument=instrument_uid, interval=interval)

    SCHEDULER_JOBS_ACTIVE.set(len(scheduler.get_jobs()))

    job_start = time.monotonic()
    status = "success"

    try:
        df = await fetch_candles(
            instrument_uid=instrument_uid,
            interval=interval,
            limit=100,
        )

        result = await asyncio.to_thread(
            predict_next_close,
            interval=interval,
            df=df,
            instrument_uid=instrument_uid,
        )

        try:
            await save_prediction(result)
        except Exception as e:
            logger.warning("Prediction was not saved", error=str(e))

        mlflow.set_experiment(f"Inference_{instrument_uid}_{interval}")

        run_name = f"pred_{datetime.now().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model_used": result["model_used"],
                    "prediction_date": result["prediction_date"],
                    "last_candle_time": result["last_candle_time"],
                    "train_run_id": run_id,
                }
            )

            mlflow.log_metrics(
                {
                    "predicted_close": result["predicted_close"],
                    "last_close": result["last_close"],
                    "change_pct": result["change_pct"],
                }
            )

            if "confidence_interval" in result:
                mlflow.log_metrics(
                    {
                        "lower_90": result["confidence_interval"]["lower_90"],
                        "upper_90": result["confidence_interval"]["upper_90"],
                    }
                )

        logger.info("Scheduled prediction success", result=result)

    except Exception as e:
        status = "error"
        logger.exception("Scheduled inference failed", error=str(e))

    finally:
        duration = time.monotonic() - job_start
        SCHEDULER_JOB_RUNS_TOTAL.labels(
            instrument_uid=instrument_uid, interval=interval, status=status
        ).inc()
        SCHEDULER_JOB_DURATION_SECONDS.labels(
            instrument_uid=instrument_uid, interval=interval
        ).observe(duration)
        SCHEDULER_JOBS_ACTIVE.set(len(scheduler.get_jobs()))
