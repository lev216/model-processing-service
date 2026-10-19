import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.ml_service import predict_next_close, train_model
from app.repository import (
    audit_done,
    audit_error,
    audit_start,
    fetch_active_run,
    fetch_best_models,
    fetch_candles,
    fetch_training_run_by_id,
    fetch_training_runs,
)
from app.scheduler import execute_scheduled_inference, scheduler

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1")


class TrainRequest(BaseModel):
    instrument_uid: str = Field(
        default="e6123145-9665-43e0-8413-cd61b8aa9b13", description="Unique instrument's id"
    )
    interval: str = Field(
        default="1DAY", description="Candles' timeframe: 1DAY, 1HOUR, 5MIN и т.д."
    )
    time_limit: int | None = Field(
        None,
        ge=30,
        le=3600,
        description="Training limit in seconds (config value is used by default)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "instrument_uid": "e6123145-9665-43e0-8413-cd61b8aa9b13",
                "interval": "1DAY",
                "time_limit": 120,
            }
        }
    }


class TrainResponse(BaseModel):
    instrument_uid: str
    interval: str
    status: str
    model_path: str
    training_samples: int
    best_model: str
    candle_range: dict


class PredictRequest(BaseModel):
    instrument_uid: str = Field(
        default="e6123145-9665-43e0-8413-cd61b8aa9b13", description="Unique instrument's id"
    )
    interval: str = Field(default="1DAY", description="Candles' timeframe")
    lookback: int | None = Field(
        None,
        ge=30,
        le=500,
        description="The amount of candles to fetch from db (config value is used by default)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "instrument_uid": "e6123145-9665-43e0-8413-cd61b8aa9b13",
                "interval": "1DAY",
                "lookback": 60,
            }
        }
    }


class PredictResponse(BaseModel):
    instrument_uid: str
    interval: str
    predicted_close: float
    prediction_date: str
    last_close: float
    last_candle_time: str
    change_pct: float
    confidence_interval: dict | None = None
    model_used: str


@router.post(
    "/",
    response_model=TrainResponse,
    summary="Train model with data from db",
    tags=["Train"],
)
async def train(request: TrainRequest):
    key = f"{request.instrument_uid}__{request.interval}"

    active_run_id = await fetch_active_run(request.instrument_uid, request.interval)

    if active_run_id:
        raise HTTPException(
            status_code=409,
            detail=f"Training for instrument_uid={request.instrument_uid!r} "
            f"interval={request.interval!r} already started.",
        )

    try:
        df = await fetch_candles(
            instrument_uid=request.instrument_uid,
            interval=request.interval,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    run_id = await audit_start(
        instrument_uid=request.instrument_uid,
        interval=request.interval,
        model_key=key,
        time_limit_sec=request.time_limit,
    )

    try:
        result = train_model(
            instrument_uid=request.instrument_uid,
            interval=request.interval,
            df=df,
            time_limit=request.time_limit,
        )
    except Exception as e:
        await audit_error(run_id=run_id, error_message=str(e))
        logger.error("training failed", key=key, run_id=run_id, error=str(e))
        if isinstance(e, ValueError):
            raise HTTPException(status_code=422, detail=str(e))
        if isinstance(e, RuntimeError):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error during the training: {e}")

    await audit_done(
        run_id=run_id,
        training_samples=result["training_samples"],
        candles_from=result["candles_from"],
        candles_to=result["candles_to"],
        best_model=result["best_model"],
        mape=result["mape"],
        leaderboard=result["leaderboard"],
    )

    return TrainResponse(
        run_id=run_id,
        instrument_uid=request.instrument_uid,
        interval=request.interval,
        status="done",
        model_path=f"models/{key}",
        training_samples=result["training_samples"],
        best_model=result["best_model"],
        mape=result["mape"],
        leaderboard=result["leaderboard"],
        candle_range={"from": result["candles_from"], "to": result["candles_to"]},
    )


@router.get("/status/{instrument_uid}/{interval}", summary="Training status", tags=["Train"])
async def training_status(instrument_uid: str, interval: str):
    from sqlalchemy import desc, select

    from app.database import AsyncSessionLocal, ModelTrainingRun

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ModelTrainingRun)
            .where(
                ModelTrainingRun.instrument_uid == instrument_uid,
                ModelTrainingRun.interval == interval,
            )
            .order_by(desc(ModelTrainingRun.started_at))
            .limit(1)
        )
        run = result.scalar_one_or_none()

    if not run:
        return {"instrument_uid": instrument_uid, "interval": interval, "status": "not_started"}

    duration = None
    if run.started_at and run.finished_at:
        duration = int((run.finished_at - run.started_at).total_seconds())

    return {
        "instrument_uid": instrument_uid,
        "interval": interval,
        "run_id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_sec": duration,
    }


@router.get("/runs", summary="History of training runs", tags=["Audit"])
async def list_runs(
    instrument_uid: str | None = Query(
        "e6123145-9665-43e0-8413-cd61b8aa9b13", description="Instrument filter"
    ),
    interval: str | None = Query("1DAY", description="Timeframe filter"),
    status: str | None = Query(None, description="running | done | error"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    runs = await fetch_training_runs(
        instrument_uid=instrument_uid,
        interval=interval,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"total": len(runs), "offset": offset, "runs": runs}


@router.get("/runs/{run_id}", summary="Concrete run details", tags=["Audit"])
async def get_run(run_id: int):
    run = await fetch_training_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run with run_id={run_id} not found.")
    return run


@router.get("/runs/{run_id}/leaderboard", summary="Run leaderboard", tags=["Audit"])
async def get_leaderboard(run_id: int):
    run = await fetch_training_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run with run_id={run_id} not found.")
    if run["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Leaderboard is available only for finished runs. "
            f"Current status: {run['status']}.",
        )
    return {
        "run_id": run_id,
        "instrument_uid": run["instrument_uid"],
        "interval": run["interval"],
        "trained_at": run["finished_at"],
        "leaderboard": run["leaderboard"],
    }


@router.get("/best", summary="Best models", tags=["Audit"])
async def best_models():
    models = await fetch_best_models()
    return {"models": models}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prediction of close price for the next day",
    tags=["Prediction"],
)
async def predict(request: PredictRequest):
    try:
        df = await fetch_candles(
            instrument_uid=request.instrument_uid,
            interval=request.interval,
            limit=request.lookback or 60,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = predict_next_close(
            instrument_uid=request.instrument_uid,
            interval=request.interval,
            df=df,
        )
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Prediction error [{request.instrument_uid}]: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@router.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@router.on_event("startup")
async def start_scheduler():
    if not scheduler.running:
        scheduler.start()


@router.on_event("shutdown")
async def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()


class ScheduleRequest(BaseModel):
    instrument_uid: str
    interval: str = "1DAY"
    run_id: str = Field(..., description="MLflow Run ID с обученной моделью")
    cron_expr: str = Field(..., description="CRON выражение, например '0 * * * *' (раз в час)")


@router.post("/inference/schedule", tags=["Inference"])
async def schedule_inference(req: ScheduleRequest):
    job_id = f"job_{req.instrument_uid}_{req.interval}"

    parts = req.cron_expr.split()
    if len(parts) != 5:
        raise HTTPException(400, "Invalid CRON expression")

    minute, hour, day, month, day_of_week = parts

    job = scheduler.add_job(
        execute_scheduled_inference,
        trigger="cron",
        minute=minute,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
        id=job_id,
        replace_existing=True,
        kwargs={
            "instrument_uid": req.instrument_uid,
            "interval": req.interval,
            "run_id": req.run_id,
        },
    )

    return {"status": "scheduled", "job_id": job.id, "next_run": job.next_run_time.isoformat()}


@router.get("/inference/jobs", tags=["Inference"])
async def list_jobs():
    jobs = scheduler.get_jobs()
    return {
        "jobs": [
            {"id": j.id, "next_run": j.next_run_time.isoformat() if j.next_run_time else None}
            for j in jobs
        ]
    }


@router.delete("/inference/jobs/{job_id}", tags=["Inference"])
async def remove_job(job_id: str):
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
        return {"status": "removed", "job_id": job_id}
    raise HTTPException(404, "Job not found")
