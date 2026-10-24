"""Retrain a model and replace the serving one only if the new model is not worse."""

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import structlog

from app.config import settings
from app.ml.features import add_technical_features
from app.ml.model_store import _model_key, _model_path
from app.ml.training import train_model

logger = structlog.get_logger(__name__)


def _candidate_path(instrument_uid: str, interval: str) -> Path:
    return Path(settings.models_dir) / f"{_model_key(instrument_uid, interval)}__candidate"


def _previous_path(instrument_uid: str, interval: str) -> Path:
    return Path(settings.models_dir) / f"{_model_key(instrument_uid, interval)}__previous"


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual) / np.abs(actual)))


def _holdout_forecasts(model_path: Path, frame: pd.DataFrame, start: int) -> np.ndarray:
    """One-step forecasts for rows start..end, each made only from the rows before it."""
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    meta = json.loads((model_path / "meta.json").read_text(encoding="utf-8"))
    covariates = meta.get("past_covariates", meta.get("known_covariates", []))
    predictor = TimeSeriesPredictor.load(str(model_path))

    forecasts = []
    for end in range(start, len(frame)):
        ts_df = TimeSeriesDataFrame.from_data_frame(
            frame.iloc[:end][["item_id", "timestamp", "target"] + covariates],
            id_column="item_id",
            timestamp_column="timestamp",
        )
        forecasts.append(float(predictor.predict(ts_df)["mean"].iloc[0]))
    return np.array(forecasts)


def compare_on_holdout(
    instrument_uid: str, df: pd.DataFrame, candidate: Path, production: Path, points: int
) -> dict:
    """MAPE of the candidate, the serving model and the naive forecast on the last candles."""
    frame = add_technical_features(df)
    frame["item_id"] = instrument_uid
    frame = frame.rename(columns={"time": "timestamp", settings.target: "target"})

    start = len(frame) - points
    target = frame["target"].to_numpy()
    actual = target[start:]

    return {
        "points": points,
        "candidate_mape": _mape(actual, _holdout_forecasts(candidate, frame, start)),
        "production_mape": _mape(actual, _holdout_forecasts(production, frame, start)),
        "naive_mape": _mape(actual, target[start - 1 : -1]),
    }


def _replace_serving_model(candidate: Path, production: Path, previous: Path) -> None:
    """The serving model is kept as `previous`, so it can be restored by hand."""
    shutil.rmtree(previous, ignore_errors=True)
    production.rename(previous)
    candidate.rename(production)


def _log_decision(run_id: str, comparison: dict, decision: str) -> None:
    client = mlflow.MlflowClient()
    for key, value in comparison.items():
        client.log_metric(run_id, f"holdout_{key}", value)
    client.set_tag(run_id, "decision", decision)


def retrain_model(
    interval: str,
    df: pd.DataFrame,
    instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
    time_limit: int | None = None,
) -> dict:
    """Train a new model and publish it only if it is not worse than the serving one.

    Returns the training meta plus `status` ("done" when the model was replaced, "rejected"
    when the serving model was kept), `comparison` and, if rejected, `reason`.
    """
    production = _model_path(instrument_uid, interval)
    if not (production / "meta.json").exists():
        meta = train_model(interval, df, instrument_uid=instrument_uid, time_limit=time_limit)
        return {**meta, "status": "done", "comparison": None}

    points = settings.retrain_holdout_size
    if len(df) - points < settings.min_candles_train:
        raise ValueError(
            f"Not enough candles to retrain: {len(df)}, need {settings.min_candles_train} "
            f"for training plus {points} to compare with the serving model."
        )

    df = df.sort_values("time").reset_index(drop=True)
    candidate = _candidate_path(instrument_uid, interval)
    shutil.rmtree(candidate, ignore_errors=True)

    try:
        meta = train_model(
            interval,
            df.iloc[:-points],
            instrument_uid=instrument_uid,
            time_limit=time_limit,
            path=candidate,
            register=False,
        )
        comparison = compare_on_holdout(instrument_uid, df, candidate, production, points)
    except Exception:
        shutil.rmtree(candidate, ignore_errors=True)
        raise

    limit = comparison["production_mape"] * (1 + settings.retrain_mape_tolerance)
    log = logger.bind(instrument_uid=instrument_uid, interval=interval, **comparison)

    if comparison["candidate_mape"] > limit:
        shutil.rmtree(candidate, ignore_errors=True)
        _log_decision(meta["mlflow_run_id"], comparison, "rejected")
        reason = (
            f"New model MAPE {comparison['candidate_mape']:.6f} is worse than "
            f"the serving model MAPE {comparison['production_mape']:.6f} "
            f"on the last {points} candles."
        )
        log.info("Retrained model rejected, serving model kept")
        return {**meta, "status": "rejected", "comparison": comparison, "reason": reason}

    _replace_serving_model(candidate, production, _previous_path(instrument_uid, interval))
    _log_decision(meta["mlflow_run_id"], comparison, "published")
    try:
        mlflow.register_model(
            f"runs:/{meta['mlflow_run_id']}/model", f"Price_Predictor_{instrument_uid}_{interval}"
        )
    except Exception as e:
        # the model on disk is what serves predictions, so a registry failure is not fatal
        log.warning("Model registration in MLflow failed", error=str(e))

    log.info("Retrained model published")
    return {**meta, "status": "done", "comparison": comparison}
