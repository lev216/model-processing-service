"""Load cached models and produce price forecasts."""

import json
import time
from datetime import timedelta

import mlflow
import pandas as pd
import structlog

from app.config import settings
from app.metrics import (
    PREDICTED_CHANGE_PCT,
    PREDICTED_CLOSE_PRICE,
    PREDICTION_DURATION_SECONDS,
    PREDICTION_TOTAL,
)
from app.ml.features import add_technical_features
from app.ml.model_store import _model_path

logger = structlog.get_logger(__name__)

_PREDICTOR_CACHE = {}


@mlflow.trace(name="predict_next_close")
def predict_next_close(
    interval: str,
    df: pd.DataFrame,
    instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
) -> dict:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError:
        raise RuntimeError("AutoGluon not installed.")

    path = _model_path(instrument_uid, interval)
    if not path.exists():
        PREDICTION_TOTAL.labels(
            instrument_uid=instrument_uid, interval=interval, status="error"
        ).inc()
        raise FileNotFoundError(
            f"Model not found: instrument_uid={instrument_uid!r}, interval={interval!r}."
        )

    with open(path / "meta.json") as f:
        meta = json.load(f)

    if len(df) < settings.min_candles_predict:
        PREDICTION_TOTAL.labels(
            instrument_uid=instrument_uid, interval=interval, status="error"
        ).inc()
        raise ValueError(f"Not enough candles for predictions: {len(df)}")

    predict_start = time.monotonic()

    df = add_technical_features(df)

    last_close = float(df["close"].iloc[-1])
    last_candle_time = df["time"].iloc[-1]

    df["item_id"] = instrument_uid
    df = df.rename(columns={"time": "timestamp", settings.target: "target"})

    trained_covariates = meta.get("past_covariates", meta.get("known_covariates", []))

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df[["item_id", "timestamp", "target"] + trained_covariates],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    cache_key = f"{instrument_uid}_{interval}"
    if cache_key not in _PREDICTOR_CACHE:
        logger.info("Loading model from disk into cache...", key=cache_key)
        _PREDICTOR_CACHE[cache_key] = TimeSeriesPredictor.load(str(path))

    predictor = _PREDICTOR_CACHE[cache_key]
    predictions = predictor.predict(ts_df)
    predicted_close = float(predictions["mean"].iloc[0])

    next_dt = last_candle_time + timedelta(days=1)
    if interval == "1DAY":
        while next_dt.weekday() >= 5:
            next_dt += timedelta(days=1)

    change_pct = ((predicted_close - last_close) / last_close) * 100

    duration = time.monotonic() - predict_start
    PREDICTION_DURATION_SECONDS.labels(instrument_uid=instrument_uid, interval=interval).observe(
        duration
    )
    PREDICTION_TOTAL.labels(
        instrument_uid=instrument_uid, interval=interval, status="success"
    ).inc()
    PREDICTED_CHANGE_PCT.labels(instrument_uid=instrument_uid, interval=interval).observe(
        change_pct
    )
    PREDICTED_CLOSE_PRICE.labels(instrument_uid=instrument_uid, interval=interval).set(
        predicted_close
    )

    result = {
        "instrument_uid": instrument_uid,
        "interval": interval,
        "predicted_close": round(predicted_close, 4),
        "prediction_date": next_dt.strftime("%Y-%m-%d"),
        "last_close": round(last_close, 4),
        "last_candle_time": last_candle_time.isoformat(),
        "change_pct": round(change_pct, 2),
        "model_used": meta["best_model"],
    }

    if "0.1" in predictions.columns and "0.9" in predictions.columns:
        result["confidence_interval"] = {
            "lower_90": round(float(predictions["0.1"].iloc[0]), 4),
            "upper_90": round(float(predictions["0.9"].iloc[0]), 4),
        }

    return result
