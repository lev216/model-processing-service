import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import structlog

from app.config import settings
from app.metrics import (
    MODEL_BEST_MAPE,
    MODEL_LEADERBOARD_POSITION,
    MODEL_TRAINING_SAMPLES,
    PREDICTED_CHANGE_PCT,
    PREDICTED_CLOSE_PRICE,
    PREDICTION_DURATION_SECONDS,
    PREDICTION_TOTAL,
    TRAINING_DURATION_SECONDS,
    TRAINING_RUNS_TOTAL,
)

logger = structlog.get_logger(__name__)
mlflow.set_tracking_uri(settings.mlflow_uri)


def _model_key(instrument_uid: str, interval: str) -> str:
    return f"{instrument_uid}__{interval}"


def _model_path(instrument_uid: str, interval: str) -> Path:
    return Path(settings.models_dir) / _model_key(instrument_uid, interval)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["body"] = df["close"] - df["open"]
    df["shadow_upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["shadow_lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["hl_spread"] = df["high"] - df["low"]
    df["body_pct"] = df["body"] / (df["open"] + 1e-9)

    for w in settings.ma_windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"ma_{w}_dist"] = (df["close"] - df[f"ma_{w}"]) / (df[f"ma_{w}"] + 1e-9)

    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(settings.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(settings.rsi_period).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-9)

    for lag in [1, 2, 3, 5]:
        df[f"ret_{lag}d"] = df["close"].pct_change(lag)

    return df.dropna().reset_index(drop=True)


KNOWN_COVARIATES = [
    "open",
    "high",
    "low",
    "volume",
    "body",
    "shadow_upper",
    "shadow_lower",
    "hl_spread",
    "rsi",
    "macd",
    "bb_pct",
    "vol_ratio",
    "ret_1d",
    "ret_2d",
    "ret_5d",
]

PANDAS_FREQ_MAP = {
    "1MIN": "1min",
    "5MIN": "5min",
    "15MIN": "15min",
    "30MIN": "30min",
    "1HOUR": "1H",
    "2HOUR": "2H",
    "4HOUR": "4H",
    "1DAY": "1D",
    "1WEEK": "1W",
    "1MONTH": "1M",
}


def train_model(
    interval: str,
    df: pd.DataFrame,
    instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
    time_limit: int | None = None,
) -> dict:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError:
        raise RuntimeError("AutoGluon not installed. Run pip install autogluon.timeseries")

    if len(df) < settings.min_candles_train:
        raise ValueError(f"Not enough candles to train: {len(df)} < {settings.min_candles_train}. ")

    mlflow.set_experiment(f"Price_Predictor_{instrument_uid}_{interval}")

    train_start = time.monotonic()

    try:
        with mlflow.start_run(run_name=f"{instrument_uid}_{interval}") as run:
            time_limit = time_limit or settings.training_time_limit

            mlflow.log_params(
                {
                    "instrument_uid": instrument_uid,
                    "interval": interval,
                    "time_limit": time_limit,
                    "training_samples": len(df),
                    "prediction_length": settings.prediction_length,
                    "min_candles_train": settings.min_candles_train,
                    "known_covariates": str(KNOWN_COVARIATES),
                }
            )

            candles_from = (
                df["time"].min().isoformat() if "time" in df else df.index.min().isoformat()
            )
            candles_to = (
                df["time"].max().isoformat() if "time" in df else df.index.max().isoformat()
            )
            mlflow.set_tags(
                {
                    "dataset_start": candles_from,
                    "dataset_end": candles_to,
                    "auto_gluon_version": "1.4.0",
                }
            )

            df = add_technical_features(df)

            df["item_id"] = instrument_uid
            df = df.rename(columns={"time": "timestamp", settings.target: "target"})

            exclude_cols = {
                "item_id",
                "timestamp",
                "target",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
            dynamic_covariates = [col for col in df.columns if col not in exclude_cols]

            mlflow.log_dict({"features": dynamic_covariates}, "features_list.json")

            mlflow.log_param("total_features", len(dynamic_covariates))

            mlflow.log_params(
                {
                    "time_limit": time_limit,
                    "prediction_length": settings.prediction_length,
                    "min_candles_train": settings.min_candles_train,
                }
            )

            candles_from = df["timestamp"].min().isoformat()
            candles_to = df["timestamp"].max().isoformat()
            mlflow.set_tags(
                {
                    "dataset_start": candles_from,
                    "dataset_end": candles_to,
                    "auto_gluon_version": "1.4.0",
                }
            )

            ts_df = TimeSeriesDataFrame.from_data_frame(
                df[["item_id", "timestamp", "target"] + dynamic_covariates],
                id_column="item_id",
                timestamp_column="timestamp",
            )

            path = _model_path(instrument_uid, interval)

            predictor = TimeSeriesPredictor(
                path=str(path),
                prediction_length=settings.prediction_length,
                target="target",
                eval_metric="MAPE",
                freq=PANDAS_FREQ_MAP[interval],
            )
            predictor.fit(ts_df, time_limit=time_limit, presets="medium_quality")

            leaderboard = predictor.leaderboard(ts_df, silent=True)
            best_model = leaderboard.iloc[0]["model"]

            best_mape = abs(float(leaderboard.iloc[0]["score_val"]))

            mlflow.log_metric("best_mape", best_mape)
            mlflow.log_param("best_model_name", best_model)
            mlflow.log_metric("total_models_trained", len(leaderboard))

            for _, row in leaderboard.iterrows():
                m_name = str(row["model"])
                clean_name = m_name.replace("[", "_").replace("]", "").replace(" ", "_")

                if pd.notnull(row.get("score_val")):
                    mlflow.log_metric(f"mape_{clean_name}", abs(float(row["score_val"])))

                if pd.notnull(row.get("fit_time")):
                    mlflow.log_metric(f"fit_time_{clean_name}", float(row["fit_time"]))

            meta = {
                "instrument_uid": instrument_uid,
                "interval": interval,
                "trained_at": datetime.now().isoformat(),
                "training_samples": len(df),
                "best_model": best_model,
                "mape": best_mape,
                "past_covariates": dynamic_covariates,
                "leaderboard": leaderboard.to_dict(orient="records"),
                "candles_from": candles_from,
                "candles_to": candles_to,
            }
            with open(path / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            leaderboard.to_csv(path / "leaderboard.csv", index=False)

            class AutoGluonTSWrapper(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    from autogluon.timeseries import TimeSeriesPredictor

                    self.model = TimeSeriesPredictor.load(context.artifacts["ag_model_path"])

                def predict(self, context, model_input):

                    return self.model.predict(model_input)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AutoGluonTSWrapper(),
                artifacts={"ag_model_path": str(path)},
                registered_model_name=f"Price_Predictor_{instrument_uid}_{interval}",
            )

            mlflow.log_artifacts(str(path), artifact_path=f"{instrument_uid}_{interval}")

            duration = time.monotonic() - train_start
            TRAINING_DURATION_SECONDS.labels(
                instrument_uid=instrument_uid, interval=interval
            ).observe(duration)
            TRAINING_RUNS_TOTAL.labels(
                instrument_uid=instrument_uid, interval=interval, status="done"
            ).inc()
            MODEL_BEST_MAPE.labels(
                instrument_uid=instrument_uid, interval=interval, best_model=best_model
            ).set(best_mape)
            MODEL_TRAINING_SAMPLES.labels(instrument_uid=instrument_uid, interval=interval).set(
                len(df)
            )

            for _, row in leaderboard.iterrows():
                if pd.notnull(row.get("score_val")):
                    MODEL_LEADERBOARD_POSITION.labels(
                        instrument_uid=instrument_uid,
                        interval=interval,
                        model_name=str(row["model"]),
                    ).set(abs(float(row["score_val"])))

            logger.info(
                f"[{instrument_uid} / {interval}] Train ended. "
                f"Best model: {best_model}, MAPE: {best_mape:.4f}. MLflow Run ID: {run.info.run_id}"
            )
            return meta

    except Exception:
        TRAINING_RUNS_TOTAL.labels(
            instrument_uid=instrument_uid, interval=interval, status="error"
        ).inc()
        raise


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
