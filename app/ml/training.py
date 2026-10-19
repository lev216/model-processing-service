"""Train and register AutoGluon time-series models."""

import json
import time
from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
import structlog

from app.config import settings
from app.metrics import (
    MODEL_BEST_MAPE,
    MODEL_LEADERBOARD_POSITION,
    MODEL_TRAINING_SAMPLES,
    TRAINING_DURATION_SECONDS,
    TRAINING_RUNS_TOTAL,
)
from app.ml.constants import KNOWN_COVARIATES, PANDAS_FREQ_MAP
from app.ml.features import add_technical_features
from app.ml.model_store import _model_path

logger = structlog.get_logger(__name__)


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
