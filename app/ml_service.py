"""Public model service interface used by API handlers and the scheduler."""

import mlflow

from app.config import settings
from app.ml import prediction
from app.ml.constants import KNOWN_COVARIATES, PANDAS_FREQ_MAP
from app.ml.features import add_technical_features
from app.ml.model_store import _model_key, _model_path
from app.ml.prediction import _PREDICTOR_CACHE, predict_next_close
from app.ml.retraining import retrain_model
from app.ml.training import train_model

mlflow.set_tracking_uri(settings.mlflow_uri)

__all__ = [
    "prediction",
    "_model_key",
    "_model_path",
    "_PREDICTOR_CACHE",
    "KNOWN_COVARIATES",
    "PANDAS_FREQ_MAP",
    "add_technical_features",
    "predict_next_close",
    "retrain_model",
    "train_model",
]
