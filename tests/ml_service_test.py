import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def svc(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    service_file = project_root / "app" / "ml_service.py"

    fake_settings = SimpleNamespace(
        mlflow_uri="http://fake-mlflow",
        models_dir=str(tmp_path / "models"),
        ma_windows=[3, 5],
        rsi_period=5,
        min_candles_train=20,
        min_candles_predict=20,
        prediction_length=1,
        training_time_limit=10,
        target="close",
    )

    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    fake_config = types.ModuleType("app.config")
    fake_config.settings = fake_settings

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow_pyfunc = types.ModuleType("mlflow.pyfunc")

    class FakePythonModel:
        pass

    def trace(name=None):
        def decorator(func):
            return func

        return decorator

    class FakeRun:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow.set_tracking_uri = lambda *args, **kwargs: None
    fake_mlflow.set_experiment = lambda *args, **kwargs: None
    fake_mlflow.start_run = lambda *args, **kwargs: FakeRun()
    fake_mlflow.log_params = lambda *args, **kwargs: None
    fake_mlflow.set_tags = lambda *args, **kwargs: None
    fake_mlflow.log_dict = lambda *args, **kwargs: None
    fake_mlflow.log_param = lambda *args, **kwargs: None
    fake_mlflow.log_metric = lambda *args, **kwargs: None
    fake_mlflow.log_artifacts = lambda *args, **kwargs: None
    fake_mlflow.trace = trace

    fake_mlflow_pyfunc.PythonModel = FakePythonModel
    fake_mlflow_pyfunc.log_model = lambda *args, **kwargs: None
    fake_mlflow.pyfunc = fake_mlflow_pyfunc

    monkeypatch.setitem(sys.modules, "app.config", fake_config)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", fake_mlflow_pyfunc)

    install_fake_autogluon(monkeypatch)

    spec = importlib.util.spec_from_file_location("tested_price_service", service_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def raw_df():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "time": dates,
            "open": [100 + i for i in range(60)],
            "high": [102 + i for i in range(60)],
            "low": [99 + i for i in range(60)],
            "close": [101 + i for i in range(60)],
            "volume": [1000 + i * 10 for i in range(60)],
        }
    )


def install_fake_autogluon(monkeypatch, mean=150.0, low=145.0, high=155.0):
    fake_ts_module = types.ModuleType("autogluon.timeseries")

    class FakeTimeSeriesDataFrame:
        @classmethod
        def from_data_frame(cls, df, id_column, timestamp_column):
            return df.copy()

    class FakePredictor:
        load_calls = 0

        def __init__(
            self, path=None, prediction_length=None, target=None, eval_metric=None, freq=None
        ):
            self.path = path

        def fit(self, ts_df, time_limit=None, presets=None):
            Path(self.path).mkdir(parents=True, exist_ok=True)
            return self

        def leaderboard(self, ts_df, silent=True):
            return pd.DataFrame(
                [
                    {"model": "ModelA", "score_val": -0.123, "fit_time": 1.5},
                    {"model": "ModelB", "score_val": -0.456, "fit_time": 2.0},
                ]
            )

        def predict(self, ts_df):
            return pd.DataFrame(
                [
                    {
                        "mean": mean,
                        "0.1": low,
                        "0.9": high,
                    }
                ]
            )

        @classmethod
        def load(cls, path):
            cls.load_calls += 1
            return cls(path=path)

    monkeypatch.setitem(sys.modules, "autogluon.timeseries", fake_ts_module)
    fake_ts_module.TimeSeriesDataFrame = FakeTimeSeriesDataFrame
    fake_ts_module.TimeSeriesPredictor = FakePredictor

    return FakePredictor


def test_model_key(svc):
    assert svc._model_key("uid123", "1DAY") == "uid123__1DAY"


def test_model_path(svc):
    path = svc._model_path("uid123", "1DAY")
    assert str(path).endswith("uid123__1DAY")


def test_add_technical_features(svc, raw_df):
    result = svc.add_technical_features(raw_df)

    assert not result.empty
    assert result["time"].is_monotonic_increasing
    assert "body" in result.columns
    assert "shadow_upper" in result.columns
    assert "shadow_lower" in result.columns
    assert "hl_spread" in result.columns
    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "bb_pct" in result.columns
    assert "vol_ratio" in result.columns
    assert "ret_1d" in result.columns
    assert "ret_2d" in result.columns
    assert "ret_5d" in result.columns
    assert result.isna().sum().sum() == 0


def test_train_model_not_enough_candles(svc, raw_df):
    with pytest.raises(ValueError, match="Not enough candles to train"):
        svc.train_model(interval="1DAY", df=raw_df.head(5), instrument_uid="uid123")


def test_train_model_success(svc, raw_df, monkeypatch, tmp_path):
    install_fake_autogluon(monkeypatch)

    meta = svc.train_model(interval="1DAY", df=raw_df, instrument_uid="uid123", time_limit=5)

    model_dir = Path(svc.settings.models_dir) / "uid123__1DAY"

    assert meta["instrument_uid"] == "uid123"
    assert meta["interval"] == "1DAY"
    assert meta["best_model"] == "ModelA"
    assert model_dir.exists()
    assert (model_dir / "meta.json").exists()
    assert (model_dir / "leaderboard.csv").exists()

    saved_meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    assert saved_meta["best_model"] == "ModelA"


def test_predict_next_close_model_missing(svc, raw_df, monkeypatch):
    install_fake_autogluon(monkeypatch)

    with pytest.raises(FileNotFoundError, match="Model not found"):
        svc.predict_next_close(interval="1DAY", df=raw_df, instrument_uid="missing_uid")


def test_predict_next_close_not_enough_candles(svc, raw_df, monkeypatch):
    install_fake_autogluon(monkeypatch)

    model_dir = Path(svc.settings.models_dir) / "uid123__1DAY"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "best_model": "ModelA",
                "past_covariates": ["body"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Not enough candles for predictions"):
        svc.predict_next_close(interval="1DAY", df=raw_df.head(5), instrument_uid="uid123")


def test_predict_next_close_success(svc, raw_df, monkeypatch):
    svc._PREDICTOR_CACHE.clear()
    FakePredictor = install_fake_autogluon(monkeypatch, mean=150.0, low=145.0, high=155.0)

    model_dir = Path(svc.settings.models_dir) / "uid123__1DAY"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "best_model": "ModelA",
                "past_covariates": ["body", "rsi", "macd"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        svc, "add_technical_features", lambda df: df.assign(body=1.0, rsi=50.0, macd=0.1)
    )

    result = svc.predict_next_close(interval="1DAY", df=raw_df, instrument_uid="uid123")

    assert result["instrument_uid"] == "uid123"
    assert result["interval"] == "1DAY"
    assert result["predicted_close"] == 150.0
    assert result["model_used"] == "ModelA"
    assert result["confidence_interval"] == {
        "lower_90": 145.0,
        "upper_90": 155.0,
    }
    assert FakePredictor.load_calls == 1


def test_predict_next_close_uses_cache(svc, raw_df, monkeypatch):
    svc._PREDICTOR_CACHE.clear()
    FakePredictor = install_fake_autogluon(monkeypatch, mean=150.0, low=145.0, high=155.0)

    model_dir = Path(svc.settings.models_dir) / "uid123__1DAY"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "best_model": "ModelA",
                "past_covariates": ["body", "rsi", "macd"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        svc, "add_technical_features", lambda df: df.assign(body=1.0, rsi=50.0, macd=0.1)
    )

    svc.predict_next_close(interval="1DAY", df=raw_df, instrument_uid="uid123")
    svc.predict_next_close(interval="1DAY", df=raw_df, instrument_uid="uid123")

    assert FakePredictor.load_calls == 1


def test_predict_next_close_skips_weekend_for_1d(svc, monkeypatch):
    svc._PREDICTOR_CACHE.clear()
    install_fake_autogluon(monkeypatch, mean=110.0, low=108.0, high=112.0)

    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-11-07", periods=60, freq="D"),
            "open": [100.0] * 60,
            "high": [102.0] * 60,
            "low": [99.0] * 60,
            "close": [101.0] * 60,
            "volume": [1000.0] * 60,
        }
    )
    df.loc[df.index[-1], "time"] = pd.Timestamp("2024-01-05")  # Friday

    model_dir = Path(svc.settings.models_dir) / "uid123__1DAY"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "best_model": "ModelA",
                "past_covariates": ["body", "rsi", "macd"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        svc, "add_technical_features", lambda x: x.assign(body=1.0, rsi=50.0, macd=0.1)
    )

    result = svc.predict_next_close(interval="1DAY", df=df, instrument_uid="uid123")

    assert result["prediction_date"] == "2024-01-08"
