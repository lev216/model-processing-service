"""Resolve the existing on-disk model layout."""

from pathlib import Path

from app.config import settings


def _model_key(instrument_uid: str, interval: str) -> str:
    return f"{instrument_uid}__{interval}"


def _model_path(instrument_uid: str, interval: str) -> Path:
    return Path(settings.models_dir) / _model_key(instrument_uid, interval)
