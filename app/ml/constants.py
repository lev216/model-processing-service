"""Covariate names and interval frequencies used by model training."""

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
