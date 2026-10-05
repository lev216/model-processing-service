"""Technical feature calculations shared by training and inference."""

import pandas as pd

from app.config import settings


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
