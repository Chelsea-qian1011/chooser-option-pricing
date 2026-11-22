"""Shared pricing and model utilities for the Chooser Option project.

The GBDT model was trained on the 10 features listed in FEATURE_ORDER and
predicts a *residual* (a dollar correction) on top of the BSM call price.
This module centralises the formulas so the scripts and the Streamlit app
cannot drift apart.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Week6_Best_GBDT_Model.pkl"
FEATURES_PATH = BASE_DIR / "features.csv"

FEATURE_ORDER = [
    "Return",
    "Vol_5",
    "Vol_20",
    "Vol_60",
    "Volume_MA5",
    "VIX_Change",
    "VIX_JPM_Corr",
    "Rate_Momentum",
    "Return_Lag1",
    "MA_Gap",
]

FEATURE_LABELS = {
    "Return": "Daily return",
    "Vol_5": "5-day realised vol",
    "Vol_20": "20-day realised vol",
    "Vol_60": "60-day realised vol",
    "Volume_MA5": "5-day avg volume",
    "VIX_Change": "VIX daily change",
    "VIX_JPM_Corr": "VIX-JPM correlation",
    "Rate_Momentum": "10Y rate momentum",
    "Return_Lag1": "Previous day return",
    "MA_Gap": "Price vs 20D MA gap",
}

# Calm-market reference point (roughly the sample median of features.csv).
NORMAL_FEATURES = {
    "Return": 0.0,
    "Vol_5": 0.19,
    "Vol_20": 0.22,
    "Vol_60": 0.22,
    "Volume_MA5": 11_800_000.0,
    "VIX_Change": 0.0,
    "VIX_JPM_Corr": -0.23,
    "Rate_Momentum": 0.0,
    "Return_Lag1": 0.0,
    "MA_Gap": 0.0,
}

# Elevated-stress reference point (roughly the 2022 sell-off regime).
STRESS_FEATURES = {
    "Return": -0.03,
    "Vol_5": 0.45,
    "Vol_20": 0.35,
    "Vol_60": 0.28,
    "Volume_MA5": 20_000_000.0,
    "VIX_Change": 0.12,
    "VIX_JPM_Corr": -0.55,
    "Rate_Momentum": 0.03,
    "Return_Lag1": -0.02,
    "MA_Gap": -0.08,
}

# The scenario used in the original Week 8 draft.
BLACK_SWAN_FEATURES = {
    "Return": -0.15,
    "Vol_5": 1.20,
    "Vol_20": 0.80,
    "Vol_60": 0.50,
    "Volume_MA5": 40_000_000.0,
    "VIX_Change": 0.50,
    "VIX_JPM_Corr": -0.80,
    "Rate_Momentum": -0.02,
    "Return_Lag1": -0.05,
    "MA_Gap": -0.30,
}


def bs_call(S, K, T, r, sigma):
    """Black-Scholes-Merton price of a European call."""
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes-Merton price of a European put."""
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def chooser_option_price(S, K, T1, T2, r, sigma):
    """Rubinstein (1991) simple chooser option price.

    At T1 the holder chooses whether the payoff is a call or a put with
    strike K maturing at T2.  The replicating portfolio is a call with
    maturity T2 plus a put with maturity T1 and strike K*exp(-r*(T2-T1)).
    """
    call_leg = bs_call(S, K, T2, r, sigma)
    put_strike = K * np.exp(-r * (T2 - T1))
    put_leg = bs_put(S, put_strike, T1, r, sigma)
    return call_leg + put_leg


def load_model(path=MODEL_PATH):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def features_to_array(features):
    """Convert a dict / Series / DataFrame into the model's 10-column array."""
    if isinstance(features, dict):
        return np.array([[float(features[name]) for name in FEATURE_ORDER]])
    if isinstance(features, pd.Series):
        return np.array([[float(features[name]) for name in FEATURE_ORDER]])
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"Expected {len(FEATURE_ORDER)} features, got {arr.shape[1]}")
    return arr


def predict_correction(model, features):
    """GBDT residual correction in dollars (positive = mark the option up)."""
    return float(model.predict(features_to_array(features))[0])


def prediction_range(model, features_path=FEATURES_PATH):
    """In-sample range of the model output, used as an extrapolation guard."""
    df = pd.read_csv(features_path)
    preds = model.predict(df[FEATURE_ORDER].to_numpy())
    return float(preds.min()), float(preds.max())


def feature_quantiles(features_path=FEATURES_PATH):
    df = pd.read_csv(features_path)
    return df[FEATURE_ORDER].quantile([0.01, 0.05, 0.50, 0.95, 0.99])


def build_scenario_features(
    vol,
    spot=150.0,
    rate=0.05,
    daily_return=None,
    vix_change=None,
    volume_mult=1.0,
    corr=-0.23,
    base=None,
):
    """Build a 10-feature vector for a coherent market scenario.

    Only the features the GBDT actually consumes are set; the rest are
    interpolated from realised-volatility relationships observed in the
    training data (short vol > 20D vol > 60D vol).
    """
    feats = dict(base or NORMAL_FEATURES)
    feats["Vol_20"] = float(vol)
    feats["Vol_5"] = float(max(vol * 1.30, 0.01))
    feats["Vol_60"] = float(max(vol * 0.80, 0.01))
    feats["MA_Gap"] = float(np.clip((spot - 150.0) / 150.0, -0.95, 1.0))
    feats["Rate_Momentum"] = float(rate - 0.05)
    feats["Volume_MA5"] = float(NORMAL_FEATURES["Volume_MA5"] * volume_mult)
    feats["VIX_JPM_Corr"] = float(corr)
    if daily_return is not None:
        feats["Return"] = float(daily_return)
        feats["Return_Lag1"] = float(daily_return * 0.33)
    if vix_change is not None:
        feats["VIX_Change"] = float(vix_change)
    return feats


def ood_flags(features, quantiles=None):
    """Return the list of features outside the 1%-99% training range."""
    if quantiles is None:
        quantiles = feature_quantiles()
    lo = quantiles.loc[0.01]
    hi = quantiles.loc[0.99]
    return [name for name in FEATURE_ORDER if not (lo[name] <= features[name] <= hi[name])]


def guard_correction(correction, bsm_price, max_rel=0.20):
    """Cap the ML correction so it cannot dominate the option price.

    A tree model can only output values seen in training.  When a scenario
    is far outside the training sample the raw correction becomes
    economically implausible; this guard keeps it inside +/- max_rel of BSM.
    """
    cap = max_rel * max(bsm_price, 1e-9)
    return float(np.clip(correction, -cap, cap))


def price_chooser(S, K, T1, T2, r, sigma, model=None, features=None, max_rel=0.20):
    """Convenience wrapper used by the Streamlit app.

    Returns a dict with the plain BSM chooser price, the ML correction, the
    guarded correction and the two final prices.
    """
    bsm_price = chooser_option_price(S, K, T1, T2, r, sigma)
    call_price = bs_call(S, K, T2, r, sigma)
    correction = 0.0
    if model is not None and features is not None:
        correction = predict_correction(model, features)
    guarded = guard_correction(correction, bsm_price, max_rel=max_rel)
    return {
        "bsm_price": float(bsm_price),
        "call_price": float(call_price),
        "ml_correction_raw": float(correction),
        "ml_correction": float(guarded),
        "final_price_raw": float(bsm_price + correction),
        "final_price": float(bsm_price + guarded),
        "error_margin": float(abs(guarded)),
        "error_pct": float(abs(guarded) / bsm_price) if bsm_price > 0 else 0.0,
    }