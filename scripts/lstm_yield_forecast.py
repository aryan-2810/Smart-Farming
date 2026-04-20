"""
LSTM-based Yield Forecasting
============================

This module provides utilities to train and use an LSTM model for forecasting
crop yields over time using a multivariate time series (exogenous variables).

Expected input CSV: data/crop_yield_timeseries.csv with columns:
  - date (YYYY-MM-DD or parseable)
  - crop (str)
  - region (str)
  - N, P, K, temperature, humidity, rainfall, pH (float)
  - yield (float)  # target

Functions:
  - prepare_data(df, crop=None, region=None, seq_len=12, test_split=0.2)
  - build_lstm(input_shape)
  - train_and_save(crop, seq_len=12, epochs=50, batch_size=32)
  - predict_future(model_path, scaler_X_path, scaler_y_path, recent_window, n_steps=6)

Artifacts saved to:
  - models/lstm_yield_<crop>.h5
  - models/lstm_scaler_<crop>.joblib  (dict with keys: scaler_X, scaler_y)
  - outputs/forecasts/predictions_<crop>_<timestamp>.json
"""

from __future__ import annotations

import os
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def _ensure_dirs():
    """Ensure required directories exist."""
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('outputs/forecasts').mkdir(parents=True, exist_ok=True)


def _load_dataset(csv_path: str = 'data/crop_yield_timeseries.csv') -> pd.DataFrame:
    """Load time series dataset and parse dates.

    Parameters
    ----------
    csv_path : str
        Path to the timeseries CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a parsed 'date' column.
    """
    df = pd.read_csv(csv_path)
    if 'date' not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_data(
    df: pd.DataFrame,
    crop: Optional[str] = None,
    region: Optional[str] = None,
    seq_len: int = 12,
    test_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Prepare sliding-window sequences for LSTM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with required columns (see module docstring).
    crop : str, optional
        Filter dataset for a given crop (case-insensitive). If None, use all.
    region : str, optional
        Filter dataset for a given region (case-insensitive). If None, use all.
    seq_len : int
        Window length (timesteps) for LSTM input sequences.
    test_split : float
        Fraction of samples used for testing (split chronologically).

    Returns
    -------
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y)
        Arrays ready for LSTM training/evaluation and fitted scalers.
    """
    req_cols = ['date', 'crop', 'region', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'pH', 'yield']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_proc = df.copy()
    if crop:
        df_proc = df_proc[df_proc['crop'].str.lower() == str(crop).strip().lower()].copy()
    if region:
        df_proc = df_proc[df_proc['region'].str.lower() == str(region).strip().lower()].copy()
    df_proc = df_proc.sort_values('date').reset_index(drop=True)

    if len(df_proc) < seq_len + 2:
        raise ValueError("Not enough rows after filtering to build sequences.")

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'pH', 'yield']
    target_col = 'yield'

    features = df_proc[feature_cols].values.astype(np.float32)
    target = df_proc[target_col].values.astype(np.float32).reshape(-1, 1)

    # Scale features and target separately
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(features_scaled)):
        X_seq.append(features_scaled[i - seq_len:i, :])
        y_seq.append(target_scaled[i, 0])  # predict next yield

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

    # Chronological split
    n_total = X_seq.shape[0]
    n_test = max(1, int(math.floor(n_total * test_split)))
    n_train = n_total - n_test
    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_test, y_test = X_seq[n_train:], y_seq[n_train:]

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y


def build_lstm(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build and compile an LSTM model.

    Parameters
    ----------
    input_shape : (timesteps, num_features)
        Shape of the input sequence for the LSTM.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def _evaluate_and_print(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_y: StandardScaler) -> dict:
    """Inverse-scale predictions and print MAE, RMSE, R2."""
    y_true = scaler_y.inverse_transform(y_true_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    print("\nEVALUATION:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_save(
    crop: Optional[str] = None,
    seq_len: int = 12,
    epochs: int = 50,
    batch_size: int = 32,
    csv_path: str = 'data/crop_yield_timeseries.csv',
) -> dict:
    """Train an LSTM model for a specified crop (or globally) and save artifacts.

    Parameters
    ----------
    crop : str, optional
        Crop name to filter by (case-insensitive). If None, trains on all crops.
    seq_len : int
        Sequence length used to create input windows.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    csv_path : str
        Path to the time series dataset.

    Returns
    -------
    dict
        Summary including paths to saved model, scalers, metrics, and forecast file.
    """
    _ensure_dirs()
    df = _load_dataset(csv_path)

    # Prepare data
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(
        df, crop=crop, region=None, seq_len=seq_len, test_split=0.2
    )

    # Build model
    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    model_name_key = (crop or 'global').strip().lower().replace(' ', '_')

    # Callbacks
    ckpt_path = Path('models') / f"lstm_ckpt_{model_name_key}.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=str(ckpt_path), monitor='val_loss', save_best_only=True),
    ]

    # Train
    print("\nTRAINING START")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTRAINING HISTORY (last 5 epochs):")
    for k, v in history.history.items():
        print(f"  {k}: {v[-5:]}")

    # Evaluate
    y_pred_test = model.predict(X_test, verbose=0)
    metrics = _evaluate_and_print(y_test, y_pred_test, scaler_y)

    # Save model and scalers
    model_path = Path('models') / f"lstm_yield_{model_name_key}.h5"
    scaler_path = Path('models') / f"lstm_scaler_{model_name_key}.joblib"
    model.save(str(model_path))
    dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, scaler_path)

    # Save sample forecast (next n_steps from last window of test set)
    last_window = X_test[-1]  # scaled window
    # Inverse-transform window to original feature space for predict_future API
    last_window_orig = scaler_X.inverse_transform(last_window)
    preds = predict_future(
        str(model_path), str(scaler_path), str(scaler_path),
        recent_window=last_window_orig, n_steps=6
    )
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
    forecast_out = Path('outputs/forecasts') / f"predictions_{model_name_key}_{ts}.json"
    with forecast_out.open('w', encoding='utf-8') as f:
        json.dump({"crop": crop or "global", "seq_len": seq_len, "n_steps": 6, "predictions": preds}, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scalers to: {scaler_path}")
    print(f"Saved sample predictions to: {forecast_out}")

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metrics": metrics,
        "forecast_path": str(forecast_out),
    }


def predict_future(
    model_path: str,
    scaler_X_path: str,
    scaler_y_path: str,
    recent_window: np.ndarray,
    n_steps: int = 6,
) -> list:
    """Predict future yields for the next n_steps using a trained model.

    Parameters
    ----------
    model_path : str
        Path to saved Keras model (.h5).
    scaler_X_path : str
        Path to joblib file with scaler_X (and possibly scaler_y if combined).
    scaler_y_path : str
        Path to joblib file with scaler_y (ignored if scaler_X_path already includes it).
    recent_window : np.ndarray
        Recent window in ORIGINAL scale, shape (seq_len, num_features). It must
        include the same feature columns used in training: [N,P,K,temperature,humidity,rainfall,pH,yield].
    n_steps : int
        Number of future steps to predict.

    Returns
    -------
    list
        List of predicted yields in original scale.
    """
    # Load model and scalers
    model = tf.keras.models.load_model(model_path)
    scalers = load(scaler_X_path)
    if isinstance(scalers, dict) and 'scaler_X' in scalers and 'scaler_y' in scalers:
        scaler_X = scalers['scaler_X']
        scaler_y = scalers['scaler_y']
    else:
        scaler_X = scalers
        scaler_y = load(scaler_y_path)

    window_orig = np.array(recent_window, dtype=np.float32)
    seq_len, n_feat = window_orig.shape

    preds = []
    # Iterative forecasting: keep exogenous vars constant at last observed values
    exog_last = window_orig[-1].copy()
    for _ in range(n_steps):
        # Build scaled input with current window
        window_scaled = scaler_X.transform(window_orig)
        X_in = window_scaled.reshape(1, seq_len, n_feat)
        y_pred_scaled = model.predict(X_in, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        preds.append(float(max(0.0, y_pred)))

        # Update window: shift left, append new row with exog_last and predicted yield
        next_row = exog_last.copy()
        # yield is last column in our training feature ordering
        next_row[-1] = y_pred
        window_orig = np.vstack([window_orig[1:], next_row])

    return preds


if __name__ == '__main__':
    """Fast default run: Train on crop 'Rice' with seq_len=12, epochs=30."""
    try:
        summary = train_and_save(crop='Rice', seq_len=12, epochs=30, batch_size=32)
        print("\nTraining complete.")
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error: {e}")


