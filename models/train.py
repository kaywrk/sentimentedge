# models/train.py
# Step 4 — Build & train the LSTM model

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, MODEL_DIR, SCALER_PATH, MODEL_PATH, RESULTS_PATH,
    ALL_FEATURES, LOOKBACK_WINDOW, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, DROPOUT_RATE, LSTM_UNITS,
    TRAIN_RATIO, VAL_RATIO, TICKERS
)


# ─────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, window: int):
    """
    Convert a 2D feature array into 3D sequences for LSTM.
    Input shape:  (n_samples, n_features)
    Output shape: (n_samples - window, window, n_features)

    Each sample at index i uses features from [i, i+window)
    and predicts the label at index i+window.
    """
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ─────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT (chronological)
# ─────────────────────────────────────────

def time_split(df: pd.DataFrame):
    """
    Split into train / val / test keeping time order.
    NEVER shuffle — this causes data leakage in time-series.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"  Split sizes → train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame, scaler=None, fit_scaler=True):
    """
    Scale features and create LSTM sequences.
    Returns: X_seq, y_seq, scaler
    """
    X = df[ALL_FEATURES].values
    y = df["target"].values

    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_seq, y_seq = make_sequences(X_scaled, y, LOOKBACK_WINDOW)
    return X_seq, y_seq, scaler


# ─────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────

def build_lstm_model(input_shape: tuple, units=LSTM_UNITS, dropout=DROPOUT_RATE) -> tf.keras.Model:
    """
    Stacked LSTM with Batch Normalisation and Dropout.

    Architecture:
      Input → LSTM(64, return_sequences=True) → BN → Dropout
            → LSTM(32) → BN → Dropout
            → Dense(16, relu) → Dense(1, sigmoid)

    Binary output: 1 = UP, 0 = DOWN
    """
    model = Sequential([
        Input(shape=input_shape),

        # ── Layer 1: LSTM with sequence output
        LSTM(units[0], return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        BatchNormalization(),
        Dropout(dropout),

        # ── Layer 2: LSTM — final hidden state only
        LSTM(units[1], return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        BatchNormalization(),
        Dropout(dropout),

        # ── Dense head
        Dense(16, activation="relu"),
        Dropout(dropout / 2),

        # ── Output: sigmoid → probability of UP
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def train_model(ticker: str):
    """
    Full training pipeline for one ticker.
    Returns: model, scaler, test results dict
    """
    print(f"\n{'='*55}")
    print(f"  TRAINING: {ticker}")
    print(f"{'='*55}")

    # Load feature matrix
    feat_path = f"{DATA_DIR}features/{ticker}_features.csv"
    df = pd.read_csv(feat_path, index_col="date", parse_dates=True)

    # Split
    train_df, val_df, test_df = time_split(df)

    # Scale & sequence
    X_train, y_train, scaler = prepare_data(train_df, fit_scaler=True)
    X_val,   y_val,   _      = prepare_data(val_df,   scaler=scaler, fit_scaler=False)
    X_test,  y_test,  _      = prepare_data(test_df,  scaler=scaler, fit_scaler=False)

    print(f"  Sequence shapes → train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}")

    # Handle class imbalance
    n_up   = y_train.sum()
    n_down = len(y_train) - n_up
    class_weight = {0: len(y_train) / (2 * n_down), 1: len(y_train) / (2 * n_up)}
    print(f"  Class weights: UP={class_weight[1]:.2f} | DOWN={class_weight[0]:.2f}")

    # Build model
    model = build_lstm_model(input_shape=(LOOKBACK_WINDOW, len(ALL_FEATURES)))
    model.summary()

    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = f"{MODEL_DIR}{ticker}_best.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=ckpt_path, monitor="val_auc",
            save_best_only=True, mode="max", verbose=0
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    results = evaluate_model(model, X_test, y_test, ticker)

    # Save
    scaler_path = f"{MODEL_DIR}{ticker}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    model.save(f"{MODEL_DIR}{ticker}_lstm.keras")
    print(f"\n  Model saved → {MODEL_DIR}{ticker}_lstm.keras")
    print(f"  Scaler saved → {scaler_path}")

    # Save training history plot
    plot_training_history(history, ticker)

    return model, scaler, results


# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test, ticker: str) -> dict:
    """
    Compute full classification metrics on the held-out test set.
    Threshold: 0.5 (tune this for precision/recall tradeoff)
    """
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)

    results = {
        "ticker": ticker,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "auc_roc":   round(auc,  4),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
    }

    print(f"\n{'─'*40}")
    print(f"  TEST RESULTS — {ticker}")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    return results


# ─────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────

def plot_training_history(history, ticker: str):
    """Save training curves for accuracy and loss."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training History — {ticker}", fontsize=14)

    metrics = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC")]
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(history.history[metric], label="Train", linewidth=2)
        ax.plot(history.history[f"val_{metric}"], label="Val", linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{MODEL_DIR}{ticker}_training.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training plot saved → {plot_path}")


# ─────────────────────────────────────────
# TRAIN ALL TICKERS
# ─────────────────────────────────────────

def train_all_tickers():
    """Train separate LSTM models for each ticker."""
    all_results = {}

    for ticker in TICKERS:
        feat_path = f"{DATA_DIR}features/{ticker}_features.csv"
        if not os.path.exists(feat_path):
            print(f"  Missing features for {ticker} — skipping. Run engineer_features.py first.")
            continue

        model, scaler, results = train_model(ticker)
        all_results[ticker] = results

    # Save combined results
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved → {RESULTS_PATH}")

    # Summary table
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"  {'Ticker':<10} {'Accuracy':<12} {'F1':<10} {'AUC-ROC'}")
    print("  " + "-"*45)
    for t, r in all_results.items():
        print(f"  {t:<10} {r['accuracy']:<12} {r['f1_score']:<10} {r['auc_roc']}")

    return all_results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 4 — LSTM MODEL TRAINING")
    print("=" * 55)

    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    results = train_all_tickers()
    print("\nRun next: python api/serve.py")