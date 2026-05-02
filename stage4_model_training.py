"""
=============================================================
STAGE 4: CNN-LSTM MODEL BUILDING & TRAINING
=============================================================
PURPOSE:
    Build and train the hybrid CNN-LSTM deep learning model
    that classifies each 60-day window into one of three
    risk regimes: Low (0), Medium (1), High (2).

MODEL ARCHITECTURE:
    Input  (60, 10)  - 60 timesteps, 10 features
      |
    Conv1D (64 filters, kernel=3, ReLU)
      |
    MaxPooling1D (pool_size=2)
      |
    Conv1D (128 filters, kernel=3, ReLU)
      |
    Dropout (0.3)
      |
    LSTM (128 units, return_sequences=True)
      |
    LSTM (64 units)
      |
    Dropout (0.3)
      |
    Dense (64, ReLU)
      |
    Dense (3, Softmax)   <-- 3 risk regime classes

WHY CNN + LSTM?
    - CNN layers extract local short-term patterns (e.g. a
      sudden volatility spike over a few days).
    - LSTM layers learn long-term temporal dependencies
      (e.g. gradual regime transitions over weeks).
    - Together they capture both local and global patterns
      in financial time series, outperforming either
      architecture alone.

TRAINING DETAILS:
    - Optimizer       : Adam (adaptive learning rate)
    - Loss            : Sparse Categorical Crossentropy
    - Metrics         : Accuracy
    - Epochs          : 30 (with early stopping, patience=5)
    - Batch size      : 64
    - Early stopping  : Monitors val_loss; restores best weights
=============================================================
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PyCharm / servers
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import plot_model

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
SEQUENCES_DIR = "data/processed/sequences"
MODELS_DIR    = "models"
RESULTS_DIR   = "results"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE  = 60
N_FEATURES   = 10      # matches FEATURE_COLS in Stage 3
N_CLASSES    = 3       # Low / Medium / High Risk

EPOCHS       = 30
BATCH_SIZE   = 64
PATIENCE     = 5       # Early stopping patience


# ─────────────────────────────────────────────────────────────
# STEP 1: Load Training & Validation Arrays
# ─────────────────────────────────────────────────────────────
def load_train_val():
    """Load X_train, y_train, X_val, y_val from Stage 3 output."""
    print("  Loading training and validation arrays...")
    X_train = np.load(os.path.join(SEQUENCES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(SEQUENCES_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(SEQUENCES_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(SEQUENCES_DIR, "y_val.npy"))

    print(f"    X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"    X_val   : {X_val.shape}  |  y_val   : {y_val.shape}")
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────
# STEP 2: Build the Hybrid CNN-LSTM Model
# ─────────────────────────────────────────────────────────────
def build_model(window_size: int = WINDOW_SIZE,
                n_features:  int = N_FEATURES,
                n_classes:   int = N_CLASSES) -> tf.keras.Model:
    """
    Construct the hybrid CNN-LSTM architecture.

    Parameters
    ----------
    window_size : int  – sequence length (timesteps)
    n_features  : int  – number of input features per timestep
    n_classes   : int  – number of output classes

    Returns
    -------
    tf.keras.Model – compiled model
    """
    model = Sequential(name="Hybrid_CNN_LSTM", layers=[

        # ── Input ─────────────────────────────────────────
        Input(shape=(window_size, n_features),
              name="Input"),

        # ── CNN Block 1 ───────────────────────────────────
        # 64 filters, kernel size 3 => scans 3 consecutive days
        Conv1D(filters=64, kernel_size=3, activation="relu",
               padding="same", name="Conv1D_1"),

        MaxPooling1D(pool_size=2, name="MaxPool_1"),

        # ── CNN Block 2 ───────────────────────────────────
        # Deeper feature extraction with 128 filters
        Conv1D(filters=128, kernel_size=3, activation="relu",
               padding="same", name="Conv1D_2"),

        Dropout(0.3, name="Dropout_1"),

        # ── LSTM Block 1 ──────────────────────────────────
        # return_sequences=True passes full sequence to next LSTM
        LSTM(units=128, return_sequences=True, name="LSTM_1"),

        # ── LSTM Block 2 ──────────────────────────────────
        # return_sequences=False outputs only the last timestep
        LSTM(units=64, return_sequences=False, name="LSTM_2"),

        Dropout(0.3, name="Dropout_2"),

        # ── Fully Connected Layers ────────────────────────
        Dense(units=64, activation="relu", name="Dense_1"),

        # ── Output Layer ──────────────────────────────────
        # Softmax produces probability distribution over 3 classes
        Dense(units=n_classes, activation="softmax", name="Output"),
    ])

    # Compile with Adam and sparse crossentropy
    # (sparse = labels are integers, not one-hot vectors)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ─────────────────────────────────────────────────────────────
# STEP 3: Print Model Summary
# ─────────────────────────────────────────────────────────────
def print_model_summary(model: tf.keras.Model) -> None:
    """Print and save the model summary to a text file."""
    summary_path = os.path.join(MODELS_DIR, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    model.summary()
    print(f"    Summary saved to: {summary_path}")


# ─────────────────────────────────────────────────────────────
# STEP 4: Define Training Callbacks
# ─────────────────────────────────────────────────────────────
def get_callbacks(model_path: str) -> list:
    """
    Return a list of Keras callbacks:

    EarlyStopping      – stop training if val_loss stops improving
                         for `patience` epochs. Restores best weights.
    ModelCheckpoint    – save the best model weights automatically.
    ReduceLROnPlateau  – halve learning rate if val_loss stalls
                         for 3 epochs. Helps escape local minima.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────
# STEP 5: Plot Training History
# ─────────────────────────────────────────────────────────────
def plot_training_history(history) -> None:
    """
    Plot and save two charts:
        1. Training vs Validation Loss
        2. Training vs Validation Accuracy

    These curves help diagnose overfitting:
        - If val_loss >> train_loss  => overfitting
        - If both converge well      => good generalisation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──────────────────────────────────────────────
    ax1.plot(history.history["loss"],     label="Train Loss",  color="steelblue")
    ax1.plot(history.history["val_loss"], label="Val Loss",    color="tomato", linestyle="--")
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Sparse Categorical Crossentropy)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────
    ax2.plot(history.history["accuracy"],     label="Train Accuracy", color="steelblue")
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato", linestyle="--")
    ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    Training history plot saved to: {plot_path}")


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────
def train_model():
    """
    Full Stage 4 pipeline:
        Load data -> Build model -> Train -> Save -> Plot history
    """
    print("=" * 60)
    print("  STAGE 4 : MODEL BUILDING & TRAINING")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    X_train, y_train, X_val, y_val = load_train_val()

    # ── Build model ───────────────────────────────────────
    print("\n  Building CNN-LSTM model...")
    model = build_model()
    print_model_summary(model)

    # ── Train ─────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "cnn_lstm_best.keras")
    callbacks  = get_callbacks(model_path)

    print(f"\n  Training for up to {EPOCHS} epochs "
          f"(early stopping patience={PATIENCE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Plot ──────────────────────────────────────────────
    print("\n  Plotting training history...")
    plot_training_history(history)

    # ── Final validation metrics ──────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Best Validation Loss     : {val_loss:.4f}")
    print(f"  Best Validation Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")

    print("\n" + "=" * 60)
    print("  STAGE 4 COMPLETE")
    print(f"  Best model saved to : {model_path}")
    print(f"  Training plot saved : results/training_history.png")
    print("=" * 60)

    return model, history


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train_model()
