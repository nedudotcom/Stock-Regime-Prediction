"""
=============================================================
STAGE 3: RISK REGIME LABELLING, NORMALISATION &
          SEQUENCE CREATION
=============================================================
PURPOSE:
    1. Label each trading day as Low / Medium / High Risk
       based on rolling 20-day volatility percentiles.
    2. Normalise all feature columns using MinMaxScaler
       (fit on training data only to prevent data leakage).
    3. Build sliding-window sequences of length WINDOW_SIZE=60
       so that the CNN-LSTM model can learn temporal patterns.
    4. Split data into Train (70%), Validation (15%),
       Test (15%) sets - chronologically, not randomly.
    5. Save all arrays and the fitted scaler to disk so
       Stages 4 and 5 can load them directly.

RISK REGIME DEFINITION:
    Volatility (20-day rolling std of log returns) is computed
    per ticker, then percentile thresholds are applied:

        Volatility < 33rd percentile  ->  0  (Low Risk)
        Volatility < 66th percentile  ->  1  (Medium Risk)
        Volatility >= 66th percentile ->  2  (High Risk)

    These thresholds divide the volatility distribution into
    three equal-sized classes, creating a balanced supervised
    classification problem.

SLIDING WINDOW (WINDOW_SIZE = 60):
    For each day t, we take the 60 consecutive rows
    (t-59 ... t) as input X and the Risk_Label on day t
    as target y. This gives the LSTM component enough
    historical context to detect regime transitions.

TRAIN / VAL / TEST SPLIT:
    The split is chronological (no shuffling) to prevent
    future data from leaking into training:

        Train : first 70% of sequences
        Val   : next  15%
        Test  : last  15%
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
FEATURED_DATA_PATH = "data/processed/featured_stock_data.csv"
SEQUENCES_DIR      = "data/processed/sequences"

WINDOW_SIZE = 60   # Number of trading days per sequence

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Log_Return", "MA10", "MA20", "Volatility_20", "Momentum_10"
]
LABEL_COL = "Risk_Label"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  = 0.15

os.makedirs(SEQUENCES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: Assign Risk Regime Labels
# ─────────────────────────────────────────────────────────────
def label_risk_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a Risk_Label (0, 1, 2) to each row based on the
    Volatility_20 column using per-ticker percentile thresholds.

    Labels:
        0 = Low Risk    (volatility < 33rd percentile)
        1 = Medium Risk (33rd <= volatility < 66th percentile)
        2 = High Risk   (volatility >= 66th percentile)
    """
    print("  Labelling risk regimes via volatility percentiles...")

    processed = []
    for ticker in df["Ticker"].unique():
        group = df[df["Ticker"] == ticker].copy()
        vol   = group["Volatility_20"]
        p33   = vol.quantile(0.33)
        p66   = vol.quantile(0.66)
        conds   = [vol < p33, vol < p66]
        choices = [0, 1]
        group[LABEL_COL] = np.select(conds, choices, default=2)
        processed.append(group)

    df = pd.concat(processed, ignore_index=True)
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Print class distribution
    counts = df[LABEL_COL].value_counts().sort_index()
    names  = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    print("    Class distribution:")
    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"      {names[label]:<12} ({label}): {count:>7,}  ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: Normalise Features
# ─────────────────────────────────────────────────────────────
def normalise_features(df: pd.DataFrame):
    """
    Scale all feature columns to [0, 1] using MinMaxScaler.

    IMPORTANT - To prevent data leakage:
        The scaler is fit ONLY on the training portion of
        each ticker's data, then applied (transform) to val
        and test portions. The fitted scaler is saved so the
        Streamlit dashboard can use it for live inference.

    Returns
    -------
    df_scaled : pd.DataFrame – dataset with scaled features
    scaler    : MinMaxScaler – fitted scaler object
    """
    print("  Normalising features with MinMaxScaler...")

    # Determine the training cutoff row index globally
    n_total    = len(df)
    train_end  = int(n_total * TRAIN_RATIO)

    scaler = MinMaxScaler()
    df = df.copy()

    # Fit only on training rows
    scaler.fit(df.iloc[:train_end][FEATURE_COLS])

    # Transform all rows
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    # Save scaler for use in Stage 5 (dashboard) inference
    scaler_path = os.path.join(SEQUENCES_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Scaler saved to: {scaler_path}")

    return df, scaler


# ─────────────────────────────────────────────────────────────
# STEP 3: Build Sliding-Window Sequences
# ─────────────────────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    """
    Create overlapping sliding-window sequences of length
    WINDOW_SIZE across all tickers combined.

    For each ticker, sequences are built independently to
    avoid cross-ticker boundary contamination, then
    concatenated.

    Returns
    -------
    X : np.ndarray  shape (N, WINDOW_SIZE, n_features)
    y : np.ndarray  shape (N,)
    """
    print(f"  Building sliding-window sequences (window={WINDOW_SIZE})...")

    X_list, y_list = [], []

    for ticker in df["Ticker"].unique():
        tdf     = df[df["Ticker"] == ticker].reset_index(drop=True)
        features = tdf[FEATURE_COLS].values
        labels   = tdf[LABEL_COL].values

        for i in range(WINDOW_SIZE, len(tdf)):
            X_list.append(features[i - WINDOW_SIZE : i])
            y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"    Total sequences : {len(X):,}")
    print(f"    X shape         : {X.shape}")
    print(f"    y shape         : {y.shape}")
    print(f"    Class balance   : {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


# ─────────────────────────────────────────────────────────────
# STEP 4: Train / Val / Test Split (Chronological)
# ─────────────────────────────────────────────────────────────
def split_data(X: np.ndarray, y: np.ndarray):
    """
    Split sequences chronologically into Train / Val / Test.

    Parameters
    ----------
    X : np.ndarray  shape (N, WINDOW_SIZE, n_features)
    y : np.ndarray  shape (N,)

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    """
    n          = len(X)
    train_end  = int(n * TRAIN_RATIO)
    val_end    = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"\n  Data split summary:")
    print(f"    Train : {len(X_train):>7,} sequences  ({100*TRAIN_RATIO:.0f}%)")
    print(f"    Val   : {len(X_val):>7,} sequences  ({100*VAL_RATIO:.0f}%)")
    print(f"    Test  : {len(X_test):>7,} sequences  ({100*(1-TRAIN_RATIO-VAL_RATIO):.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────
# STEP 5: Save Arrays to Disk
# ─────────────────────────────────────────────────────────────
def save_arrays(X_train, X_val, X_test,
                y_train, y_val, y_test) -> None:
    """Save all numpy arrays to the sequences directory."""
    np.save(os.path.join(SEQUENCES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SEQUENCES_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(SEQUENCES_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SEQUENCES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SEQUENCES_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(SEQUENCES_DIR, "y_test.npy"),  y_test)
    print(f"\n  All arrays saved to: {SEQUENCES_DIR}/")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def prepare_sequences():
    """
    Full Stage 3 pipeline:
        Load -> Label -> Normalise -> Sequence -> Split -> Save
    """
    print("=" * 60)
    print("  STAGE 3 : LABELLING, NORMALISATION & SEQUENCES")
    print("=" * 60)

    # Load featured data from Stage 2
    print(f"  Loading: {FEATURED_DATA_PATH}")
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} rows, {df['Ticker'].nunique()} tickers.\n")

    df              = label_risk_regimes(df)
    df, scaler      = normalise_features(df)
    X, y            = build_sequences(df)
    splits          = split_data(X, y)
    save_arrays(*splits)

    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print("\n" + "=" * 60)
    print("  STAGE 3 COMPLETE")
    print(f"  Sequence shape (X) : {X_train.shape[1:]}")
    print(f"  Feature count      : {X_train.shape[2]}")
    print(f"  Window size        : {WINDOW_SIZE} trading days")
    print(f"  Saved arrays       : {SEQUENCES_DIR}/")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_sequences()