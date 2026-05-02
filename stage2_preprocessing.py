"""
=============================================================
STAGE 2: DATA PREPROCESSING & FEATURE ENGINEERING
=============================================================
PURPOSE:
    Load the combined stock dataset from Stage 1 and:
      1. Handle missing values (forward fill)
      2. Remove/smooth outliers using IQR clipping
      3. Engineer technical features:
           - Log Returns
           - Moving Averages (MA10, MA20)
           - Rolling Volatility (20-day std of log returns)
           - Momentum (10-day)
      4. Save the enriched dataset to
         data/processed/featured_stock_data.csv

WHY FEATURE ENGINEERING?
    Raw OHLCV price data alone is insufficient for a deep
    learning model to detect risk regimes. Technical features
    extract meaningful signals about trend, momentum, and
    market volatility - giving the model richer information
    to learn from.

FEATURE DEFINITIONS:
    Log Return     = ln(Close_t / Close_{t-1})
                     Measures daily percentage price change
                     in a way that is comparable across stocks.

    MA10 / MA20    = 10-day / 20-day simple moving average
                     of Close price. Captures short and
                     medium-term price trends.

    Volatility_20  = Rolling 20-day standard deviation of
                     Log Returns. Directly measures the
                     uncertainty (risk) in price movements.

    Momentum_10    = Close_t - Close_{t-10}
                     Measures price direction and speed
                     over the past 10 trading days.
=============================================================
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
COMBINED_DATA_PATH  = "data/processed/combined_stock_data.csv"
FEATURED_DATA_PATH  = "data/processed/featured_stock_data.csv"

os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the combined dataset and parse the Date column."""
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers.")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: Handle Missing Values
# ─────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing values within each ticker group.
    Any remaining NaNs at the start of a series are back-filled.

    WHY FORWARD FILL?
        Stock prices on non-trading days (holidays, weekends)
        are best approximated by the most recent known price.
        Forward fill preserves the time-series continuity
        without introducing data that did not exist.
    """
    print("  Handling missing values (forward fill per ticker)...")
    before = df.isnull().sum().sum()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = (
        df.groupby("Ticker")[numeric_cols]
        .transform(lambda s: s.ffill().bfill())
    )

    after = df.isnull().sum().sum()
    print(f"    NaN count before: {before:,}  |  after: {after:,}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: Remove / Smooth Outliers
# ─────────────────────────────────────────────────────────────
def clip_outliers(df: pd.DataFrame,
                  cols: list = None,
                  factor: float = 3.0) -> pd.DataFrame:
    """
    Clip extreme outliers in price/volume columns using the
    IQR (Interquartile Range) method per ticker.

    Values beyond  Q1 - factor*IQR  or  Q3 + factor*IQR
    are clipped to those boundary values.

    WHY CLIP OUTLIERS?
        Extreme outliers (e.g. from data errors or flash
        crashes) can distort the normalisation step and
        bias the model's loss function. Clipping keeps the
        data realistic while preserving true volatility.

    Parameters
    ----------
    df     : pd.DataFrame  – input dataset
    cols   : list          – columns to clip (default: OHLCV)
    factor : float         – IQR multiplier (default 3.0)
    """
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]

    print(f"  Clipping outliers (IQR x {factor}) for: {cols}")

    # Process each ticker separately in a loop to avoid pandas
    # groupby.apply dropping the Ticker column in newer versions
    processed = []
    for ticker in df["Ticker"].unique():
        group = df[df["Ticker"] == ticker].copy()
        for col in cols:
            q1    = group[col].quantile(0.25)
            q3    = group[col].quantile(0.75)
            iqr   = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            group[col] = group[col].clip(lower=lower, upper=upper)
        processed.append(group)

    df = pd.concat(processed, ignore_index=True)
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("    Outlier clipping complete.")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4: Feature Engineering
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator features per ticker:

        Log_Return    – daily log return of Close price
        MA10          – 10-day simple moving average of Close
        MA20          – 20-day simple moving average of Close
        Volatility_20 – 20-day rolling std of Log_Return
        Momentum_10   – Close_t minus Close_{t-10}

    All computations are grouped by Ticker so that data from
    one stock never leaks into another.
    """
    print("  Engineering technical features...")

    processed = []
    for ticker in df["Ticker"].unique():
        g = df[df["Ticker"] == ticker].copy()
        g["Log_Return"]    = np.log(g["Close"] / g["Close"].shift(1))
        g["MA10"]          = g["Close"].rolling(window=10).mean()
        g["MA20"]          = g["Close"].rolling(window=20).mean()
        g["Volatility_20"] = g["Log_Return"].rolling(window=20).std()
        g["Momentum_10"]   = g["Close"] - g["Close"].shift(10)
        processed.append(g)

    df = pd.concat(processed, ignore_index=True)
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows where features are NaN due to rolling windows
    before = len(df)
    df.dropna(subset=["Log_Return", "MA10", "MA20",
                      "Volatility_20", "Momentum_10"], inplace=True)
    after = len(df)
    print(f"    Rows dropped due to rolling-window warm-up: {before - after:,}")
    print(f"    Remaining rows: {after:,}")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: Save Processed Dataset
# ─────────────────────────────────────────────────────────────
def save_featured_data(df: pd.DataFrame, path: str) -> None:
    """Save the feature-enriched DataFrame to CSV."""
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    print(f"  Saved featured dataset to: {path}")


# ─────────────────────────────────────────────────────────────
# MAIN PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
def preprocess() -> pd.DataFrame:
    """
    Run the full preprocessing and feature-engineering pipeline.

    Returns
    -------
    pd.DataFrame – processed, feature-enriched dataset
    """
    print("=" * 60)
    print("  STAGE 2 : PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    df = load_data(COMBINED_DATA_PATH)
    df = handle_missing_values(df)
    df = clip_outliers(df)
    df = engineer_features(df)
    save_featured_data(df, FEATURED_DATA_PATH)

    print("\n" + "=" * 60)
    print("  STAGE 2 COMPLETE")
    print(f"  Final shape      : {df.shape}")
    print(f"  Columns          : {list(df.columns)}")
    print(f"  Tickers retained : {df['Ticker'].nunique()}")
    print(f"  Output file      : {FEATURED_DATA_PATH}")
    print("=" * 60)

    return df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    featured_df = preprocess()
    print("\nSample output:")
    print(featured_df[["Date", "Ticker", "Close", "Log_Return",
                        "MA10", "MA20", "Volatility_20",
                        "Momentum_10"]].head(15).to_string(index=False))