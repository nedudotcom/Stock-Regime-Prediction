"""
=============================================================
STAGE 1: DATA COLLECTION
=============================================================
PURPOSE:
    Download 10 years of historical OHLCV (Open, High, Low,
    Close, Volume) data for the top 50 S&P 500 companies by
    market capitalisation, using Yahoo Finance (yfinance).

    Each stock is saved as an individual CSV file inside
    data/raw/, and a single combined CSV is saved to
    data/processed/combined_stock_data.csv for use in all
    subsequent stages.

WHY THIS STEP?
    Deep learning models need large, diverse datasets to learn
    generalised patterns. Using 50 major stocks spanning 10
    years gives us sufficient data volume and variety to train
    a robust risk-regime classifier.
=============================================================
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Top 50 S&P 500 companies by market capitalisation
TICKERS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "NVDA",  # NVIDIA Corporation
    "AMZN",  # Amazon.com Inc.
    "GOOGL", # Alphabet Inc. (Class A)
    "META",  # Meta Platforms Inc.
    "BRK-B", # Berkshire Hathaway (Class B)
    "TSLA",  # Tesla Inc.
    "LLY",   # Eli Lilly and Company
    "AVGO",  # Broadcom Inc.
    "JPM",   # JPMorgan Chase & Co.
    "V",     # Visa Inc.
    "UNH",   # UnitedHealth Group
    "XOM",   # ExxonMobil Corporation
    "MA",    # Mastercard Inc.
    "JNJ",   # Johnson & Johnson
    "PG",    # Procter & Gamble
    "HD",    # Home Depot Inc.
    "COST",  # Costco Wholesale Corporation
    "MRK",   # Merck & Co.
    "ABBV",  # AbbVie Inc.
    "NFLX",  # Netflix Inc.
    "BAC",   # Bank of America Corporation
    "CRM",   # Salesforce Inc.
    "CVX",   # Chevron Corporation
    "ORCL",  # Oracle Corporation
    "AMD",   # Advanced Micro Devices
    "KO",    # The Coca-Cola Company
    "PEP",   # PepsiCo Inc.
    "TMO",   # Thermo Fisher Scientific
    "ACN",   # Accenture PLC
    "MCD",   # McDonald's Corporation
    "LIN",   # Linde PLC
    "CSCO",  # Cisco Systems Inc.
    "ABT",   # Abbott Laboratories
    "ADBE",  # Adobe Inc.
    "WMT",   # Walmart Inc.
    "TXN",   # Texas Instruments Inc.
    "DHR",   # Danaher Corporation
    "NEE",   # NextEra Energy Inc.
    "PM",    # Philip Morris International
    "INTU",  # Intuit Inc.
    "QCOM",  # Qualcomm Inc.
    "IBM",   # International Business Machines
    "CAT",   # Caterpillar Inc.
    "GE",    # GE Aerospace
    "BA",    # Boeing Company
    "RTX",   # RTX Corporation (Raytheon)
    "SPGI",  # S&P Global Inc.
    "AMGN",  # Amgen Inc.
]

# 10-year date range
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
COMBINED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "combined_stock_data.csv")

os.makedirs(RAW_DATA_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPER: Download a single ticker
# ─────────────────────────────────────────────────────────────
def download_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker via yfinance.

    Parameters
    ----------
    ticker : str  – Stock symbol, e.g. 'AAPL'
    start  : str  – Start date 'YYYY-MM-DD'
    end    : str  – End date   'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame with columns:
        Date, Open, High, Low, Close, Volume, Ticker
    Returns an empty DataFrame if download fails.
    """
    print(f"  [{ticker:<5}] Downloading ...", end=" ", flush=True)

    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
    except Exception as e:
        print(f"ERROR - {e}")
        return pd.DataFrame()

    if df.empty:
        print("NO DATA - skipped.")
        return pd.DataFrame()

    # Flatten multi-level columns (yfinance >=0.2 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Standardise Date column name
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    # Keep only required columns
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        print(f"MISSING COLUMNS {missing} - skipped.")
        return pd.DataFrame()

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker

    # Ensure correct dtypes
    df["Date"]   = pd.to_datetime(df["Date"])
    df["Volume"] = df["Volume"].astype(float)

    print(f"OK  ({len(df):,} trading days)")
    return df


# ─────────────────────────────────────────────────────────────
# MAIN ROUTINE
# ─────────────────────────────────────────────────────────────
def collect_all_stocks() -> pd.DataFrame:
    """
    Loop over all 50 tickers:
      1. Download OHLCV data
      2. Save individual CSV  ->  data/raw/<TICKER>.csv
      3. Append to master list

    Then save combined DataFrame to:
      data/processed/combined_stock_data.csv

    Returns
    -------
    pd.DataFrame - combined dataset for all stocks
    """
    print("=" * 60)
    print("  STAGE 1 : DATA COLLECTION")
    print(f"  Period  : {START_DATE}  ->  {END_DATE}")
    print(f"  Stocks  : {len(TICKERS)} (Top 50 S&P 500)")
    print("=" * 60)

    all_frames = []
    failed     = []

    for ticker in TICKERS:
        df = download_stock(ticker, START_DATE, END_DATE)

        if df.empty:
            failed.append(ticker)
            continue

        # Save individual stock CSV
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        df.to_csv(csv_path, index=False)

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(
            "No data was downloaded. Check your internet connection."
        )

    # Combine all stocks into one DataFrame
    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(["Ticker", "Date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Save combined CSV
    combined.to_csv(COMBINED_DATA_PATH, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("  STAGE 1 COMPLETE")
    print(f"  Stocks downloaded  : {len(all_frames)}")
    print(f"  Stocks failed      : {len(failed)}  {failed if failed else ''}")
    print(f"  Total rows         : {len(combined):,}")
    print(f"  Date range         : {combined['Date'].min().date()} -> "
          f"{combined['Date'].max().date()}")
    print(f"  Individual CSVs    : data/raw/<TICKER>.csv")
    print(f"  Combined dataset   : {COMBINED_DATA_PATH}")
    print("=" * 60)

    return combined


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    combined_df = collect_all_stocks()
    print("\nSample of combined dataset:")
    print(combined_df.head(10).to_string(index=False))
    print(f"\nUnique tickers in dataset: {sorted(combined_df['Ticker'].unique())}")
