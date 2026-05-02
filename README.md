# Hybrid CNN-LSTM Model for Risk Regime Detection in Blue-Chip Stocks

A final-year Computer Science project

## Overview
A deep learning system that classifies stock market conditions into **Low**, **Medium**, or **High Risk** regimes using a hybrid CNN-LSTM model trained on 10 years of data from the top 50 S&P 500 companies.

## Tech Stack
Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, yfinance

## Setup
```bash
git clone https://github.com/nedudotcom/hybrid_cnn_lstm.git
cd hybrid_cnn_lstm
pip install -r requirements.txt
```

## Run
```bash
python run_all.py                   # Runs all training stages
streamlit run stage6_dashboard.py   # Launches the dashboard
```

## Dashboard
- **Stock Analysis tab** — Select any of the 50 stocks by sector, view price chart, risk regime, and probability breakdown
- **Auto Alert Monitor tab** — Automatically scans all 50 stocks and flags High Risk stocks in real time

## Project Stages
| Stage | Description |
|---|---|
| Stage 1 | Data collection via Yahoo Finance |
| Stage 2 | Preprocessing & feature engineering |
| Stage 3 | Risk labelling, normalisation & sequence creation |
| Stage 4 | CNN-LSTM model training |
| Stage 5 | Model evaluation |
| Stage 6 | Streamlit dashboard |

> ⚠️ For educational purposes only. Not financial advice.
