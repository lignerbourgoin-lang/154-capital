import yfinance as yf
import pandas as pd
import numpy as np

ticker = 'WLN.PA'
data = yf.download(ticker, start='2010-01-01', end='2026-01-01', progress=False, auto_adjust=True)
close = data['Close'].squeeze()

print(f'Shape: {data.shape}')
print(f'Date range: {close.index[0].date()} -> {close.index[-1].date()}')
print()

returns = close.pct_change().dropna()
monthly = (1 + returns).resample('ME').prod() - 1
annual  = (1 + returns).resample('YE').prod() - 1

print('=== Annual returns ===')
print(annual.round(4).to_string())

print('\n=== Extreme monthly returns (|r| > 20%) ===')
extreme = monthly[monthly.abs() > 0.20]
print(extreme.round(4).to_string() if not extreme.empty else 'None')

print('\n=== Close price history (year-end) ===')
print(close.resample('YE').last().round(4).to_string())

print('\n=== NaNs or zeros? ===')
print(f'NaN count: {close.isna().sum()}')
print(f'Zero count: {(close == 0).sum()}')
print(f'Min price: {close.min():.4f}  on {close.idxmin().date()}')
print(f'Max price: {close.max():.4f}  on {close.idxmax().date()}')
