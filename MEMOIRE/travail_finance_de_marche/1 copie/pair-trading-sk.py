import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import par84
from pathlib import Path

images = par84.img_path + 'pair-trading-'

tickers = ['PEP', 'KO']         
y_ticker, x_ticker = tickers

df = yf.Tickers(tickers).history(period='1d', interval="1m", auto_adjust=True)['Close']

X = sm.add_constant(df[[x_ticker]], has_constant='add')
y = df[y_ticker]
model = sm.OLS(y, X).fit()

resid = model.resid.dropna()
adf_stat, pval, *_ = adfuller(resid)

print(model.summary())

Path(images + "1.html").write_text(
    "<body style='color:white; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;'>" +
    model.summary().as_html(),
    encoding="utf-8"
)

print(f"stat = {adf_stat:.4f}")
print(f"p-value = {pval:.4g}  (should be less than 0.05 --> residuals are stationary)")

predicted = model.predict()
resid_series = y - predicted 

mu = resid.mean()
std = resid.std(ddof=1)
z = (resid_series - mu) / std

upper, lower = 1.5, -1.5
signals = df.copy()
signals['predicted'] = predicted
signals['residual']  = resid_series
signals['z']         = z
signals['z upper']   = upper
signals['z lower']   = lower

z_prev = signals['z'].shift(1)
z_curr = signals['z']

short_entry = (z_prev <= upper) & (z_curr > upper)
long_entry  = (z_prev >= lower) & (z_curr < lower)

signals['signals_y'] = np.nan
signals.loc[long_entry, 'signals_y'] = 1
signals.loc[short_entry, 'signals_y'] = -1
signals['signals_y'] = signals['signals_y'].ffill().fillna(0)

signals['positions_y'] = signals['signals_y'].diff().fillna(0)
signals['signals_x']   = -signals['signals_y']
signals['positions_x'] = signals['signals_x'].diff().fillna(0)

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=signals.index, y=signals['z'], name='Z-score')
fig.add_scatter(x=signals.index, y=[upper]*len(signals), name='+1σ')
fig.add_scatter(x=signals.index, y=[lower]*len(signals), name='-1σ', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.25)')
fig.update_layout(title=f'Z-score band: {y_ticker} vs {x_ticker}')
fig.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")

df100 = df / df.iloc[0] * 100
signals['y100'] = df100[y_ticker]
signals['x100'] = df100[x_ticker]

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=signals.index, y=signals['y100'], name=f'{y_ticker} (base 100)')
fig.add_scatter(x=signals.index, y=signals['x100'], name=f'{x_ticker} (base 100)', line=dict(dash='dot'))

fig.add_scatter(
    x=signals.index[signals['positions_y'] > 0],
    y=signals.loc[signals['positions_y'] > 0, 'y100'],
    mode='markers',
    name=f'Long {y_ticker}',
    marker=dict(symbol='triangle-up', color='limegreen', size=12)
)

fig.add_scatter(
    x=signals.index[signals['positions_y'] < 0],
    y=signals.loc[signals['positions_y'] < 0, 'y100'],
    mode='markers',
    name=f'Short {y_ticker}',
    marker=dict(symbol='triangle-down', color='red', size=12)
)

fig.update_layout(title=f'Pairs Trading (base 100): {y_ticker} vs {x_ticker}')
fig.write_html(images + "3.html", config=par84.config, include_plotlyjs="cdn")
