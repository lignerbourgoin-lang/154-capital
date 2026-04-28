import numpy as np
import yfinance as yf
import par84
import statsmodels.api as sm
import plotly.graph_objects as go
from pathlib import Path

images = par84.img_path + 'regression-beta-'

tickers = ['PEP', 'KO']
y_ticker, x_ticker = tickers[0], tickers[1]
df = yf.download(tickers, period="1d", interval="1m")['Close']

returns = df.pct_change().dropna()

y = returns[y_ticker]
X = sm.add_constant(returns[[x_ticker]], has_constant='add')
model = sm.OLS(y, X).fit()
print(model.summary())

Path(images + "1.html").write_text(
    "<body style='color:white; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;'>" +
    model.summary().as_html(),
    encoding="utf-8"
)

alpha = model.params['const']
beta  = model.params[x_ticker]
print(f"\nalpha: {alpha:.5f}")
print(f"beta: {beta:.2f}")

x_grid = np.linspace(returns[x_ticker].min(), returns[x_ticker].max(), 200)
y_grid = alpha + beta * x_grid

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=returns[x_ticker], y=returns[y_ticker], mode='markers', name='Data')
fig.add_scatter(x=x_grid, y=y_grid, name=f'Regression line (β={beta:.2f})')
fig.update_layout(xaxis_title=f'Return {x_ticker}', yaxis_title=f'Return {y_ticker}')
fig.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")

returns['estim'] = model.predict(X)

cum_returns = (1 + returns).cumprod() * 100

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=cum_returns.index, y=cum_returns[y_ticker], name=f'{y_ticker} real')
fig.add_scatter(x=cum_returns.index, y=cum_returns['estim'], name=f'{y_ticker} estimated')
fig.update_layout(title=f'Estimation of {y_ticker} by {x_ticker}')
fig.write_html(images + "3.html", config=par84.config, include_plotlyjs="cdn")







