import numpy as np
import yfinance as yf
import par84
import statsmodels.api as sm
import plotly.graph_objects as go
from pathlib import Path

images = par84.img_path + 'regression-'

tickers = ['PEP', 'KO']
y_ticker, x_ticker = tickers[0], tickers[1]

df = yf.Tickers(tickers).history(period='5y', auto_adjust=True)['Close']
df = df.dropna(subset=tickers)
df100 = df/df.iloc[0]*100

y = df100[y_ticker]
X = sm.add_constant(df100[[x_ticker]], has_constant='add')

model = sm.OLS(y, X).fit() 
print(model.summary())

Path(images + "1.html").write_text(
    "<body style='color:white; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;'>" +
    model.summary().as_html(),
    encoding="utf-8"
)

alpha = model.params['const']
beta = model.params[x_ticker]

x_grid = np.linspace(df100[x_ticker].min(), df100[x_ticker].max(), 200)
y_grid = alpha + beta * x_grid

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df100[x_ticker], y=df100[y_ticker], mode='markers', name='Data')
fig.add_scatter(x=x_grid, y=y_grid, name=f'Regression line (β={beta:.2f})')
fig.update_layout(xaxis_title=x_ticker, yaxis_title=y_ticker)
fig.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")

df100['estim'] = model.predict(X)

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df100.index, y=df100[y_ticker], name=f'{y_ticker} real')
fig.add_scatter(x=df100.index, y=df100['estim'], name=f'{y_ticker} estimated')
fig.update_layout(title=f'Estimation of {y_ticker} by {x_ticker}')
fig.write_html(images + "3.html", config=par84.config, include_plotlyjs="cdn")








