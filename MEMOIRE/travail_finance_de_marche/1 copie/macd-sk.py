import numpy as np
import yfinance as yf
import par84
import plotly.graph_objects as go
from plotly.subplots import make_subplots

symbol = 'KO'
images = par84.img_path+'macd-'

df = yf.Ticker(symbol).history(period='1d', interval='1m', auto_adjust=True)

short_ema = 12
long_ema = 26
df['short_ema'] = df.Close.ewm(span = short_ema, min_periods=short_ema).mean()
df['long_ema'] = df.Close.ewm(span = long_ema, min_periods=long_ema).mean()
df['positions'] = 0
df['positions'] = np.where(df['short_ema'] >= df['long_ema'],1,0)
df['signals'] = df['positions'].diff()
df['oscillator'] = df['short_ema'] - df['long_ema']
print(df.to_string())

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df.index, y=df['short_ema'], name="Short moving average", line_width=.7)
fig.add_scatter(x=df.index, y=df['long_ema'], name="Long moving average")
fig.update_layout(title_text=symbol+' MACD Oscillator')
fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")

fig = go.Figure(layout=par84.layout)
fig.add_bar(x=df.index, y=df['oscillator'])
fig.update_layout(title_text=symbol+' MACD Oscillator')
fig.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df.index, y=df.Close, name="Price")
fig.add_scatter(x=df.index, y=df['short_ema'], name="Short moving average", line_width=.7)
fig.add_scatter(x=df.index, y=df['long_ema'], name="Long moving average")
fig.add_scatter(x=df.loc[df['signals']==1].index, y=df.Close[df['signals']==1], mode='markers', marker=dict(size=12, symbol="triangle-up", color='greenyellow'), name="Long position")
fig.add_scatter(x=df.loc[df['signals']==-1].index, y=df.Close[df['signals']==-1], mode='markers', marker=dict(size=12, symbol="triangle-down", color='red'), name="Short position")
fig.update_layout(title_text=symbol+' MACD decisions')
fig.write_html(images + "3.html", config=par84.config, include_plotlyjs="cdn")
