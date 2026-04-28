import pandas as pd
import yfinance as yf
import par84
import plotly.graph_objects as go

symbol = 'AAPL'
images = par84.img_path + 'ichimoku-'

df = yf.Ticker(symbol).history(period='6mo', interval='1d', auto_adjust=True)
print(df.tail())

df = df.copy()

high_9 = df['High'].rolling(window=9).max()
low_9 = df['Low'].rolling(window=9).min()
df['tenkan_sen'] = (high_9 + low_9) / 2

high_26 = df['High'].rolling(window=26).max()
low_26 = df['Low'].rolling(window=26).min()
df['kijun_sen'] = (high_26 + low_26) / 2

df['senkou_span_a_raw'] = (df['tenkan_sen'] + df['kijun_sen']) / 2

high_52 = df['High'].rolling(window=52).max()
low_52 = df['Low'].rolling(window=52).min()
df['senkou_span_b_raw'] = (high_52 + low_52) / 2

df['chikou_span'] = df['Close'].shift(-26)

future_index = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=26)
extended_index = df.index.append(future_index)

senkou_span_a = pd.Series(index=extended_index, dtype=float)
senkou_span_b = pd.Series(index=extended_index, dtype=float)

senkou_span_a.iloc[26:26 + len(df)] = df['senkou_span_a_raw'].values
senkou_span_b.iloc[26:26 + len(df)] = df['senkou_span_b_raw'].values

bullish_a = senkou_span_a.where(senkou_span_a >= senkou_span_b)
bullish_b = senkou_span_b.where(senkou_span_a >= senkou_span_b)

bearish_a = senkou_span_a.where(senkou_span_a < senkou_span_b)
bearish_b = senkou_span_b.where(senkou_span_a < senkou_span_b)

fig = go.Figure(layout=par84.layout)
fig.update_layout(title_text=symbol + ' Ichimoku Cloud')

# Bullish cloud
fig.add_scatter(
    x=extended_index,
    y=bullish_a,
    mode='lines',
    line=dict(width=0),
    name='Bullish cloud upper',
    showlegend=False
)
fig.add_scatter(
    x=extended_index,
    y=bullish_b,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(173,255,47,0.25)',
    name='Bullish cloud'
)

# Bearish cloud
fig.add_scatter(
    x=extended_index,
    y=bearish_a,
    mode='lines',
    line=dict(width=0),
    name='Bearish cloud upper',
    showlegend=False
)
fig.add_scatter(
    x=extended_index,
    y=bearish_b,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(255,0,76,0.20)',
    name='Bearish cloud'
)

fig.add_candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='rgba(173,255,47,1)',
    decreasing_line_color='rgba(255,0,76,1)',
    name='Price'
)
fig.add_scatter(
    x=df.index,
    y=df['tenkan_sen'],
    mode='lines',
    name='Tenkan-sen'
)
fig.add_scatter(
    x=df.index,
    y=df['kijun_sen'],
    mode='lines',
    name='Kijun-sen'
)
fig.add_scatter(
    x=extended_index,
    y=senkou_span_a,
    mode='lines',
    name='Senkou Span A'
)
fig.add_scatter(
    x=extended_index,
    y=senkou_span_b,
    mode='lines',
    name='Senkou Span B'
)
fig.add_scatter(
    x=df.index,
    y=df['chikou_span'],
    mode='lines',
    name='Chikou Span'
)
fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")
