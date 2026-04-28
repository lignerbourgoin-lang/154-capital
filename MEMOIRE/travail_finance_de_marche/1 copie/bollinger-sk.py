import yfinance as yf
import par84
import plotly.graph_objects as go

symbol = 'TSLA'
images = par84.img_path+'bollinger-'

df = yf.Ticker(symbol).history(period='1d', interval='1m', auto_adjust=True)

df['mean'] = df['Close'].rolling(window=20).mean()
df['std'] = df['Close'].rolling(window=20).std()
df['upper'] = df['mean'] + 2 * df['std']
df['lower'] = df['mean'] - 2 * df['std']

df['buy_signal'] = (
    (df['Close'] < df['lower']) &
    (df['Close'].shift(1) >= df['lower'].shift(1))
)

df['sell_signal'] = (
    (df['Close']> df['upper']) &
    (df['Close'].shift(1) <= df['upper'].shift(1))
)

df['buy_price'] = df['Close'].where(df['buy_signal'])
df['sell_price'] = df['Close'].where(df['sell_signal'])

fig = go.Figure()

fig.add_scatter(x=df.index, y=df['Close'], name="Price")
fig.add_scatter(x=df.index, y=df['mean'], name="Moving average", line_width=0.7)
fig.add_scatter(
    x=df.index, y=df['lower'],
    name="Lower band",
    line_color="rgba(255,255,255,0.1)"
)
fig.add_scatter(
    x=df.index, y=df['upper'],
    name="Upper band",
    fill="tonexty",
    fillcolor="rgba(255,255,255,0.1)",
    line_color="rgba(255,255,255,0.1)"
)

fig.add_scatter(
    x=df.index,
    y=df['buy_price'],
    mode='markers',
    name='Buy signal',
    marker=dict(symbol='triangle-up', size=10, color='green')
)

fig.add_scatter(
    x=df.index,
    y=df['sell_price'],
    mode='markers',
    name='Sell signal',
    marker=dict(symbol='triangle-down', size=10, color='red')
)

fig.update_layout(par84.layout, title_text=symbol+' Bollinger bands')
fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")
