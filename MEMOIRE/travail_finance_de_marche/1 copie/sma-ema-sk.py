import yfinance as yf
import par84
import plotly.graph_objects as go

images = par84.img_path+'sma-ema-'

ticker = '^GSPC'
start = '2008-07-01'
end   = '2008-12-01'
sma_window = 20
ema_span   = 20

df = yf.Ticker(ticker).history(start=start, end=end)
#df = yf.Ticker(ticker).history(period='1y')

df[f'SMA_{sma_window}'] = df['Close'].rolling(window=sma_window).mean()
df[f'EMA_{ema_span}']   = df['Close'].ewm(span=ema_span, adjust=False).mean()

print(df[['Close', f'SMA_{sma_window}', f'EMA_{ema_span}']].tail())

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df.index, y=df['Close'], name='Close Price')
fig.add_scatter(x=df.index, y=df[f'SMA_{sma_window}'], name=f'SMA {sma_window}', line=dict(dash='dash'))
fig.add_scatter(x=df.index, y=df[f'EMA_{ema_span}'], name=f'EMA {ema_span}', line=dict(dash='dot'))
fig.update_layout(title=f'{ticker} Closing Price with {sma_window}-Day SMA & {ema_span}-Day EMA', hovermode='x unified')
fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")
