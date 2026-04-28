import yfinance as yf
import par84
import plotly.graph_objects as go
from plotly.offline import plot

symbol = 'TSLA'
images = par84.img_path+'5-'

df = yf.Ticker(symbol).history(period='1y', auto_adjust=True)
print(df.tail())

fig = go.Figure(layout=par84.layout)
fig.add_scatter(x=df.index, y=df.Close)
plot(fig, filename=images+'1.html', config=par84.config, auto_open=False)

