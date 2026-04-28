import yfinance as yf
import par84
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from scipy.stats import norm

symbol = 'BTC-USD'
images = par84.img_path+'montecarlo-'

df = yf.Ticker(symbol).history(period="1y")['Close']
returns = df.pct_change().dropna()
mean = returns.mean()
variance = returns.var()
standard_deviation = returns.std()
print(f'Standard deviation is {standard_deviation:.2%}')

t_intervals = 10
simulations = 100

daily_returns_simulated = mean + standard_deviation * norm.ppf(np.random.rand(t_intervals, simulations))
daily_simplereturns_simulated = np.exp(daily_returns_simulated)
price_list = np.zeros_like(daily_simplereturns_simulated)
price_list[0] = df.iloc[-1]
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_simplereturns_simulated[t]

fig = go.Figure(layout=par84.layout)
for i in range(simulations):
    fig.add_scatter(x=np.arange(t_intervals), y=price_list[:, i], showlegend=False)
fig.update_layout(title_text=f"{symbol} – Monte Carlo Simulation ({simulations} paths, {t_intervals} days)", xaxis_title="Days", yaxis_title="Price")
plot(fig, filename=images+'1.html', config=par84.config, auto_open=False)

print(f"Worst scenario : $ {round(price_list[-1].min())}")
print(f"Average scenario : $ {round(price_list[-1].mean())}")
print(f"Best scenario : $ {round(price_list[-1].max())}")
UpperInterval = price_list[-1].mean() + price_list[-1].std()
LowerInterval = price_list[-1].mean() - price_list[-1].std()
print(f'Price after {t_intervals} days should be between ${round(LowerInterval, 2)} and ${round(UpperInterval, 2)}')
