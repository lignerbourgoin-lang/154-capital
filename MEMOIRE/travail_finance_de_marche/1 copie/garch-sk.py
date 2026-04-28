import numpy as np
import pandas as pd
import yfinance as yf
import par84
import plotly.graph_objects as go
from arch import arch_model
from pandas.tseries.offsets import BDay

images = par84.img_path + 'garch-'

ticker = "AAPL"
prices = yf.Ticker(ticker).history(period="1y")['Close']
returns = 100*np.log(prices/prices.shift(1)).dropna()

am = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
res = am.fit(disp="off")
print(res.summary())
conditional_vol = res.conditional_volatility    
conditional_var = conditional_vol ** 2           

df = pd.DataFrame(index=returns.index)
df["Price"] = prices.reindex(returns.index)
df["Return"] = returns.values
df["Sigma"] = conditional_vol
df["Sigma2"] = conditional_var

fig_price = go.Figure(layout=par84.layout)
fig_price.add_scatter(x=df.index, y=df['Price'], name="Price")
fig_price.update_layout(title=f"{ticker} Price",  yaxis_title="Price")
fig_price.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")

fig_ret = go.Figure(layout=par84.layout)
fig_ret.add_scatter(x=df.index, y=df['Return'], name="Returns (%)")
fig_ret.update_layout(title=f"Daily Returns (%) - {ticker}", yaxis_title="Return (%)")
fig_ret.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")

fig_sigma = go.Figure(layout=par84.layout)
fig_sigma.add_scatter(x=df.index, y=df['Sigma'], name="Volatility (%)")
fig_sigma.update_layout(title=f"Conditional Volatility (GARCH(1,1)) - {ticker}",  yaxis_title="Volatility (%)")
fig_sigma.write_html(images + "3.html", config=par84.config, include_plotlyjs="cdn")

fig_sigma2 = go.Figure(layout=par84.layout)
fig_sigma2.add_scatter(x=df.index, y=df['Sigma2'], name="Variance (%²)")
fig_sigma2.update_layout(title=f"Conditional Variance (GARCH(1,1)) - {ticker}", yaxis_title="Variance (%²)")
fig_sigma2.write_html(images + "4.html", config=par84.config, include_plotlyjs="cdn")

forecast_horizon = 10
forecast = res.forecast(horizon=forecast_horizon)
var_forecast = forecast.variance.iloc[-1]
print(f"\nVariance forecast for {forecast_horizon} days ahead:\n{var_forecast}")
sigma_forecast = np.sqrt(var_forecast)
future_dates = pd.date_range(df.index[-1] + BDay(1), periods=forecast_horizon, freq="B")
lookback = 200
hist_index = df.index[-lookback:]

fig_forecast = go.Figure(layout=par84.layout)
fig_forecast.add_scatter(x=hist_index, y=df["Sigma"].iloc[-lookback:], name="Historical Volatility")
x_connect = [hist_index[-1], future_dates[0]]
y_connect = [df["Sigma"].iloc[-1], sigma_forecast[0]]
fig_forecast.add_scatter(x=x_connect, y=y_connect, mode="lines", line=dict(color="limegreen"), showlegend=False)
fig_forecast.add_scatter(x=future_dates, y=sigma_forecast, mode="lines", name="Forecasted Volatility", line=dict(color="limegreen"))
fig_forecast.update_layout(title=f"GARCH(1,1) Volatility Forecast ({forecast_horizon} days) - {ticker}", yaxis_title="Volatility (%)")
fig_forecast.write_html(images + "5.html", config=par84.config, include_plotlyjs="cdn")
