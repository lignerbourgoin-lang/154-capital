import requests
import pandas as pd
import plotly.graph_objects as go
import par84
from datetime import datetime, timedelta

images = par84.img_path+'onchain-'
url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

ticker = "btc"
metrics = ["PriceUSD", "AdrActCnt", "TxCnt"]
window = 30
buy_level = -1.5
sell_level = 1.5
end = datetime.today().strftime("%Y-%m-%d")
start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

def coinmetrics(ticker, metrics, start, end):
    params = {"assets": ticker, "metrics": ",".join(metrics), "frequency": "1d" , "start_time": start, "end_time": end, "page_size": 10000}
    response = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(response.json()["data"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("time")
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=metrics)

def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

df = coinmetrics(ticker, metrics, start, end)

df["addr_growth"] = df["AdrActCnt"].pct_change(7)
df["tx_growth"] = df["TxCnt"].pct_change(7)
df["z_addr"] = rolling_zscore(df["addr_growth"], window)
df["z_tx"] = rolling_zscore(df["tx_growth"], window)
df["zscore"] = 0.5 * df["z_addr"] + 0.5 * df["z_tx"]
df["buy"] = (df["zscore"] > buy_level) & (df["zscore"].shift(1) <= buy_level)
df["sell"] = (df["zscore"] < sell_level) & (df["zscore"].shift(1) >= sell_level)
buy_points = df[df["buy"]]
sell_points = df[df["sell"]]

print(df)

fig = go.Figure()
fig.add_scatter(x=df.index,y=df['zscore'],name="Z-score")
fig.add_scatter(x=df.index, y=[buy_level] * len(df), name="Buy Level", line=dict(color="green", dash="dash"))
fig.add_scatter(x=df.index, y=[sell_level] * len(df), name="Sell Level", line=dict(color="red", dash="dash"))
fig.update_layout(par84.layout, title="Z-score", yaxis_title="Z", hovermode="x unified")
fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")

fig = go.Figure()
fig.add_scatter(x=df.index, y=df["PriceUSD"], name="Z-score")
fig.add_scatter(x=buy_points.index,y=buy_points["PriceUSD"],mode="markers",name="Buy",marker=dict(symbol="triangle-up", size=12, color="green"))
fig.add_scatter(x=sell_points.index,y=sell_points["PriceUSD"],mode="markers",name="Sell",marker=dict(symbol="triangle-down", size=12, color="red"))
fig.update_layout(par84.layout,title="Signals",yaxis_title="Price",hovermode="x unified")
fig.write_html(images + "2.html", config=par84.config, include_plotlyjs="cdn")
