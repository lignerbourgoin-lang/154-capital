import numpy as np
import yfinance as yf
import par84
import plotly.graph_objects as go

symbol = 'AAPL'
images = par84.img_path + 'ha-'

df = yf.Ticker(symbol).history(period='1d', interval='1m', auto_adjust=True)

max_consecutive_signals = 3

def heikin_ashi(df1):
    df1 = df1.copy()
    df1.reset_index(inplace=True)

    # Heikin-Ashi close = average of the 4 prices
    df1['HA close'] = (df1['Open'] + df1['High'] + df1['Low'] + df1['Close']) / 4

    # Heikin-Ashi open
    df1['HA open'] = 0.0
    df1.at[0, 'HA open'] = (df1.at[0, 'Open'] + df1.at[0, 'Close']) / 2

    for n in range(1, len(df1)):
        previous_ha_open = df1.at[n - 1, 'HA open']
        previous_ha_close = df1.at[n - 1, 'HA close']
        df1.at[n, 'HA open'] = (previous_ha_open + previous_ha_close) / 2

    # Heikin-Ashi high and low
    df1['HA high'] = df1[['HA open', 'High', 'HA close']].max(axis=1)
    df1['HA low'] = df1[['HA open', 'Low', 'HA close']].min(axis=1)

    return df1


def signal_generation(df, method):
    df1 = method(df)
    df1['signals'] = 0

    long_count = 0
    short_count = 0

    for n in range(1, len(df1)):
        # Candle body size
        previous_body = abs(df1.at[n - 1, 'HA close'] - df1.at[n - 1, 'HA open'])
        current_body = abs(df1.at[n, 'HA close'] - df1.at[n, 'HA open'])

        # Candle direction
        bullish_now = df1.at[n, 'HA close'] > df1.at[n, 'HA open']
        bearish_now = df1.at[n, 'HA close'] < df1.at[n, 'HA open']
        bullish_before = df1.at[n - 1, 'HA close'] > df1.at[n - 1, 'HA open']
        bearish_before = df1.at[n - 1, 'HA close'] < df1.at[n - 1, 'HA open']

        # Wick conditions
        no_lower_wick = np.isclose(df1.at[n, 'HA open'], df1.at[n, 'HA low'])
        no_upper_wick = np.isclose(df1.at[n, 'HA open'], df1.at[n, 'HA high'])

        # Strong continuation candles
        bullish_signal = bullish_now and bullish_before and no_lower_wick and current_body >= previous_body
        bearish_signal = bearish_now and bearish_before and no_upper_wick and current_body <= previous_body

        if bullish_signal:
            long_count += 1
            short_count = 0

            if long_count <= max_consecutive_signals:
                df1.at[n, 'signals'] = 1

        elif bearish_signal:
            short_count += 1
            long_count = 0

            if short_count <= max_consecutive_signals:
                df1.at[n, 'signals'] = -1

        else:
            long_count = 0
            short_count = 0

    return df1


df1 = signal_generation(df, heikin_ashi)
new = df1.iloc[0:].copy()

xcol = new.columns[0]

long_color = 'rgba(173,255,47,1)'
short_color = 'rgba(255,0,76,1)'

fig = go.Figure(layout=par84.layout)
fig.update_layout(title_text=symbol + ' Heikin Ashi')

fig.add_candlestick(
    x=new[xcol],
    open=new['HA open'],
    high=new['HA high'],
    low=new['HA low'],
    close=new['HA close'],
    increasing_line_color=long_color,
    decreasing_line_color=short_color,
    name='Heikin Ashi'
)

fig.add_scatter(
    x=new.loc[new['signals'] == 1, xcol],
    y=new.loc[new['signals'] == 1, 'HA low'],
    mode='markers',
    marker=dict(
        size=12,
        symbol="triangle-up",
        color=long_color,
        line=dict(width=1, color='black')
    ),
    name="Long position"
)

fig.add_scatter(
    x=new.loc[new['signals'] == -1, xcol],
    y=new.loc[new['signals'] == -1, 'HA high'],
    mode='markers',
    marker=dict(
        size=12,
        symbol="triangle-down",
        color=short_color,
        line=dict(width=1, color='black')
    ),
    name="Short position"
)

fig.write_html(images + "1.html", config=par84.config, include_plotlyjs="cdn")
