import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import TICKERS, LAYOUT, CONFIG, CHARTS_TECHNICAL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Company display name -> safe filename token."""
    return name.lower().replace(" ", "_")


def _write(fig: go.Figure, path: Path) -> None:
    fig.write_html(str(path), config=CONFIG, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _ichimoku(df: pd.DataFrame) -> dict:
    """Return all Ichimoku series as a dict of pd.Series."""
    high_9  = df["High"].rolling(9).max()
    low_9   = df["Low"].rolling(9).min()
    high_26 = df["High"].rolling(26).max()
    low_26  = df["Low"].rolling(26).min()
    high_52 = df["High"].rolling(52).max()
    low_52  = df["Low"].rolling(52).min()

    tenkan   = (high_9  + low_9)  / 2
    kijun    = (high_26 + low_26) / 2
    span_a   = (tenkan  + kijun)  / 2
    span_b   = (high_52 + low_52) / 2
    chikou   = df["Close"].shift(-26)

    future       = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=26)
    ext_index    = df.index.append(future)
    fwd_span_a   = pd.Series(index=ext_index, dtype=float)
    fwd_span_b   = pd.Series(index=ext_index, dtype=float)
    fwd_span_a.iloc[26 : 26 + len(df)] = span_a.values
    fwd_span_b.iloc[26 : 26 + len(df)] = span_b.values

    return dict(
        tenkan   = tenkan,
        kijun    = kijun,
        span_a   = fwd_span_a,
        span_b   = fwd_span_b,
        chikou   = chikou,
        ext_idx  = ext_index,
    )


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    rs     = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Chart builders  (one HTML per indicator per ticker)
# ---------------------------------------------------------------------------

def build_ichimoku(ticker: str, name: str, df: pd.DataFrame) -> None:
    ichi = _ichimoku(df)
    idx  = ichi["ext_idx"]
    sa, sb = ichi["span_a"], ichi["span_b"]

    bullish_a = sa.where(sa >= sb)
    bullish_b = sb.where(sa >= sb)
    bearish_a = sa.where(sa <  sb)
    bearish_b = sb.where(sa <  sb)

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(title_text=f"{name} — Ichimoku Cloud", hovermode="x unified")

    # Bullish cloud
    fig.add_scatter(x=idx, y=bullish_a, mode="lines", line=dict(width=0),
                    name="Bullish upper", showlegend=False)
    fig.add_scatter(x=idx, y=bullish_b, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(173,255,47,0.20)",
                    name="Bullish cloud")

    # Bearish cloud
    fig.add_scatter(x=idx, y=bearish_a, mode="lines", line=dict(width=0),
                    name="Bearish upper", showlegend=False)
    fig.add_scatter(x=idx, y=bearish_b, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(255,0,76,0.20)",
                    name="Bearish cloud")

    # Price + lines
    fig.add_candlestick(x=df.index,
                        open=df["Open"], high=df["High"],
                        low=df["Low"],  close=df["Close"],
                        increasing_line_color="rgba(173,255,47,1)",
                        decreasing_line_color="rgba(255,0,76,1)",
                        name="Price")
    fig.add_scatter(x=df.index,  y=ichi["tenkan"], name="Tenkan-sen",  line=dict(width=1))
    fig.add_scatter(x=df.index,  y=ichi["kijun"],  name="Kijun-sen",   line=dict(width=1, dash="dash"))
    fig.add_scatter(x=idx,       y=sa,             name="Senkou A",    line=dict(width=1))
    fig.add_scatter(x=idx,       y=sb,             name="Senkou B",    line=dict(width=1))
    fig.add_scatter(x=df.index,  y=ichi["chikou"], name="Chikou Span", line=dict(width=1, dash="dot"))

    _write(fig, CHARTS_TECHNICAL / f"ichimoku_{_slug(name)}.html")


def build_macd(ticker: str, name: str, df: pd.DataFrame) -> None:
    macd_line, signal_line, histogram = _macd(df["Close"])

    positions = (macd_line >= signal_line).astype(int)
    signals   = positions.diff()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)
    fig.update_layout(LAYOUT, title_text=f"{name} — MACD", hovermode="x unified")

    # Price + signals
    fig.add_scatter(x=df.index, y=df["Close"], name="Price",
                    line=dict(width=1.2), row=1, col=1)
    fig.add_scatter(x=df.index[signals == 1],  y=df["Close"][signals == 1],
                    mode="markers", marker=dict(size=10, symbol="triangle-up",   color="limegreen"),
                    name="Long signal",  row=1, col=1)
    fig.add_scatter(x=df.index[signals == -1], y=df["Close"][signals == -1],
                    mode="markers", marker=dict(size=10, symbol="triangle-down", color="red"),
                    name="Short signal", row=1, col=1)

    # MACD panel
    fig.add_scatter(x=df.index, y=macd_line,   name="MACD",   line=dict(width=1),          row=2, col=1)
    fig.add_scatter(x=df.index, y=signal_line, name="Signal", line=dict(width=1, dash="dash"), row=2, col=1)
    fig.add_bar(    x=df.index, y=histogram,   name="Hist",                                 row=2, col=1)

    _write(fig, CHARTS_TECHNICAL / f"macd_{_slug(name)}.html")


def build_rsi(ticker: str, name: str, df: pd.DataFrame, window: int = 14) -> None:
    rsi = _rsi(df["Close"], window)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)
    fig.update_layout(LAYOUT, title_text=f"{name} — RSI ({window})", hovermode="x unified")

    fig.add_scatter(x=df.index, y=df["Close"], name="Price", line=dict(width=1.2), row=1, col=1)

    fig.add_scatter(x=df.index, y=rsi,              name=f"RSI {window}", line=dict(width=1.2), row=2, col=1)
    fig.add_scatter(x=df.index, y=[70] * len(df),   name="Overbought",    line=dict(dash="dash", color="red",       width=1), row=2, col=1)
    fig.add_scatter(x=df.index, y=[30] * len(df),   name="Oversold",      line=dict(dash="dash", color="limegreen", width=1), row=2, col=1)
    fig.add_scatter(x=df.index, y=[50] * len(df),   name="Neutral",       line=dict(dash="dot",  color="grey",      width=1), row=2, col=1)

    _write(fig, CHARTS_TECHNICAL / f"rsi_{_slug(name)}.html")


def build_ma(ticker: str, name: str, df: pd.DataFrame) -> None:
    close = df["Close"]
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(title_text=f"{name} — Moving Averages", hovermode="x unified")

    fig.add_scatter(x=df.index, y=close,  name="Close",   line=dict(width=1.5))
    fig.add_scatter(x=df.index, y=sma20,  name="SMA 20",  line=dict(dash="dash", width=1))
    fig.add_scatter(x=df.index, y=sma50,  name="SMA 50",  line=dict(dash="dash", width=1))
    fig.add_scatter(x=df.index, y=sma200, name="SMA 200", line=dict(dash="dot",  width=1.5))
    fig.add_scatter(x=df.index, y=ema20,  name="EMA 20",  line=dict(dash="dot",  width=1))

    _write(fig, CHARTS_TECHNICAL / f"ma_{_slug(name)}.html")


# ---------------------------------------------------------------------------
# Main — generate all charts for all tickers
# ---------------------------------------------------------------------------

def run(prices: pd.DataFrame, ohlcv: dict[str, pd.DataFrame]) -> None:
    for ticker, name in TICKERS.items():
        print(f"  Technical — {name}")
        df = ohlcv[ticker]
        build_ichimoku(ticker, name, df)
        build_macd(    ticker, name, df)
        build_rsi(     ticker, name, df)
        build_ma(      ticker, name, df)


if __name__ == "__main__":
    import yfinance as yf
    from config import START, END

    # Fetch OHLCV for all tickers
    ohlcv = {}
    for ticker in TICKERS:
        ohlcv[ticker] = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True)

    # Close prices (wide)
    prices = pd.DataFrame({name: ohlcv[t]["Close"] for t, name in TICKERS.items()})

    run(prices, ohlcv)
    print("Done — charts written to", CHARTS_TECHNICAL)
