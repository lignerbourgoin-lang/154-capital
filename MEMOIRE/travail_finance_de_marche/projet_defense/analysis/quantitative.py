import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
from scipy.stats import norm

from config import TICKERS, LAYOUT, CONFIG, CHARTS_QUANTITATIVE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _write(fig: go.Figure, path: Path) -> None:
    fig.write_html(str(path), config=CONFIG, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def build_montecarlo(name: str, close: pd.Series,
                     t_intervals: int = 30, simulations: int = 200) -> None:
    returns = close.pct_change(fill_method=None).dropna()
    mu      = returns.mean()
    sigma   = returns.std()

    sim_returns = mu + sigma * norm.ppf(np.random.rand(t_intervals, simulations))
    sim_simple  = np.exp(sim_returns)

    price_list      = np.zeros_like(sim_simple)
    price_list[0]   = close.iloc[-1]
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * sim_simple[t]

    worst   = price_list[-1].min()
    avg     = price_list[-1].mean()
    best    = price_list[-1].max()
    std_end = price_list[-1].std()

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(
        title_text = f"{name} — Monte Carlo ({simulations} paths, {t_intervals} days)",
        xaxis_title = "Days",
        yaxis_title = "Price",
    )

    for i in range(simulations):
        fig.add_scatter(x=np.arange(t_intervals), y=price_list[:, i],
                        showlegend=False, line=dict(width=0.5),
                        opacity=0.4)

    # Reference lines
    fig.add_scatter(x=[t_intervals - 1], y=[avg],
                    mode="markers+text",
                    text=[f"Avg  {avg:,.0f}"],
                    textposition="middle right",
                    marker=dict(size=10, color="white"),
                    name="Average")
    fig.add_scatter(x=[t_intervals - 1], y=[worst],
                    mode="markers+text",
                    text=[f"Worst  {worst:,.0f}"],
                    textposition="middle right",
                    marker=dict(size=10, color="red"),
                    name="Worst")
    fig.add_scatter(x=[t_intervals - 1], y=[best],
                    mode="markers+text",
                    text=[f"Best  {best:,.0f}"],
                    textposition="middle right",
                    marker=dict(size=10, color="limegreen"),
                    name="Best")

    _write(fig, CHARTS_QUANTITATIVE / f"montecarlo_{_slug(name)}.html")


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def build_garch(name: str, close: pd.Series, forecast_horizon: int = 10) -> None:
    returns = 100 * np.log(close / close.shift(1)).dropna()

    am  = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")

    cond_vol = res.conditional_volatility
    cond_var = cond_vol ** 2

    df = pd.DataFrame(index=returns.index)
    df["Price"]  = close.reindex(returns.index)
    df["Return"] = returns.values
    df["Sigma"]  = cond_vol
    df["Sigma2"] = cond_var

    # Forecast
    forecast      = res.forecast(horizon=forecast_horizon)
    var_forecast  = forecast.variance.iloc[-1]
    sigma_forecast = np.sqrt(var_forecast)
    future_dates  = pd.date_range(df.index[-1] + BDay(1), periods=forecast_horizon, freq="B")

    # --- Fig 1 : price
    fig1 = go.Figure(layout=LAYOUT)
    fig1.update_layout(title=f"{name} — Price", yaxis_title="Price")
    fig1.add_scatter(x=df.index, y=df["Price"], name="Price")
    _write(fig1, CHARTS_QUANTITATIVE / f"garch_price_{_slug(name)}.html")

    # --- Fig 2 : returns
    fig2 = go.Figure(layout=LAYOUT)
    fig2.update_layout(title=f"{name} — Daily Returns (%)", yaxis_title="Return (%)")
    fig2.add_scatter(x=df.index, y=df["Return"], name="Returns (%)")
    _write(fig2, CHARTS_QUANTITATIVE / f"garch_returns_{_slug(name)}.html")

    # --- Fig 3 : conditional volatility + forecast
    lookback    = 200
    hist_index  = df.index[-lookback:]
    x_connect   = [hist_index[-1], future_dates[0]]
    y_connect   = [df["Sigma"].iloc[-1], sigma_forecast.iloc[0]]

    fig3 = go.Figure(layout=LAYOUT)
    fig3.update_layout(
        title      = f"{name} — GARCH(1,1) Conditional Volatility + {forecast_horizon}-day Forecast",
        yaxis_title = "Volatility (%)",
        hovermode  = "x unified",
    )
    fig3.add_scatter(x=hist_index, y=df["Sigma"].iloc[-lookback:], name="Historical Volatility")
    fig3.add_scatter(x=x_connect,  y=y_connect,
                     mode="lines", line=dict(color="limegreen"), showlegend=False)
    fig3.add_scatter(x=future_dates, y=sigma_forecast,
                     mode="lines", name="Forecasted Volatility", line=dict(color="limegreen"))
    _write(fig3, CHARTS_QUANTITATIVE / f"garch_volatility_{_slug(name)}.html")


# ---------------------------------------------------------------------------
# OLS Regression  (same spirit as regression-sk.py and regression-beta-sk.py)
# ---------------------------------------------------------------------------

def build_ols(name_y: str, name_x: str,
              close_y: pd.Series, close_x: pd.Series) -> None:
    """
    Two regressions mirroring the course:
      1. Price level regression (base 100) — like regression-sk.py
      2. Returns regression (beta) — like regression-beta-sk.py
    """
    # Align series
    df = pd.DataFrame({name_y: close_y, name_x: close_x}).dropna()

    slug = f"{_slug(name_y)}_vs_{_slug(name_x)}"

    # ---- 1. Price level (base 100) ----------------------------------------
    df100 = df / df.iloc[0] * 100
    y     = df100[name_y]
    X     = sm.add_constant(df100[[name_x]], has_constant="add")
    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta  = model.params[name_x]
    x_grid = np.linspace(df100[name_x].min(), df100[name_x].max(), 200)
    y_grid = alpha + beta * x_grid

    r2      = model.rsquared
    pval_a  = model.pvalues["const"]
    pval_b  = model.pvalues[name_x]
    tval_a  = model.tvalues["const"]
    tval_b  = model.tvalues[name_x]

    stats_text = (
        f"<b>R² = {r2:.4f}</b><br>"
        f"α = {alpha:.4f}  (t={tval_a:.2f}, p={pval_a:.4f})<br>"
        f"β = {beta:.4f}  (t={tval_b:.2f}, p={pval_b:.4f})<br>"
        f"N = {int(model.nobs)}"
    )

    # Scatter + regression line
    fig1 = go.Figure(layout=LAYOUT)
    fig1.update_layout(
        title       = f"OLS Price (base 100) : {name_y} vs {name_x}  —  R²={r2:.4f}  β={beta:.4f}",
        xaxis_title = name_x,
        yaxis_title = name_y,
    )
    fig1.add_scatter(x=df100[name_x], y=df100[name_y],
                     mode="markers", name="Data", marker=dict(size=3, opacity=0.5))
    fig1.add_scatter(x=x_grid, y=y_grid, name=f"Regression (β={beta:.4f})")
    fig1.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text=stats_text,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1,
        borderpad=6,
    )
    _write(fig1, CHARTS_QUANTITATIVE / f"ols_price_{slug}.html")

    # Time-series estimated vs real
    df100["estim"] = model.predict(X)
    fig2 = go.Figure(layout=LAYOUT)
    fig2.update_layout(
        title       = f"OLS Price Estimation : {name_y} explained by {name_x}",
        hovermode   = "x unified",
    )
    fig2.add_scatter(x=df100.index, y=df100[name_y],    name=f"{name_y} real")
    fig2.add_scatter(x=df100.index, y=df100["estim"],   name=f"{name_y} estimated", line=dict(dash="dash"))
    _write(fig2, CHARTS_QUANTITATIVE / f"ols_price_ts_{slug}.html")

    # ---- 2. Returns (beta) ------------------------------------------------
    rets = df.pct_change().dropna()
    yr   = rets[name_y]
    Xr   = sm.add_constant(rets[[name_x]], has_constant="add")
    model_r = sm.OLS(yr, Xr).fit()

    alpha_r = model_r.params["const"]
    beta_r  = model_r.params[name_x]
    xr_grid = np.linspace(rets[name_x].min(), rets[name_x].max(), 200)
    yr_grid = alpha_r + beta_r * xr_grid

    r2_r     = model_r.rsquared
    pval_ar  = model_r.pvalues["const"]
    pval_br  = model_r.pvalues[name_x]
    tval_ar  = model_r.tvalues["const"]
    tval_br  = model_r.tvalues[name_x]

    stats_text_r = (
        f"<b>R² = {r2_r:.4f}</b><br>"
        f"α = {alpha_r:.4f}  (t={tval_ar:.2f}, p={pval_ar:.4f})<br>"
        f"β = {beta_r:.4f}  (t={tval_br:.2f}, p={pval_br:.4f})<br>"
        f"N = {int(model_r.nobs)}"
    )

    fig3 = go.Figure(layout=LAYOUT)
    fig3.update_layout(
        title       = f"OLS Returns (beta) : {name_y} vs {name_x}  —  R²={r2_r:.4f}  β={beta_r:.4f}",
        xaxis_title = f"Return {name_x}",
        yaxis_title = f"Return {name_y}",
    )
    fig3.add_scatter(x=rets[name_x], y=rets[name_y],
                     mode="markers", name="Data", marker=dict(size=3, opacity=0.5))
    fig3.add_scatter(x=xr_grid, y=yr_grid, name=f"Regression (β={beta_r:.4f})")
    fig3.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text=stats_text_r,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1,
        borderpad=6,
    )
    _write(fig3, CHARTS_QUANTITATIVE / f"ols_returns_{slug}.html")

    # Cumulative returns
    cum = (1 + rets).cumprod() * 100
    rets["estim"] = model_r.predict(Xr)
    cum["estim"]  = (1 + rets["estim"]).cumprod() * 100

    fig4 = go.Figure(layout=LAYOUT)
    fig4.update_layout(
        title     = f"OLS Returns Estimation : {name_y} explained by {name_x}",
        hovermode = "x unified",
    )
    fig4.add_scatter(x=cum.index, y=cum[name_y],  name=f"{name_y} real")
    fig4.add_scatter(x=cum.index, y=cum["estim"], name=f"{name_y} estimated", line=dict(dash="dash"))
    _write(fig4, CHARTS_QUANTITATIVE / f"ols_returns_ts_{slug}.html")


# ---------------------------------------------------------------------------
# NAV base 100  (placeholder — will be plugged to optimisation later)
# ---------------------------------------------------------------------------

def build_nav(prices: pd.DataFrame) -> None:
    df100 = prices / prices.iloc[0] * 100

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(
        title       = "Performance base 100 — Defence Basket",
        yaxis_title = "Base 100",
        hovermode   = "x unified",
    )
    for col in df100.columns:
        fig.add_scatter(x=df100.index, y=df100[col], name=col)

    _write(fig, CHARTS_QUANTITATIVE / "nav_base100.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(prices: pd.DataFrame) -> None:
    tickers_list = list(TICKERS.keys())
    names_list   = list(TICKERS.values())

    for ticker, name in TICKERS.items():
        close = prices[name]
        print(f"  Quant — {name}")
        build_montecarlo(name, close)
        build_garch(     name, close)

    # OLS : each company vs the next one (circular), mirroring the course style
    for i in range(len(names_list)):
        name_y = names_list[i]
        name_x = names_list[(i + 1) % len(names_list)]
        build_ols(name_y, name_x, prices[name_y], prices[name_x])

    build_nav(prices)


if __name__ == "__main__":
    prices = pd.read_excel(
        Path(__file__).parent.parent / "data" / "prices.xlsx",
        index_col=0, parse_dates=True,
    )
    run(prices)
    print("Done — charts written to", CHARTS_QUANTITATIVE)
