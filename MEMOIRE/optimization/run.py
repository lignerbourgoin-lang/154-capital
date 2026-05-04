"""
Point d'entree principal — Defence Basket Portfolio Optimization
================================================================
Lance les 4 strategies + ML et genere tous les graphiques.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf

from models.markowitz      import MarkowitzOptimizer
from models.risk_parity    import RiskParityOptimizer
from models.black_litterman import BlackLittermanOptimizer
from models.hybrid         import HybridOptimizer
from models.metrics        import PortfolioMetrics, compare_metrics
from models.ml             import load_prices, build_features, run_ols, run_xgboost, ml_weights

from plots.portfolio_plots import (plot_weights, plot_cumulative_returns,
                                   plot_drawdown, plot_correlation,
                                   plot_risk_contributions, plot_metrics_comparison)
from plots.ml_plots        import plot_ols, plot_xgb, plot_ml_allocation

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = {
    "LMT":       "Lockheed Martin",
    "AIR.PA":    "Airbus",
    "AM.PA":     "Dassault Aviation",
    "RHM.DE":    "Rheinmetall",
    "SAAB-B.ST": "Saab",
}
START = "2000-01-01"
RF    = 0.02

VIEWS = [
    {"assets": ["Rheinmetall", "Airbus"], "expected_return": 0.12, "confidence": 0.8},
    {"assets": ["Lockheed Martin"],       "expected_return": 0.07, "confidence": 0.6},
]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_returns() -> pd.DataFrame:
    print("\n[Data] Downloading prices ...")
    prices = yf.download(list(TICKERS.keys()), start=START, auto_adjust=True, progress=False)["Close"]
    prices = prices.rename(columns=TICKERS).dropna(how="all")
    print(f"  {prices.shape[0]} jours  |  {prices.index[0].date()} -> {prices.index[-1].date()}")
    return prices.pct_change(fill_method=None).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────

def run_strategies(returns: pd.DataFrame) -> dict:
    results = {}

    print("\n[1] Markowitz")
    mk = MarkowitzOptimizer(returns)
    results["Min Variance"]  = mk.min_variance()
    results["Max Sharpe"]    = mk.max_sharpe(rf=RF)

    print("\n[2] Risk Parity")
    rp = RiskParityOptimizer(returns, rf=RF)
    results["Risk Parity"] = rp.optimize()

    print("\n[3] Black-Litterman")
    bl = BlackLittermanOptimizer(returns, rf=RF)
    bl.add_views(VIEWS)
    results["Black-Litterman"] = bl.optimize()

    print("\n[4] Hybrid (RP + BL)")
    hy = HybridOptimizer(returns, rf=RF)
    hy.add_views(VIEWS)
    results["Hybrid"] = hy.optimize()

    for name, r in results.items():
        print(f"  {name:<20}  ret={r['return']*100:+.3f}%  vol={r['volatility']*100:.3f}%  sharpe={r.get('sharpe', 0):.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ML
# ─────────────────────────────────────────────────────────────────────────────

def run_ml() -> tuple[dict, dict, dict, dict]:
    print("\n[ML] Loading data & building features ...")
    prices   = load_prices(START)
    datasets = build_features(prices)

    print("\n[ML] OLS")
    ols_res = run_ols(datasets)

    print("\n[ML] XGBoost")
    xgb_res = run_xgboost(datasets)

    ols_preds = {n: r["last_pred"] for n, r in ols_res.items()}
    xgb_preds = {n: r["last_pred"] for n, r in xgb_res.items()}

    return ols_res, xgb_res, ml_weights(ols_preds), ml_weights(xgb_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    returns  = load_returns()
    results  = run_strategies(returns)

    print("\n[Metrics]")
    metrics = compare_metrics(results, returns)
    print(metrics.to_string())

    print("\n[Plots] Portfolio strategies ...")
    plot_weights(results)
    plot_cumulative_returns(returns, results)
    plot_drawdown(returns, results)
    plot_correlation(returns)
    plot_risk_contributions(results)
    plot_metrics_comparison(metrics)

    print("\n[ML]")
    ols_res, xgb_res, ols_w, xgb_w = run_ml()

    print("\n[Plots] ML ...")
    plot_ols(ols_res)
    plot_xgb(xgb_res)
    plot_ml_allocation(ols_w, xgb_w)

    print("\nDone.")


if __name__ == "__main__":
    main()
