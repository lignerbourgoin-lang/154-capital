"""
2 — Risk Parity + Black-Litterman
==================================
- Risk Parity : égalise les contributions au risque
- Black-Litterman : intègre des vues subjectives sur les rendements

Usage : python 2_risk_parity_bl.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from common import setup_logging, load_universe, log_results, OUT, RF
from models.risk_parity import RiskParityOptimizer
from models.black_litterman import BlackLittermanOptimizer
from models.metrics import compare_metrics

setup_logging()
plt.style.use("seaborn-v0_8-darkgrid")

PALETTE = {
    "Risk Parity":    "#937860",
    "Black-Litterman": "#64B5CD",
    "SPY":            "#000000",
}

VIEWS = [
    {"assets": ["NVDA"], "expected_return": 0.20, "confidence": 0.7},
    {"assets": ["MSFT"], "expected_return": 0.12, "confidence": 0.6},
]


def _save(name):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.debug(f"Graphique -> {path.name}")
    plt.close()


def plot_risk_contributions(rp_result, title="Risk Parity — Contributions au risque"):
    rc   = rp_result["rc_pct"]
    assets = list(rc.keys())
    vals   = list(rc.values())
    eq     = 100 / len(assets)

    fig, ax = plt.subplots(figsize=(max(12, len(assets) * 0.4), 5))
    colors = ["#55A868" if abs(v - eq) < 2 else "#937860" for v in vals]
    ax.bar(assets, vals, color=colors, edgecolor="white")
    ax.axhline(eq, color="#FF6B35", linewidth=1.5, linestyle="--",
               label=f"Cible ({eq:.1f}%)")
    ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Contribution au risque (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("2_risk_contributions.png")


def plot_weights(results):
    strats = list(results.keys())
    assets = list(results[strats[0]]["weights"].keys())
    x  = np.arange(len(assets))
    bw = 0.8 / len(strats)

    fig, ax = plt.subplots(figsize=(max(12, len(assets) * 0.4), 5))
    for j, s in enumerate(strats):
        w = [results[s]["weights"].get(a, 0) * 100 for a in assets]
        ax.bar(x + j * bw, w, width=bw, label=s,
               color=PALETTE.get(s, f"C{j}"), edgecolor="white")
    ax.set_xticks(x + bw * (len(strats) - 1) / 2)
    ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Poids (%)")
    ax.set_title("Risk Parity vs Black-Litterman — Allocation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("2_weights.png")


def plot_bl_mu(bl_result, title="Black-Litterman — Rendements implicites vs posterieurs"):
    assets  = list(bl_result["implied_mu"].keys())
    implied = [bl_result["implied_mu"][a] * 252 * 100 for a in assets]
    post    = [bl_result["posterior_mu"][a] * 252 * 100 for a in assets]
    hist    = [bl_result["historical_mu"][a] * 252 * 100 for a in assets]

    x  = np.arange(len(assets))
    bw = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(assets) * 0.4), 5))
    ax.bar(x - bw, hist,    width=bw, label="Historique", color="#888888", edgecolor="white")
    ax.bar(x,      implied, width=bw, label="Implicite",  color="#64B5CD", edgecolor="white")
    ax.bar(x + bw, post,    width=bw, label="Posterieur", color="#FF6B35", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Rendement annualisé (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("2_bl_mu.png")


def plot_cumulative(returns, results, benchmark):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Risk Parity vs Black-Litterman — Performance", fontsize=13, fontweight="bold")

    for name, r in results.items():
        w  = np.array([r["weights"].get(c, 0) for c in returns.columns])
        w /= w.sum()
        ret = (returns * w).sum(axis=1)
        cum = (1 + ret).cumprod()
        dd  = (cum - cum.expanding().max()) / cum.expanding().max()
        axes[0].plot(cum.index, cum, label=name, linewidth=2, color=PALETTE.get(name))
        axes[1].plot(dd.index, dd * 100, label=name, linewidth=2, color=PALETTE.get(name))

    if benchmark is not None:
        spy = benchmark.pct_change(fill_method=None).reindex(returns.index).fillna(0)
        cum_spy = (1 + spy).cumprod()
        dd_spy  = (cum_spy - cum_spy.expanding().max()) / cum_spy.expanding().max()
        axes[0].plot(cum_spy.index, cum_spy, label="SPY", linewidth=1.5,
                     linestyle="--", color=PALETTE["SPY"])
        axes[1].plot(dd_spy.index, dd_spy * 100, label="SPY", linewidth=1.5,
                     linestyle="--", color=PALETTE["SPY"])

    axes[0].set_ylabel("Valeur (base 1)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    _save("2_cumulative.png")


def main():
    logger.info("── 2. RISK PARITY + BLACK-LITTERMAN ─────────────────────")
    prices, benchmark = load_universe()
    returns = prices.pct_change(fill_method=None).dropna()

    logger.info("Risk Parity ...")
    rp        = RiskParityOptimizer(returns, rf=RF)
    rp_result = rp.optimize()

    logger.info("Black-Litterman ...")
    bl = BlackLittermanOptimizer(returns, rf=RF)
    bl.add_views(VIEWS)
    bl_result = bl.optimize()

    results = {"Risk Parity": rp_result, "Black-Litterman": bl_result}
    met = compare_metrics(results, returns)
    log_results(results)

    logger.info("Métriques :")
    logger.info(f"\n{met[['Annual Return','Annual Vol','Sharpe','Max Drawdown']].to_string()}")

    plot_risk_contributions(rp_result)
    plot_bl_mu(bl_result)
    plot_weights(results)
    plot_cumulative(returns, results, benchmark)

    logger.success(f"Outputs -> {OUT}")


if __name__ == "__main__":
    main()
