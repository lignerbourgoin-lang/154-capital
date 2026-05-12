"""
1 — Markowitz
=============
- Minimum Variance
- Maximum Sharpe
- Frontière efficiente

Usage : python 1_markowitz.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from common import setup_logging, load_universe, log_results, OUT, RF
from models.markowitz import MarkowitzOptimizer
from models.metrics import compare_metrics

setup_logging()
plt.style.use("seaborn-v0_8-darkgrid")

PALETTE = {
    "Min Variance": "#C44E52",
    "Max Sharpe":   "#8172B2",
    "SPY":          "#000000",
}


def _save(name):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.debug(f"Graphique -> {path.name}")
    plt.close()


def plot_frontier(mk, results):
    frontier = mk.efficient_frontier(n_points=80)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([v * np.sqrt(252) * 100 for v in frontier["volatilities"]],
            [r * 252 * 100 for r in frontier["returns"]],
            color="#4C72B0", linewidth=2, label="Frontière efficiente")

    for name, r in results.items():
        ax.scatter(r["volatility"] * np.sqrt(252) * 100,
                   r["return"] * 252 * 100,
                   s=120, zorder=5, color=PALETTE.get(name, "gray"),
                   label=name, edgecolors="white", linewidths=1.5)

    ax.set_xlabel("Volatilité annualisée (%)")
    ax.set_ylabel("Rendement annualisé (%)")
    ax.set_title("Markowitz — Frontière efficiente", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("1_frontier.png")


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
    ax.set_title("Markowitz — Allocation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("1_weights.png")


def plot_cumulative(returns, results, benchmark):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Markowitz — Performance historique", fontsize=13, fontweight="bold")

    for name, r in results.items():
        w   = np.array([r["weights"].get(c, 0) for c in returns.columns])
        w  /= w.sum()
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
    _save("1_cumulative.png")


def main():
    logger.info("── 1. MARKOWITZ ──────────────────────────────────────────")
    prices, benchmark = load_universe()
    returns = prices.pct_change(fill_method=None).dropna()

    mk = MarkowitzOptimizer(returns)
    results = {
        "Min Variance": mk.min_variance(),
        "Max Sharpe":   mk.max_sharpe(rf=RF),
    }

    met = compare_metrics(results, returns)
    log_results(results)

    logger.info("Métriques :")
    logger.info(f"\n{met[['Annual Return','Annual Vol','Sharpe','Max Drawdown']].to_string()}")

    plot_frontier(mk, results)
    plot_weights(results)
    plot_cumulative(returns, results, benchmark)

    logger.success(f"Outputs -> {OUT}")


if __name__ == "__main__":
    main()
