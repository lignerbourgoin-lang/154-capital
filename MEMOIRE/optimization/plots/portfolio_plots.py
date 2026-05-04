import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "outputs"
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _save(name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.show()


def plot_weights(results: dict) -> None:
    """Pie charts des poids pour chaque strategie."""
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        w      = res["weights"]
        labels = [k for k, v in w.items() if v > 0.001]
        values = [v for v in w.values() if v > 0.001]
        ax.pie(values, labels=labels, autopct="%1.1f%%",
               colors=COLORS[:len(labels)], startangle=90)
        ax.set_title(name, fontweight="bold")

    plt.suptitle("Allocation par strategie", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("portfolio_weights.png")


def plot_cumulative_returns(returns: pd.DataFrame, results: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, res) in enumerate(results.items()):
        w    = np.array([res["weights"].get(c, 0) for c in returns.columns])
        port = (returns * w).sum(axis=1)
        cum  = (1 + port).cumprod()
        ax.plot(cum.index, (cum - 1) * 100, label=name, color=COLORS[i % len(COLORS)], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Rendement cumule (%)")
    ax.set_title("Rendements cumules", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save("cumulative_returns.png")


def plot_drawdown(returns: pd.DataFrame, results: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (name, res) in enumerate(results.items()):
        w    = np.array([res["weights"].get(c, 0) for c in returns.columns])
        port = (returns * w).sum(axis=1)
        cum  = (1 + port).cumprod()
        dd   = (cum - cum.expanding().max()) / cum.expanding().max() * 100
        ax.fill_between(dd.index, dd, 0, alpha=0.25, color=COLORS[i % len(COLORS)])
        ax.plot(dd.index, dd, color=COLORS[i % len(COLORS)], linewidth=1.5, label=name)

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save("drawdown.png")


def plot_correlation(returns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Matrice de correlation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("correlation.png")


def plot_risk_contributions(results: dict) -> None:
    """Bar chart des contributions au risque (pour strategies avec rc_pct)."""
    has_rc = {n: r for n, r in results.items() if "rc_pct" in r}
    if not has_rc:
        return

    n   = len(has_rc)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, has_rc.items()):
        rc     = res["rc_pct"]
        target = 100 / len(rc)
        colors = ["#55A868" if abs(v - target) < 5 else "#C44E52" for v in rc.values()]
        ax.bar(rc.keys(), rc.values(), color=colors)
        ax.axhline(target, color="gold", linestyle="--", linewidth=1.5, label=f"Target {target:.1f}%")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("RC (%)")
        ax.legend(fontsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)

    plt.suptitle("Contribution au risque", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("risk_contributions.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """Heatmap des metriques comparees entre strategies."""
    display_cols = ["Annual Return", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Calmar"]
    df = metrics_df[[c for c in display_cols if c in metrics_df.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.T, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar=False)
    ax.set_title("Comparaison des metriques", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("metrics_comparison.png")
