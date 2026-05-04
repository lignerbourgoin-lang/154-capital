import numpy as np
import matplotlib.pyplot as plt
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


def plot_ols(ols: dict) -> None:
    """Coefficients OLS + p-values par actif."""
    names    = list(ols.keys())
    features = list(ols[names[0]]["params"].index)
    n        = len(names)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("OLS — Regression Lineaire", fontsize=13, fontweight="bold")

    for i, (name, r) in enumerate(ols.items()):
        coefs  = r["params"].values
        pvals  = r["pvalues"].values

        # Coefficients
        ax = axes[0, i]
        ax.barh(features, coefs,
                color=["#55A868" if p < 0.05 else "#888888" for p in pvals])
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}\nR2={r['r2']:.4f}  F-pval={r['f_pval']:.4f}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

        # p-values
        ax2 = axes[1, i]
        ax2.bar(features, pvals,
                color=["#55A868" if p < 0.05 else "#C44E52" for p in pvals])
        ax2.axhline(0.05, color="yellow", linewidth=1, linestyle="--", label="5%")
        ax2.set_ylim(0, 1)
        ax2.set_title("p-values", fontsize=9)
        ax2.tick_params(axis="x", rotation=45, labelsize=7)
        if i == 0:
            ax2.legend(fontsize=7)

    plt.tight_layout()
    _save("ols_results.png")


def plot_xgb(xgb_res: dict) -> None:
    """Feature importance XGBoost par actif."""
    n   = len(xgb_res)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("XGBoost — Feature Importance", fontsize=13, fontweight="bold")

    for i, (name, r) in enumerate(xgb_res.items()):
        imp = r["importance"]
        axes[i].barh(imp.index[::-1], imp.values[::-1], color=COLORS[i % len(COLORS)])
        axes[i].set_title(f"{name}\nOOF R2={r['r2']:.4f}", fontsize=9)
        axes[i].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    _save("xgb_results.png")


def plot_ml_allocation(ols_w: dict, xgb_w: dict) -> None:
    """Poids OLS vs XGBoost vs Equal Weight."""
    names = list(ols_w.keys())
    eq    = [1 / len(names)] * len(names)
    x     = np.arange(len(names))
    w     = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, eq,                        width=w, label="Equal Weight", color="#888888")
    ax.bar(x,     [ols_w[n] for n in names], width=w, label="OLS",          color=COLORS[0])
    ax.bar(x + w, [xgb_w[n] for n in names], width=w, label="XGBoost",      color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Poids")
    ax.set_title("Allocation ML vs Equal Weight", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend()

    plt.tight_layout()
    _save("ml_allocation.png")
