"""
Machine Learning Portfolio Optimization — Defence Basket
=========================================================
Block 1 : Data & Features
Block 2 : OLS Linear Regression  (statistiques interprétables)
Block 3 : XGBoost                (non-linéaire, feature importance)
Block 4 : ML-Driven Portfolio Weights
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = {
    "LMT":       "Lockheed Martin",
    "AIR.PA":    "Airbus",
    "AM.PA":     "Dassault Aviation",
    "RHM.DE":    "Rheinmetall",
    "SAAB-B.ST": "Saab",
}

START = "2000-01-01"

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — Data & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    print(f"Downloading {len(TICKERS)} tickers from {START} ...")
    raw = yf.download(
        tickers     = list(TICKERS.keys()),
        start       = START,
        auto_adjust = True,
        progress    = False,
    )["Close"]
    raw = raw.rename(columns=TICKERS)
    raw = raw.dropna(how="all")
    print(f"  {raw.shape[0]} jours  |  {raw.index[0].date()} -> {raw.index[-1].date()}")
    return raw


def build_features(prices: pd.DataFrame) -> dict:
    """
    Features sur rendements journaliers :
      ret_1  : rendement J-1
      ret_5  : moyenne 5j
      ret_10 : moyenne 10j
      ret_21 : moyenne 21j
      vol_5  : volatilite realisee 5j
      vol_21 : volatilite realisee 21j
      mom_63 : momentum 63j (trimestre)

    Target : rendement J+1
    """
    returns  = prices.pct_change(fill_method=None)
    datasets = {}

    for col in returns.columns:
        r = returns[col].dropna()

        X = pd.DataFrame(index=r.index)
        X["ret_1"]  = r.shift(1)
        X["ret_5"]  = r.shift(1).rolling(5).mean()
        X["ret_10"] = r.shift(1).rolling(10).mean()
        X["ret_21"] = r.shift(1).rolling(21).mean()
        X["vol_5"]  = r.shift(1).rolling(5).std()
        X["vol_21"] = r.shift(1).rolling(21).std()
        X["mom_63"] = r.shift(1).rolling(63).mean()

        y  = r.shift(-1)
        df = X.join(y.rename("target")).dropna()

        datasets[col] = (df.drop(columns="target"), df["target"])
        print(f"  {col:<25}  {len(df)} observations")

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — OLS Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

def run_ols(datasets: dict) -> dict:
    """OLS statsmodels, erreurs robustes HC3."""
    results = {}

    for name, (X, y) in datasets.items():
        X_const = sm.add_constant(X)
        model   = sm.OLS(y, X_const).fit(cov_type="HC3")

        results[name] = {
            "r2":        model.rsquared,
            "r2_adj":    model.rsquared_adj,
            "f_stat":    model.fvalue,
            "f_pval":    model.f_pvalue,
            "n_obs":     int(model.nobs),
            "params":    model.params.drop("const"),
            "tvalues":   model.tvalues.drop("const"),
            "pvalues":   model.pvalues.drop("const"),
            "last_pred": float(model.predict(X_const.iloc[[-1]])),
        }

        sig = "[sig]" if model.f_pvalue < 0.05 else "[ns] "
        print(f"  {name:<25}  R2={model.rsquared:.4f}  F-pval={model.f_pvalue:.4f} {sig}  n={int(model.nobs)}")

    return results


def plot_ols(ols: dict) -> None:
    names    = list(ols.keys())
    features = list(ols[names[0]]["params"].index)
    n        = len(names)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle("OLS Linear Regression — Defence Basket", fontsize=13, fontweight="bold")

    for i, (name, r) in enumerate(ols.items()):
        # -- Coefficients avec intervalles de confiance (approx ±2*SE)
        ax = axes[0, i]
        coefs  = r["params"].values
        pvals  = r["pvalues"].values
        colors = ["#55A868" if p < 0.05 else "#888888" for p in pvals]
        bars   = ax.barh(features, coefs, color=colors)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}\nR²={r['r2']:.4f}  F-pval={r['f_pval']:.4f}", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        if i == 0:
            ax.set_ylabel("Feature")

        # -- p-values
        ax2 = axes[1, i]
        bar_colors = ["#55A868" if p < 0.05 else "#C44E52" for p in pvals]
        ax2.bar(features, pvals, color=bar_colors)
        ax2.axhline(0.05, color="yellow", linewidth=1, linestyle="--", label="seuil 5%")
        ax2.set_ylim(0, 1)
        ax2.set_title("p-values", fontsize=9)
        ax2.tick_params(axis="x", rotation=45, labelsize=7)
        if i == 0:
            ax2.set_ylabel("p-value")
            ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT / "ols_results.png", dpi=150, bbox_inches="tight")
    print("  Saved -> ols_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost(datasets: dict, n_splits: int = 5) -> dict:
    """XGBoost avec TimeSeriesSplit cross-validation."""
    results = {}
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    params  = dict(
        n_estimators     = 300,
        max_depth        = 3,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
    )

    for name, (X, y) in datasets.items():
        oof_preds = np.full(len(y), np.nan)

        for train_idx, val_idx in tscv.split(X):
            m = xgb.XGBRegressor(**params, verbosity=0)
            m.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
            oof_preds[val_idx] = m.predict(X.iloc[val_idx])

        valid = ~np.isnan(oof_preds)
        r2    = r2_score(y.values[valid], oof_preds[valid])
        rmse  = np.sqrt(mean_squared_error(y.values[valid], oof_preds[valid]))

        final = xgb.XGBRegressor(**params, verbosity=0)
        final.fit(X, y)

        importance = pd.Series(
            final.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        results[name] = {
            "r2":        r2,
            "rmse":      rmse,
            "importance": importance,
            "last_pred": float(final.predict(X.iloc[[-1]])),
        }

        print(f"  {name:<25}  OOF R2={r2:.4f}  RMSE={rmse:.6f}")

    return results


def plot_xgb(xgb_res: dict) -> None:
    names    = list(xgb_res.keys())
    features = list(xgb_res[names[0]]["importance"].index)
    n        = len(names)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("XGBoost — Feature Importance par actif", fontsize=13, fontweight="bold")

    for i, (name, r) in enumerate(xgb_res.items()):
        imp = r["importance"]
        axes[i].barh(
            imp.index[::-1], imp.values[::-1],
            color=COLORS[i % len(COLORS)]
        )
        axes[i].set_title(
            f"{name}\nOOF R2={r['r2']:.4f}",
            fontsize=9
        )
        axes[i].tick_params(axis="y", labelsize=8)
        if i == 0:
            axes[i].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(OUT / "xgb_results.png", dpi=150, bbox_inches="tight")
    print("  Saved -> xgb_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — ML Portfolio Weights
# ─────────────────────────────────────────────────────────────────────────────

def ml_weights(predictions: dict) -> dict:
    """Poids proportionnels aux rendements predits positifs. Fallback equal weight."""
    clipped = {k: max(v, 0.0) for k, v in predictions.items()}
    total   = sum(clipped.values())
    if total == 0:
        n = len(predictions)
        return {k: 1 / n for k in predictions}
    return {k: v / total for k, v in clipped.items()}


def plot_allocation(ols_w: dict, xgb_w: dict) -> None:
    names = list(ols_w.keys())
    eq    = [1 / len(names)] * len(names)
    x     = np.arange(len(names))
    w     = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,     eq,                       width=w, label="Equal Weight", color="#888888")
    ax.bar(x,         [ols_w[n] for n in names], width=w, label="OLS",         color=COLORS[0])
    ax.bar(x + w,     [xgb_w[n] for n in names], width=w, label="XGBoost",     color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Poids")
    ax.set_title("Allocation ML vs Equal Weight", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / "ml_allocation.png", dpi=150, bbox_inches="tight")
    print("  Saved -> ml_allocation.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("\n[Block 1] Data & Features")
    prices   = load_prices()
    datasets = build_features(prices)

    print("\n[Block 2] OLS Linear Regression")
    ols_res = run_ols(datasets)
    plot_ols(ols_res)

    print("\n[Block 3] XGBoost (5-fold TimeSeriesCV)")
    xgb_res = run_xgboost(datasets)
    plot_xgb(xgb_res)

    print("\n[Block 4] ML Portfolio Weights")
    ols_preds = {n: r["last_pred"] for n, r in ols_res.items()}
    xgb_preds = {n: r["last_pred"] for n, r in xgb_res.items()}
    ols_w     = ml_weights(ols_preds)
    xgb_w     = ml_weights(xgb_preds)

    print("  OLS predictions :")
    for n, v in ols_preds.items():
        print(f"    {n:<25}  {v*100:+.4f}%  ->  poids {ols_w[n]*100:.1f}%")

    print("  XGB predictions :")
    for n, v in xgb_preds.items():
        print(f"    {n:<25}  {v*100:+.4f}%  ->  poids {xgb_w[n]*100:.1f}%")

    plot_allocation(ols_w, xgb_w)
    print("\nDone.")


if __name__ == "__main__":
    run()
