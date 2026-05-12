"""
Portfolio Optimization — MaxDiv ML
====================================
Usage :
    python portfolio_optimization.py             # XGBoost + MaxDiv ML vs SPY
    python portfolio_optimization.py --backtest  # backtest walk-forward vs SPY

Optimisation :
    max   w'(sigma o mu) / sqrt(w' Sigma w)
    s.c.  sum(w) = 1
          w_i >= 10bp
          w'mu >= mu_bar   (contrainte alpha)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats.mstats import winsorize as _winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, colorize=True,
           format="<green>{time:HH:mm:ss}</green> │ <level>{level: <8}</level> │ {message}",
           level="INFO")
logger.add(OUT / "run.log",
           format="{time:YYYY-MM-DD HH:mm:ss} │ {level: <8} │ {message}",
           level="DEBUG", rotation="10 MB", encoding="utf-8")

def _section(title: str) -> None:
    bar = "─" * 58
    logger.info(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE_FILE = Path(r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx")
BENCHMARK     = "SPY"
RF            = 0.02
TRAIN_YEARS   = 8
MIN_WEIGHT    = 0.001
MAX_NAN_PCT   = 0.20

XGB_PARAMS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.1, random_state=42, n_jobs=-1, verbosity=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# DONNEES
# ─────────────────────────────────────────────────────────────────────────────

def load_universe() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Chargement : {UNIVERSE_FILE.name}")
    df = pd.read_excel(UNIVERSE_FILE, parse_dates=["Date"]).set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    benchmark = df[BENCHMARK].ffill() if BENCHMARK in df.columns else None
    raw       = df.drop(columns=[BENCHMARK], errors="ignore")

    excluded = raw.columns[raw.isna().mean() > MAX_NAN_PCT].tolist()
    if excluded:
        logger.warning(f"Exclus (>{MAX_NAN_PCT*100:.0f}% NaN) : {excluded}")

    prices = raw.drop(columns=excluded).ffill().dropna(how="all")
    prices = prices.dropna(thresh=int(len(prices.columns) * 0.80))

    logger.success(f"{prices.shape[1]} actifs  |  {prices.shape[0]} jours  |  "
                   f"{prices.index[0].date()} → {prices.index[-1].date()}")
    return prices, benchmark


# ─────────────────────────────────────────────────────────────────────────────
# METRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_metrics(r: pd.Series, rf: float = RF) -> dict:
    r       = r.dropna()
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std()  * np.sqrt(252))
    down    = r[r < rf / 252]
    dv      = float(np.sqrt(np.mean(down**2)) * np.sqrt(252)) if len(down) > 0 else 1e-10
    cum     = (1 + r).cumprod()
    dd      = float(((cum - cum.expanding().max()) / cum.expanding().max()).min())
    return {
        "Annual Return": ann_ret,
        "Annual Vol":    ann_vol,
        "Sharpe":        (ann_ret - rf) / ann_vol if ann_vol > 0 else 0,
        "Sortino":       (ann_ret - rf) / dv      if dv > 0      else 0,
        "Max Drawdown":  dd,
        "Calmar":        ann_ret / abs(dd)         if dd != 0     else 0,
    }


def w2r(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = np.array([weights.get(c, 0) for c in returns.columns])
    w /= w.sum() if w.sum() > 0 else 1
    return (returns * w).sum(axis=1)


def _log_metrics(m: dict, name: str) -> None:
    logger.info(f"  {name:<18}  ret={m['Annual Return']*100:>+6.2f}%  "
                f"vol={m['Annual Vol']*100:>5.2f}%  sharpe={m['Sharpe']:>6.3f}  "
                f"maxdd={m['Max Drawdown']*100:>6.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def _features(r: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=r.index)
    X["ret_1m"]  = r.rolling(21).sum().shift(1)
    X["ret_3m"]  = r.rolling(63).sum().shift(1)
    X["ret_6m"]  = r.rolling(126).sum().shift(1)
    X["ret_12m"] = r.rolling(252).sum().shift(1)
    X["vol_1m"]  = r.rolling(21).std().shift(1)  * np.sqrt(252)
    X["vol_3m"]  = r.rolling(63).std().shift(1)  * np.sqrt(252)
    X["vol_6m"]  = r.rolling(126).std().shift(1) * np.sqrt(252)
    X["mom_1m"]  = r.shift(1).rolling(21).mean()
    X["mom_3m"]  = r.shift(1).rolling(63).mean()
    X["skew_63"] = r.shift(1).rolling(63).skew()
    X["kurt_63"] = r.shift(1).rolling(63).kurt()
    return X


def xgb_predict(prices: pd.DataFrame, train_years: int = TRAIN_YEARS) -> dict:
    """
    Entraine XGBoost par actif sur `train_years` ans.
    Retourne : {actif: {pred_annual, mu_norm, r2_oos, importance, ...}}
    """
    returns = prices.pct_change(fill_method=None).dropna()
    n_train = train_years * 252
    t_end   = len(returns) - 1
    t_start = max(0, t_end - n_train)

    logger.info(f"  Fenetre : {returns.index[t_start].date()} → {returns.index[t_end].date()}")

    results = {}
    for col in returns.columns:
        r  = returns[col].dropna()
        X  = _features(r)
        y  = r.rolling(21).sum().shift(-21).rename("target")
        df = X.join(y).dropna()

        tr = (df.index >= returns.index[t_start]) & (df.index <= returns.index[t_end])
        te = df.index > returns.index[t_end]
        X_tr, y_tr = df.loc[tr].drop(columns="target"), df.loc[tr]["target"]
        X_te, y_te = df.loc[te].drop(columns="target"), df.loc[te]["target"]

        if len(X_tr) < 200:
            logger.warning(f"  {col:<20} ignore (train={len(X_tr)} obs)")
            continue

        sc     = StandardScaler()
        Xtr_sc = sc.fit_transform(X_tr)
        Xte_sc = sc.transform(X_te) if len(X_te) > 0 else np.empty((0, Xtr_sc.shape[1]))

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr_sc, y_tr.values)

        r2_oos = float("nan")
        if len(X_te) > 10:
            r2_oos = float(r2_score(y_te.values, model.predict(Xte_sc)))

        pred_annual = float(model.predict(sc.transform(X.iloc[[-1]]))[0]) * 12

        X_ols = sm.add_constant(X_tr, has_constant="add")
        ols   = sm.OLS(y_tr, X_ols).fit(cov_type="HC3")

        results[col] = {
            "pred_annual": pred_annual,
            "r2_oos":      r2_oos,
            "ols_r2":      ols.rsquared,
            "ols_fpval":   ols.f_pvalue,
            "ols_pvalues": ols.pvalues.drop("const", errors="ignore"),
            "importance":  pd.Series(model.feature_importances_,
                                     index=X_tr.columns).sort_values(ascending=False),
        }

    raw  = pd.Series({k: v["pred_annual"] for k, v in results.items()})
    wins = pd.Series(np.array(_winsorize(raw.values, limits=[0.05, 0.05])), index=raw.index)
    mn, mx = wins.min(), wins.max()
    mu_n = (wins - mn) / (mx - mn) * 0.9 + 0.1 if mx > mn else pd.Series(0.55, index=wins.index)
    for k in results:
        results[k]["mu_norm"] = float(mu_n[k])

    logger.success(f"XGBoost : {len(results)} actifs traites")
    return results


def ranking_df(xgb_res: dict) -> pd.DataFrame:
    raw = pd.Series({k: v["pred_annual"] for k, v in xgb_res.items()})
    mu  = pd.Series({k: v["mu_norm"]     for k, v in xgb_res.items()})
    df  = pd.DataFrame({"predicted_annual": raw, "mu_normalized": mu,
                        "rank": raw.rank(ascending=False).astype(int)}).sort_values("rank")
    df.index.name = "asset"
    return df


def _log_ranking(rk: pd.DataFrame) -> None:
    logger.info(f"  {'#':<4}  {'Actif':<20}  {'Pred/an':>9}  {'mu':>7}")
    logger.info("  " + "─" * 45)
    for asset, row in rk.iterrows():
        sym = "▲" if row["predicted_annual"] >= 0 else "▼"
        logger.info(f"  {int(row['rank']):<4}  {asset:<20}  "
                    f"{sym} {row['predicted_annual']*100:>+6.2f}%  {row['mu_normalized']:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def _solve(obj_fn, n: int, constraints: list) -> np.ndarray:
    bounds   = [(MIN_WEIGHT, 1.0)] * n
    best_val = np.inf
    best_w   = None
    for seed in [42, 123, 456, 789, 1337]:
        rng = np.random.default_rng(seed)
        w0  = np.clip(rng.dirichlet(np.ones(n)), MIN_WEIGHT, 1.0)
        w0 /= w0.sum()
        res = minimize(obj_fn, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"ftol": 1e-10, "maxiter": 2000})
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x.copy()
    if best_w is None:
        best_w = np.ones(n) / n
    best_w = np.clip(best_w, MIN_WEIGHT, 1.0)
    best_w /= best_w.sum()
    return best_w


def maxdiv_ml(returns: pd.DataFrame, xgb_res: dict) -> dict:
    """
    max   w'(sigma o mu) / sqrt(w' Sigma w)
    s.c.  sum(w) = 1
          w_i >= 10bp
          w'mu >= mu_bar
    """
    names   = returns.columns.tolist()
    n       = len(names)
    sigma   = returns.std().values  * np.sqrt(252)
    Sigma   = returns.cov().values  * 252
    mu_hist = returns.mean().values * 252

    mu_s   = pd.Series({k: xgb_res[k]["mu_norm"] for k in xgb_res if k in names})
    mu     = mu_s.reindex(names).fillna(mu_s.median()).values
    mu_bar = float(mu.mean())

    def objective(w):
        num = float(w @ (sigma * mu))
        den = float(np.sqrt(w @ Sigma @ w))
        return -num / den if den > 1e-12 else 1e10

    w = _solve(objective, n, [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: float(w @ mu) - mu_bar},
    ])

    vol = float(np.sqrt(w @ Sigma @ w))
    ret = float(mu_hist @ w)
    return {
        "weights":    dict(zip(names, w)),
        "return":     ret,
        "volatility": vol,
        "sharpe":     (ret - RF) / vol if vol > 0 else 0,
        "dr":         float((w @ (sigma * mu)) / (vol + 1e-12)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame,
                 benchmark: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train   = TRAIN_YEARS * 252
    dates     = prices.index
    rebal_idx = list(range(n_train, len(dates), 252))

    logger.info(f"  {len(rebal_idx)} rebalancements  "
                f"({dates[n_train].date()} → {dates[-1].date()})")

    ml_rets, all_dates = [], []

    for i, idx in enumerate(rebal_idx):
        t_start = max(0, idx - n_train)
        t_end   = min(len(dates) - 1, idx + 252)
        r_train = prices.iloc[t_start:idx].pct_change(fill_method=None).dropna()
        r_test  = prices.iloc[idx:t_end].pct_change(fill_method=None).dropna()

        if len(r_test) == 0:
            continue

        logger.info(f"  [{i+1:>2}/{len(rebal_idx)}]  "
                    f"train {dates[t_start].date()} → {dates[idx-1].date()}  │  "
                    f"test  {dates[idx].date()} → {dates[t_end-1].date()}")

        try:
            xr = xgb_predict(prices.iloc[t_start:idx], train_years=TRAIN_YEARS)
            w  = maxdiv_ml(r_train, xr)["weights"]
        except Exception as e:
            logger.warning(f"    MaxDiv ML echec : {e}")
            names = r_train.columns.tolist()
            w = dict(zip(names, np.ones(len(names)) / len(names)))

        ww = np.array([w.get(c, 0) for c in r_test.columns])
        ww /= ww.sum() if ww.sum() > 0 else 1
        ml_rets.extend((r_test * ww).sum(axis=1).tolist())
        all_dates.extend(r_test.index.tolist())

    rets_df = pd.DataFrame({"MaxDiv ML": ml_rets},
                            index=pd.DatetimeIndex(all_dates[:len(ml_rets)]))

    if benchmark is not None:
        rets_df[BENCHMARK] = (benchmark.pct_change(fill_method=None)
                                        .reindex(rets_df.index).fillna(0))

    metrics_df = pd.DataFrame(
        [{**portfolio_metrics(rets_df[s]), "Strategy": s} for s in rets_df.columns]
    ).set_index("Strategy")

    return rets_df, metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PALETTE = {"MaxDiv ML": "#FF6B35", BENCHMARK: "#000000"}
plt.style.use("seaborn-v0_8-darkgrid")


def _save(name: str) -> None:
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.debug(f"Graphique -> {path.name}")
    plt.close()


def plot_weights(weights: dict, fname: str = "weights.png") -> None:
    assets = list(weights.keys())
    vals   = [weights[a] * 100 for a in assets]
    colors = ["#55A868" if v >= 100 / len(assets) else "#888888" for v in vals]

    fig, ax = plt.subplots(figsize=(max(12, len(assets) * 0.35), 5))
    ax.bar(assets, vals, color=colors, edgecolor="white")
    ax.axhline(100 / len(assets), color="#FF6B35", linewidth=1.5, linestyle="--",
               label=f"Equal weight ({100/len(assets):.1f}%)")
    ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("Poids (%)")
    ax.set_title("MaxDiv ML — Allocation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); _save(fname)


def plot_cumulative(returns_df: pd.DataFrame, title: str, fname: str) -> None:
    cum = (1 + returns_df).cumprod()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for col in cum.columns:
        lw = 2.5 if col == "MaxDiv ML" else 1.8
        ls = "--" if col == BENCHMARK else "-"
        axes[0].plot(cum.index, cum[col], label=col, linewidth=lw, linestyle=ls,
                     color=PALETTE.get(col, "gray"))
    axes[0].set_ylabel("Valeur (base 1)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    axes[0].legend(fontsize=10)

    for col in returns_df.columns:
        c  = (1 + returns_df[col]).cumprod()
        dd = (c - c.expanding().max()) / c.expanding().max()
        axes[1].plot(dd.index, dd * 100, label=col, linewidth=2.5 if col == "MaxDiv ML" else 1.8,
                     linestyle="--" if col == BENCHMARK else "-",
                     color=PALETTE.get(col, "gray"))
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[1].legend(fontsize=10)

    plt.tight_layout(); _save(fname)


def plot_ranking(rk: pd.DataFrame, fname: str = "xgb_ranking.png") -> None:
    vals   = rk["predicted_annual"].values * 100
    names  = rk.index.tolist()
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(5, len(rk) * 0.35 + 1)))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="white", lw=1, linestyle="--")
    ax.set_xlabel("Rendement annuel predit (%)")
    ax.set_title("XGBoost — Classement des actifs", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + np.sign(v) * 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", fontsize=6)
    plt.tight_layout(); _save(fname)


def plot_metrics_table(metrics_df: pd.DataFrame, fname: str = "metrics.png") -> None:
    cols  = {"Annual Return": True, "Annual Vol": False, "Sharpe": True,
             "Sortino": True, "Max Drawdown": False, "Calmar": True}
    avail = [c for c in cols if c in metrics_df.columns]
    sub   = metrics_df[avail]
    norm  = pd.DataFrame(index=sub.index, columns=sub.columns, dtype=float)
    for col, up in cols.items():
        if col not in sub.columns: continue
        mn, mx = sub[col].min(), sub[col].max()
        v = (sub[col] - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=sub.index)
        norm[col] = v if up else (1 - v)

    fig, ax = plt.subplots(figsize=(11, max(3, len(sub) * 0.8 + 1.5)))
    fig.suptitle("Performance — MaxDiv ML vs SPY", fontsize=12, fontweight="bold")
    im = ax.imshow(norm.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(avail)))
    ax.set_xticklabels(avail, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=10)
    for i in range(len(sub)):
        for j, col in enumerate(avail):
            v   = sub.iloc[i][col]
            fmt = f"{v*100:.1f}%" if any(x in col for x in ["Return","Vol","Drawdown"]) else f"{v:.3f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout(); _save(fname)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Backtest walk-forward vs SPY")
    args = parser.parse_args()

    logger.info("Portfolio Optimization — MaxDiv ML")
    prices, benchmark = load_universe()
    returns = prices.pct_change(fill_method=None).dropna()

    if args.backtest:
        _section(f"BACKTEST  ({TRAIN_YEARS} ans train / 1 an test)")
        rets_df, met = run_backtest(prices, benchmark)
        for s in met.index:
            _log_metrics(met.loc[s].to_dict(), s)
        plot_cumulative(rets_df, "Backtest Walk-Forward — MaxDiv ML vs SPY", "backtest_cumulative.png")
        plot_metrics_table(met, "backtest_metrics.png")

    else:
        _section(f"XGBOOST + MAXDIV ML  ({TRAIN_YEARS} ans)")
        xgb_res = xgb_predict(prices)
        rk      = ranking_df(xgb_res)
        _log_ranking(rk)

        _section("OPTIMISATION")
        result = maxdiv_ml(returns, xgb_res)
        _log_metrics(portfolio_metrics(w2r(returns, result["weights"])), "MaxDiv ML")

        spy_r = (benchmark.pct_change(fill_method=None).reindex(returns.index).fillna(0)
                 if benchmark is not None else None)
        rows = [{"Strategy": "MaxDiv ML", **portfolio_metrics(w2r(returns, result["weights"]))}]
        if spy_r is not None:
            rows.append({"Strategy": BENCHMARK, **portfolio_metrics(spy_r)})
            _log_metrics(portfolio_metrics(spy_r), BENCHMARK)
        met_df  = pd.DataFrame(rows).set_index("Strategy")

        rets_df = pd.DataFrame({"MaxDiv ML": w2r(returns, result["weights"])})
        if spy_r is not None:
            rets_df[BENCHMARK] = spy_r

        plot_ranking(rk)
        plot_weights(result["weights"])
        plot_cumulative(rets_df, "MaxDiv ML vs SPY", "cumulative.png")
        plot_metrics_table(met_df)

    logger.success(f"Outputs -> {OUT}")


if __name__ == "__main__":
    main()
