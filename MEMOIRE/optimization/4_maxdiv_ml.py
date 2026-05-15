"""
Portfolio Optimization — MaxDiv ML
====================================
Usage :
    python 4_maxdiv_ml.py             # XGBoost + MaxDiv ML vs SPY
    python 4_maxdiv_ml.py --backtest  # backtest walk-forward vs SPY

Optimisation :
    max   w'(σ ⊙ μ)
    s.c.  w'Σ_factor w  ≤  VOL_TARGET²   (budget de variance)
          sum(w) = 1,  w_i ≥ 0
          ||w - w_prev||₁ ≤ TURNOVER_MAX  (contrainte turnover)
          sum(w_i pour i ∈ S) ≤ MAX_SECTOR_WEIGHT  ∀ S

Covariance :
    Σ = B Σ_f B' + Σ_ε  (factor model via PCA)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize as _winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger
from collections import defaultdict
from datetime import datetime

try:
    import cvxpy as cp
    _MOSEK_OK = True
except ImportError:
    _MOSEK_OK = False

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

UNIVERSE_FILE    = Path(r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx")
BENCHMARK        = "URTH"
RF               = 0.02
TRAIN_YEARS      = 3
REBAL_DAYS       = 21
MIN_WEIGHT       = 0.0
MAX_NAN_PCT      = 0.20
MAX_SECTOR_WEIGHT = 0.25   # max 25% de poids par secteur dans l'optimiseur
TURNOVER_MAX     = 0.80
N_FACTORS        = 5
VOL_TARGET       = 0.20

# Mapping secteur GICS — 76 actifs de l'univers (source : yfinance)
SECTOR_MAP = {
    "0700.HK": "Communication Services", "AAPL": "Technology",
    "ABBV": "Healthcare",                "ADYEN.AS": "Technology",
    "AIR.PA": "Industrials",             "ALC.SW": "Healthcare",
    "ALV.DE": "Financial Services",      "AMZN": "Consumer Cyclical",
    "ANSS": "Technology",                "ASML.AS": "Technology",
    "AXP": "Financial Services",         "AZN": "Healthcare",
    "BABA": "Consumer Cyclical",         "BAS.DE": "Basic Materials",
    "BLK": "Financial Services",         "BNP.PA": "Financial Services",
    "BRK-B": "Financial Services",       "C": "Financial Services",
    "CAP.PA": "Technology",              "DBK.DE": "Financial Services",
    "DHER.DE": "Consumer Cyclical",      "DHR": "Healthcare",
    "DIS": "Communication Services",     "EDEN.PA": "Financial Services",
    "FAST": "Industrials",               "GOOGL": "Communication Services",
    "GS": "Financial Services",          "HD": "Consumer Cyclical",
    "HDFCBANK.NS": "Financial Services", "HSBC": "Financial Services",
    "HUBS": "Technology",                "ICICIBANK.NS": "Financial Services",
    "IEX": "Industrials",                "INFY.NS": "Technology",
    "INGA.AS": "Financial Services",     "JD": "Consumer Cyclical",
    "JNJ": "Healthcare",                 "JPM": "Financial Services",
    "KER.PA": "Consumer Cyclical",       "KNEBV.HE": "Industrials",
    "KO": "Consumer Defensive",          "LULU": "Consumer Cyclical",
    "MC.PA": "Consumer Cyclical",        "MCD": "Consumer Cyclical",
    "META": "Communication Services",    "MRK": "Healthcare",
    "MS": "Financial Services",          "MSFT": "Technology",
    "NESN.SW": "Consumer Defensive",     "NKE": "Consumer Cyclical",
    "NOVN.SW": "Healthcare",             "NVDA": "Technology",
    "NVS": "Healthcare",                 "ODFL": "Industrials",
    "OR.PA": "Consumer Defensive",       "PAYC": "Technology",
    "PDD": "Consumer Cyclical",          "PEP": "Consumer Defensive",
    "PFE": "Healthcare",                 "PG": "Consumer Defensive",
    "RELIANCE.NS": "Energy",             "RMS.PA": "Consumer Cyclical",
    "RNO.PA": "Consumer Cyclical",       "ROG.SW": "Healthcare",
    "ROK": "Industrials",                "SAN.PA": "Healthcare",
    "SAP.DE": "Technology",              "SBUX": "Consumer Cyclical",
    "SIE.DE": "Industrials",             "SU.PA": "Industrials",
    "TCS.NS": "Technology",              "TMO": "Healthcare",
    "TSM": "Technology",                 "TT": "Industrials",
    "TTE.PA": "Energy",                  "UMI.BR": "Industrials",
    "UNH": "Healthcare",                 "V": "Financial Services",
    "WAT": "Healthcare",                 "ZAL.DE": "Consumer Cyclical",
}

# Suffixes → (ticker yfinance, multiply=True / divide=False)
FX_MAP = {
    ".PA": ("EURUSD=X", True),
    ".DE": ("EURUSD=X", True),
    ".AS": ("EURUSD=X", True),
    ".HE": ("EURUSD=X", True),
    ".BR": ("EURUSD=X", True),
    ".SW": ("USDCHF=X", False),
    ".HK": ("USDHKD=X", False),
    ".NS": ("USDINR=X", False),
}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_lambda=5.0,
    early_stopping_rounds=30,
    random_state=42, n_jobs=-1, verbosity=0,
)


# ─────────────────────────────────────────────────────────────────────────────
# DONNEES
# ─────────────────────────────────────────────────────────────────────────────

def _download_fx(tickers: list, start: str, end: str) -> pd.DataFrame:
    needed = {fx for t in tickers
              for sfx, (fx, _) in FX_MAP.items() if t.endswith(sfx)}
    if not needed:
        return pd.DataFrame()
    rates = {}
    for fx in needed:
        try:
            s = yf.download(fx, start=start, end=end,
                            auto_adjust=True, progress=False)["Close"]
            rates[fx] = s.squeeze().ffill()
            logger.debug(f"  FX {fx} : {len(s)} jours")
        except Exception as e:
            logger.warning(f"  FX {fx} echec : {e}")
    return pd.DataFrame(rates)


def _convert_to_usd(prices: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    if fx.empty:
        return prices
    out = prices.copy()
    converted = []
    for col in prices.columns:
        match = next(((ticker, mul) for sfx, (ticker, mul) in FX_MAP.items()
                      if col.endswith(sfx)), None)
        if match:
            ticker, multiply = match
            if ticker in fx.columns:
                rate = fx[ticker].reindex(prices.index).ffill().bfill()
                out[col] = prices[col] * rate if multiply else prices[col] / rate
                converted.append(col)
    if converted:
        logger.info(f"  Convertis en USD ({len(converted)}) : {converted}")
    return out


def load_universe() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Chargement : {UNIVERSE_FILE.name}")
    df = pd.read_excel(UNIVERSE_FILE, parse_dates=["Date"]).set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Benchmark : SPY depuis le fichier, URTH depuis yfinance si absent
    if BENCHMARK in df.columns:
        benchmark = df[BENCHMARK].ffill()
    else:
        logger.info(f"  Téléchargement benchmark {BENCHMARK} ...")
        start_str = str(df.index[0].date())
        end_str   = str((df.index[-1] + pd.Timedelta(days=1)).date())
        bm = yf.download(BENCHMARK, start=start_str, end=end_str,
                         auto_adjust=True, progress=False)["Close"].squeeze()
        benchmark = bm.reindex(df.index).ffill().bfill()
    raw = df.drop(columns=[BENCHMARK], errors="ignore")

    excluded = raw.columns[raw.isna().mean() > MAX_NAN_PCT].tolist()
    if excluded:
        logger.warning(f"Exclus (>{MAX_NAN_PCT*100:.0f}% NaN) : {excluded}")

    prices = raw.drop(columns=excluded).ffill().dropna(how="all")
    prices = prices.dropna(thresh=int(len(prices.columns) * 0.80))

    # Conversion FX → USD
    start = str(prices.index[0].date())
    end   = str((prices.index[-1] + pd.Timedelta(days=1)).date())
    logger.info("  Téléchargement taux FX ...")
    fx = _download_fx(prices.columns.tolist(), start, end)
    prices = _convert_to_usd(prices, fx)

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

# Features cross-sectionnelles (rangs percentiles dans l'univers)
CS_FEATURES = [
    "cs_ret_1m", "cs_ret_3m", "cs_ret_6m", "cs_ret_12m",
    "cs_vol_1m", "cs_vol_3m",
    "cs_mom_12_1", "cs_mom_6_1", "cs_risk_adj_mom",
    "cs_vol_stress",   # vol_3m / vol_36m — proxy de stress/déclin structurel
]


def _build_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Panel mensuel (date × actif) : observations non-chevauchantes.
    Cible = rendement cumulé sur les 3 prochains mois (horizon trimestriel).
    Features = rangs cross-sectionnels (percentiles) + vol_stress (vol_3m / vol_36m).
    """
    # Résumé mensuel : rendements cumulés sur chaque mois calendaire
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    frames = []
    for col in monthly.columns:
        r = monthly[col].dropna()
        if len(r) < 24:   # au moins 2 ans de données mensuelles
            continue
        # Cible 3 mois : somme des rendements des 3 prochains mois
        target_3m = r.shift(-1) + r.shift(-2) + r.shift(-3)
        vol_6m    = r.rolling(6).std()
        vol_36m   = r.rolling(24).std().replace(0, np.nan)  # 24m = proxy vol long terme
        df = pd.DataFrame({
            "ret_1m":     r.shift(1),
            "ret_3m":     r.rolling(3).sum().shift(1),
            "ret_6m":     r.rolling(6).sum().shift(1),
            "ret_12m":    r.rolling(12).sum().shift(1),
            "vol_1m":     r.rolling(12).std().shift(1) * np.sqrt(12),
            "vol_3m":     r.rolling(24).std().shift(1) * np.sqrt(12),
            "vol_stress": (vol_6m / vol_36m).shift(1),   # stress = vol récente / vol longue
            "target":     target_3m,
        }, index=r.index)
        df["asset"] = col
        frames.append(df.dropna())

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()

    # Features dérivées (facteurs académiques)
    panel["mom_12_1"]     = panel["ret_12m"] - panel["ret_1m"]
    panel["mom_6_1"]      = panel["ret_6m"]  - panel["ret_1m"]
    panel["risk_adj_mom"] = panel["ret_6m"] / panel["vol_3m"].replace(0, np.nan)

    # Rang cross-sectionnel à chaque date (percentile 0-1)
    base = ["ret_1m", "ret_3m", "ret_6m", "ret_12m",
            "vol_1m", "vol_3m", "mom_12_1", "mom_6_1", "risk_adj_mom", "vol_stress"]
    for feature in base:
        panel[f"cs_{feature}"] = panel.groupby("date")[feature].rank(pct=True)

    return panel.dropna(subset=CS_FEATURES + ["target"])


def xgb_predict(prices: pd.DataFrame, train_years: int = TRAIN_YEARS) -> dict:
    """
    XGBoost cross-sectionnel : 1 modèle sur tous les actifs.
    Panel mensuel non-chevauchant : ~76 actifs × 8 ans × 12 mois ≈ 7 400 obs indépendantes.
    Features : rangs cross-sectionnels (percentiles) à chaque fin de mois.
    Retourne : {actif: {pred_annual, mu_norm, importance}}
    """
    returns = prices.pct_change(fill_method=None)
    n_train = train_years * 252
    all_dates = returns.dropna(how="all").index
    t_end     = all_dates[-1]
    t_start   = all_dates[max(0, len(all_dates) - n_train)]
    ret_tr    = returns.loc[t_start:t_end]

    logger.info(f"  Fenetre : {t_start.date()} → {t_end.date()}")

    panel = _build_panel(ret_tr)
    if panel.empty:
        logger.warning("Panel vide — pas assez de données")
        return {}

    X_all = panel[CS_FEATURES]
    y_all = panel["target"]

    # Split temporel : derniers 20% des mois pour validation (early stopping)
    # max(6, ...) pour garantir au moins 6 mois de validation OOS
    sorted_dates = sorted(panel["date"].unique())
    n_val_dates  = max(6, int(len(sorted_dates) * 0.20))
    split_date   = sorted_dates[-n_val_dates]

    tr_mask  = panel["date"] <  split_date
    val_mask = panel["date"] >= split_date

    sc       = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_all[tr_mask])
    X_val_sc = sc.transform(X_all[val_mask])

    # Pondération temporelle : derniers 12 mois d'entraînement × 3
    tr_dates       = panel.loc[tr_mask, "date"]
    unique_tr      = sorted(tr_dates.unique())
    recent_cutoff  = unique_tr[-12] if len(unique_tr) >= 12 else unique_tr[0]
    sw_tr          = np.where(tr_dates >= recent_cutoff, 3.0, 1.0)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr_sc, y_all[tr_mask].values,
        sample_weight=sw_tr,
        eval_set=[(X_val_sc, y_all[val_mask].values)],
        verbose=False,
    )

    logger.success(
        f"XGBoost CS : {len(panel):,} obs  |  "
        f"train {tr_mask.sum():,}  val {val_mask.sum():,}  |  "
        f"{panel['asset'].nunique()} actifs"
    )

    # Prédiction sur la dernière date disponible (cross-section complète)
    last_date  = sorted_dates[-1]
    last_panel = panel[panel["date"] == last_date]
    if last_panel.empty:
        last_date  = sorted_dates[-2]
        last_panel = panel[panel["date"] == last_date]

    X_last = sc.transform(last_panel[CS_FEATURES])
    preds  = model.predict(X_last)
    imp    = pd.Series(model.feature_importances_, index=CS_FEATURES)

    results = {}
    for j, (_, row) in enumerate(last_panel.iterrows()):
        asset = row["asset"]
        results[asset] = {
            "pred_annual": float(preds[j]) * 4,   # cible = 3 mois → ×4 pour annualiser
            "importance":  imp,
        }

    # Normalisation mu_norm ∈ [-1, +1] centré sur la moyenne cross-sectionelle
    raw           = pd.Series({k: v["pred_annual"] for k, v in results.items()})
    mu_winsorized = pd.Series(np.array(_winsorize(raw.values, limits=[0.05, 0.05])), index=raw.index)
    mu_centered   = mu_winsorized - mu_winsorized.mean()
    abs_max       = mu_centered.abs().max()
    mu_normalized = (mu_centered / abs_max if abs_max > 0
                     else pd.Series(0.0, index=mu_centered.index))
    for k in results:
        results[k]["mu_norm"] = float(mu_normalized.get(k, 0.0))

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
    logger.info("  " + "─" * 46)
    for asset, row in rk.iterrows():
        sym = "▲" if row["predicted_annual"] >= 0 else "▼"
        logger.info(f"  {int(row['rank']):<4}  {asset:<20}  "
                    f"{sym} {row['predicted_annual']*100:>+6.2f}%  {row['mu_normalized']:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def _solve(obj_fn, n: int, constraints: list) -> np.ndarray:
    bounds   = [(MIN_WEIGHT, 1.0)] * n   # poids minimum garanti sur chaque actif
    best_val = np.inf
    best_w   = None
    for seed in [42, 123, 456, 789, 1337]:
        w0  = np.ones(n) / n + np.random.default_rng(seed).normal(0, 0.02, n)
        w0  = np.clip(w0, MIN_WEIGHT, 1.0)
        w0 /= w0.sum()
        res = minimize(obj_fn, w0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"ftol": 1e-10, "maxiter": 2000})
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x.copy()
    if best_w is None:
        best_w = np.ones(n) / n
    best_w /= best_w.sum()
    return best_w


def _solve_mosek(signal: np.ndarray, Sigma: np.ndarray, n: int,
                 sector_indices: dict, wp: np.ndarray, has_prev: bool) -> np.ndarray | None:
    """
    Reformulation Charnes-Cooper de max w'signal / sqrt(w'Σw) en SOCP exact.
    Substitution y = t·w, t = 1/sqrt(w'Σw) → max y'signal  s.t. ||L'y||₂ ≤ 1.
    """
    lmin = np.linalg.eigvalsh(Sigma).min()
    S    = Sigma + max(0.0, -lmin + 1e-8) * np.eye(n)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return None

    y = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)

    constrs = [
        cp.norm(L.T @ y, 2) <= 1,      # normalise le dénominateur
        cp.sum(y) == t,                  # sum(w)=1 ↔ sum(y)=t
        t >= 1.0 / VOL_TARGET,          # contrainte volatilité cible
    ]
    for idx_list in sector_indices.values():
        idx = list(idx_list)
        constrs.append(cp.sum(y[idx]) <= MAX_SECTOR_WEIGHT * t)

    if has_prev:
        z = cp.Variable(n, nonneg=True)
        constrs += [
            z >= y - t * wp,
            z >= -(y - t * wp),
            cp.sum(z) <= TURNOVER_MAX * t,
        ]

    prob = cp.Problem(cp.Maximize(signal @ y), constrs)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        logger.warning(f"  MOSEK erreur : {e}")
        return None

    if (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and y.value is not None and t.value is not None
            and t.value > 1e-10):
        w = np.maximum(y.value / t.value, 0.0)
        w_sum = w.sum()
        return w / w_sum if w_sum > 1e-10 else None

    logger.warning(f"  MOSEK status={prob.status} — fallback SLSQP")
    return None


def factor_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Σ = B Σ_f B' + Σ_ε  via PCA
    Plus robuste que la covariance empirique brute.
    """
    R    = returns.values
    T, n = R.shape
    k    = min(N_FACTORS, n - 1, T - 1)

    pca = PCA(n_components=k)
    pca.fit(R)

    B         = pca.components_.T          # (n, k) loadings
    F         = R @ B                      # (T, k) factor returns
    Sigma_f   = np.cov(F.T) * 252
    residuals = R - F @ B.T
    Sigma_eps = np.diag(np.var(residuals, axis=0) * 252)

    Sigma = B @ Sigma_f @ B.T + Sigma_eps

    # Correction PSD
    lmin = np.linalg.eigvalsh(Sigma).min()
    if lmin < 1e-8:
        Sigma += (abs(lmin) + 1e-8) * np.eye(n)

    return Sigma


def maxdiv_ml(returns: pd.DataFrame, xgb_res: dict,
              w_prev: dict | None = None) -> dict:
    """
    Univers complet (tous les actifs disponibles) — pas de sélection TOP_N.
    Factor covariance via PCA : Σ = B Σ_f B' + Σ_ε
    Optimisation :
       max   w'(σ ⊙ μ) / sqrt(w'Σw)
       s.c.  w'Σ w ≤ VOL_TARGET²
             sum(w) = 1,  w_i ≥ MIN_WEIGHT
             ||w - w_prev||₁ ≤ TURNOVER_MAX
             sum(w_i pour i dans secteur S) ≤ MAX_SECTOR_WEIGHT  ∀ S
    """
    # Tous les actifs disponibles dans les données de train ET scorés par XGBoost
    mu_all = pd.Series({k: xgb_res[k]["mu_norm"] for k in xgb_res if k in returns.columns})
    names  = mu_all.index.tolist()
    logger.info(f"  Univers complet : {len(names)} actifs")

    returns  = returns[names]
    n        = len(names)
    sigma    = returns.std().values  * np.sqrt(252)
    Sigma    = factor_covariance(returns)
    mu_hist  = returns.mean().values * 252
    mu       = mu_all.values
    signal   = sigma * mu                  # σ ⊙ μ

    # Poids précédents — contrainte turnover seulement si un portefeuille précédent existe
    if w_prev is not None:
        wp       = np.array([w_prev.get(a, 0.0) for a in names])
        has_prev = True
    else:
        wp       = np.zeros(n)
        has_prev = False

    sigma2_target = VOL_TARGET ** 2

    def objective(w):
        num = float(w @ signal)
        den = float(np.sqrt(w @ Sigma @ w))
        return -num / den if den > 1e-12 else 1e10

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: sigma2_target - float(w @ Sigma @ w)},
    ]
    if has_prev:
        constraints.append(
            {"type": "ineq", "fun": lambda w: TURNOVER_MAX - float(np.sum(np.abs(w - wp)))}
        )

    # Contraintes sectorielles : sum(w_i pour secteur S) ≤ MAX_SECTOR_WEIGHT
    sector_indices: dict[str, list[int]] = defaultdict(list)
    for j, asset in enumerate(names):
        sec = SECTOR_MAP.get(asset, "Unknown")
        sector_indices[sec].append(j)
    for sec, idx_list in sector_indices.items():
        idx_frozen = list(idx_list)
        constraints.append({
            "type": "ineq",
            "fun": lambda w, ix=idx_frozen: MAX_SECTOR_WEIGHT - float(w[ix].sum())
        })

    if _MOSEK_OK:
        w_arr = _solve_mosek(signal, Sigma, n, sector_indices, wp, has_prev)
        w = w_arr if w_arr is not None else _solve(objective, n, constraints)
    else:
        w = _solve(objective, n, constraints)
    vol = float(np.sqrt(w @ Sigma @ w))
    ret = float(mu_hist @ w)
    return {
        "weights":    dict(zip(names, w)),
        "return":     ret,
        "volatility": vol,
        "sharpe":     (ret - RF) / vol if vol > 0 else 0,
        "signal":     float(w @ signal),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame,
                 benchmark: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train   = TRAIN_YEARS * 252
    dates     = prices.index
    rebal_idx = list(range(n_train, len(dates), REBAL_DAYS))

    freq_label = (f"mensuel" if REBAL_DAYS == 21 else
                  f"trimestriel" if REBAL_DAYS == 63 else
                  f"annuel" if REBAL_DAYS == 252 else f"{REBAL_DAYS}j")
    logger.info(f"  {len(rebal_idx)} rebalancements ({freq_label})  "
                f"│  {dates[n_train].date()} → {dates[-1].date()}")

    ml_rets, all_dates   = [], []
    w_prev_dict          = None
    log_records          = []
    weights_history      = {}   # {date: weights_dict}
    importance_history   = []   # [{date, feature, importance}]

    for i, idx in enumerate(rebal_idx):
        t_start = max(0, idx - n_train)
        t_end   = min(len(dates) - 1, idx + REBAL_DAYS)
        r_train = prices.iloc[t_start:idx].pct_change(fill_method=None).dropna()
        r_test  = prices.iloc[idx:t_end].pct_change(fill_method=None).dropna()

        if len(r_test) == 0:
            continue

        try:
            xr     = xgb_predict(prices.iloc[t_start:idx], train_years=TRAIN_YEARS)
            result = maxdiv_ml(r_train, xr, w_prev=w_prev_dict)
            w      = result["weights"]

            # Spearman rank correlation : prédictions XGBoost vs rendements réels (OOS)
            common = [a for a in xr if a in r_test.columns]
            if len(common) > 4:
                y_pred = np.array([xr[a]["pred_annual"] for a in common], dtype=float)
                y_real = np.array([r_test[a].sum() for a in common], dtype=float)
                mask   = np.isfinite(y_pred) & np.isfinite(y_real)
                if mask.sum() > 4:
                    rho, _       = spearmanr(y_pred[mask], y_real[mask])
                    r2_oos = float(rho)
                else:
                    r2_oos = float("nan")
            else:
                r2_oos = float("nan")
            # Turnover réalisé
            if w_prev_dict is not None:
                all_assets = set(w) | set(w_prev_dict)
                turnover   = sum(abs(w.get(a, 0) - w_prev_dict.get(a, 0)) for a in all_assets)
            else:
                turnover = float("nan")

            w_prev_dict = w
            weights_history[dates[idx]] = w

            # Feature importances moyennes
            for v in xr.values():
                for feat, imp in v["importance"].items():
                    importance_history.append({"date": dates[idx], "feature": feat, "importance": imp})

            n_active = sum(1 for v in w.values() if v > MIN_WEIGHT * 5)
            logger.info(
                f"  [{i+1:>3}/{len(rebal_idx)}]  "
                f"{dates[idx].date()} → {dates[t_end-1].date()}  │  "
                f"R²={r2_oos:>+.3f}  actifs={n_active}/{len(w)}  "
                f"turn={turnover*100:>5.1f}%"
            )

        except Exception as e:
            logger.warning(f"  [{i+1:>3}/{len(rebal_idx)}] echec : {e}")
            names = r_train.columns.tolist()
            w = dict(zip(names, np.ones(len(names)) / len(names)))
            w_prev_dict = w
            r2_oos = float("nan")
            turnover     = float("nan")

        weights_array  = np.array([w.get(c, 0) for c in r_test.columns])
        weights_array /= weights_array.sum() if weights_array.sum() > 0 else 1

        # Buy-and-hold : poids dérivent avec les prix
        price_rel = (1 + r_test).cumprod()
        port_val  = price_rel.dot(weights_array)
        port_rets = port_val.pct_change()
        port_rets.iloc[0] = port_val.iloc[0] - 1

        period_ret = float((port_val.iloc[-1]) - 1)

        spy_period = (benchmark.pct_change(fill_method=None)
                      .reindex(r_test.index).fillna(0)
                      if benchmark is not None else None)
        spy_ret = float((1 + spy_period).prod() - 1) if spy_period is not None else float("nan")

        # Contributions individuelles par actif sur la période
        contrib = {a: float(weights_array[j] * r_test.iloc[:, j].sum())
                   for j, a in enumerate(r_test.columns) if weights_array[j] > 1e-6}
        top_contrib    = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]
        bottom_contrib = sorted(contrib.items(), key=lambda x: x[1])[:3]

        beat_str = "BEAT" if period_ret > spy_ret else "MISS"
        logger.debug(
            f"  [{i+1:>3}/{len(rebal_idx)}]  R²={r2_oos:>+.3f}  "
            f"ML={period_ret*100:>+5.1f}%  {BENCHMARK}={spy_ret*100:>+5.1f}%  [{beat_str}]  "
            f"turn={turnover*100:>5.1f}%\n"
            f"    TOP   : {', '.join(f'{a}={v*100:+.1f}%' for a,v in top_contrib)}\n"
            f"    DRAG  : {', '.join(f'{a}={v*100:+.1f}%' for a,v in bottom_contrib)}"
        )

        log_records.append({
            "date":      dates[idx].date(),
            "r2_oos": r2_oos,
            "turnover":  turnover,
            "ml_ret":    period_ret,
            "spy_ret":   spy_ret,
            "beat_spy":  period_ret > spy_ret if not np.isnan(spy_ret) else False,
        })

        ml_rets.extend(port_rets.tolist())
        all_dates.extend(r_test.index.tolist())

    # ── Résumé backtest ──────────────────────────────────────────────────────
    if log_records:
        rec        = pd.DataFrame(log_records)
        r2_valid   = rec["r2_oos"].dropna()
        turn_valid = rec["turnover"].dropna()
        win_rate   = rec["beat_spy"].mean() * 100

        logger.info("─" * 58)
        logger.info(f"  R² OOS moyen          : {r2_valid.mean():>+.4f}  "
                    f"(min {r2_valid.min():>+.4f} / max {r2_valid.max():>+.4f})")
        logger.info(f"  Turnover moyen        : {turn_valid.mean()*100:>5.1f}%  "
                    f"(max {turn_valid.max()*100:.1f}%)")
        logger.info(f"  Win rate vs SPY       : {win_rate:>5.1f}%  "
                    f"({rec['beat_spy'].sum()}/{len(rec)} périodes)")
        logger.info("─" * 58)

    rets_df = pd.DataFrame({"MaxDiv ML": ml_rets},
                            index=pd.DatetimeIndex(all_dates[:len(ml_rets)]))

    if benchmark is not None:
        rets_df[BENCHMARK] = (benchmark.pct_change(fill_method=None)
                                        .reindex(rets_df.index).fillna(0))

    metrics_df = pd.DataFrame(
        [{**portfolio_metrics(rets_df[s]), "Strategy": s} for s in rets_df.columns]
    ).set_index("Strategy")

    # ── Export Excel poids (format long : date / ticker / weight) ────────────
    if weights_history:
        rows_xl = [
            {"date": str(dt.date()) if hasattr(dt, "date") else str(dt),
             "ticker": ticker, "weight": round(float(w), 6)}
            for dt, wdict in weights_history.items()
            for ticker, w in wdict.items()
            if w > 1e-6
        ]
        xl_path = OUT / "backtest_weights.xlsx"
        try:
            pd.DataFrame(rows_xl).to_excel(xl_path, index=False)
        except PermissionError:
            xl_path = OUT / f"backtest_weights_{datetime.now().strftime('%H%M%S')}.xlsx"
            pd.DataFrame(rows_xl).to_excel(xl_path, index=False)
        logger.info(f"  Poids exportés -> {xl_path.name}  ({len(rows_xl)} lignes)")

    return rets_df, metrics_df, weights_history, log_records, importance_history


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


def plot_cumulative(returns_df: pd.DataFrame, title: str, fname: str,
                    log_records: list | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── NAV cumulée + R² en axe secondaire ──────────────────────────────────
    cum = (1 + returns_df).cumprod()
    for col in cum.columns:
        lw = 2.5 if col == "MaxDiv ML" else 1.8
        ls = "--" if col == BENCHMARK else "-"
        axes[0].plot(cum.index, cum[col], label=col, linewidth=lw, linestyle=ls,
                     color=PALETTE.get(col, "gray"))
    axes[0].set_ylabel("Valeur (base 1)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))

    if log_records:
        rec = pd.DataFrame(log_records)
        rec["date"] = pd.to_datetime(rec["date"])
        r2  = rec.dropna(subset=["r2_oos"]).sort_values("date")
        clr = ["#55A868" if v >= 0 else "#C44E52" for v in r2["r2_oos"]]
        ax2 = axes[0].twinx()
        ax2.bar(r2["date"], r2["r2_oos"], color=clr, width=15,
                edgecolor="none", alpha=0.35, label="R² OOS")
        ax2.axhline(0, color="white", lw=0.8, linestyle=":")
        ax2.set_ylabel("R² OOS", fontsize=9)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        # Légende combinée
        handles0, labels0 = axes[0].get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        axes[0].legend(handles0 + handles2, labels0 + labels2, fontsize=9)
    else:
        axes[0].legend(fontsize=10)

    # ── Drawdown ─────────────────────────────────────────────────────────────
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


def plot_rolling_metrics(rets_df: pd.DataFrame, fname: str = "backtest_rolling.png") -> None:
    window = 252
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("Métriques glissantes 12 mois — MaxDiv ML vs SPY", fontsize=13, fontweight="bold")

    for col in rets_df.columns:
        r   = rets_df[col]
        lw  = 2.5 if col == "MaxDiv ML" else 1.5
        ls  = "-" if col == "MaxDiv ML" else "--"
        clr = PALETTE.get(col, "gray")

        roll_ret = r.rolling(window).mean() * 252
        roll_vol = r.rolling(window).std() * np.sqrt(252)
        roll_sh  = (roll_ret - RF) / roll_vol.replace(0, np.nan)

        axes[0].plot(roll_ret.index, roll_ret * 100, label=col, linewidth=lw,
                     linestyle=ls, color=clr)
        axes[1].plot(roll_sh.index, roll_sh, label=col, linewidth=lw,
                     linestyle=ls, color=clr)

    axes[0].axhline(0, color="white", lw=0.8, linestyle=":")
    axes[1].axhline(0, color="white", lw=0.8, linestyle=":")
    axes[0].set_ylabel("Rendement annualisé glissant (%)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Sharpe glissant (12 mois)")
    axes[1].legend(fontsize=9)
    plt.tight_layout(); _save(fname)


def plot_weights_evolution(weights_history: dict, fname: str = "backtest_weights_evolution.png") -> None:
    if not weights_history:
        return
    df = pd.DataFrame(weights_history).T.fillna(0)
    df.index = pd.to_datetime(df.index)

    # Garder les top 10 actifs par poids moyen
    top10 = df.mean().nlargest(10).index.tolist()
    df_top = df[top10]
    other  = df.drop(columns=top10).sum(axis=1)

    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Évolution des poids — Top 10 actifs", fontsize=13, fontweight="bold")

    bottom = np.zeros(len(df_top))
    for j, col in enumerate(top10):
        ax.fill_between(df_top.index, bottom, bottom + df_top[col].values,
                        label=col, alpha=0.85, color=colors[j % 10])
        bottom += df_top[col].values
    ax.fill_between(df_top.index, bottom, bottom + other.reindex(df_top.index).fillna(0).values,
                    label="Autres", alpha=0.4, color="gray")

    ax.set_ylabel("Poids (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout(); _save(fname)


def plot_xgb_diagnostics(log_records: list, importance_history: list,
                          fname: str = "backtest_xgb_diagnostics.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XGBoost — Diagnostics backtest", fontsize=13, fontweight="bold")

    # R² OOS par période
    rec = pd.DataFrame(log_records)
    rec["date"] = pd.to_datetime(rec["date"])
    r2  = rec.dropna(subset=["r2_oos"])
    clr = ["#55A868" if v >= 0 else "#C44E52" for v in r2["r2_oos"]]
    axes[0].bar(r2["date"], r2["r2_oos"], color=clr, width=15, edgecolor="none")
    axes[0].axhline(0, color="white", lw=1, linestyle="--")
    axes[0].set_title("R² OOS (XGBoost ranking vs réalisé) par période", fontsize=11)
    axes[0].set_ylabel("R² OOS")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))

    # Feature importances moyennes
    if importance_history:
        imp_df  = pd.DataFrame(importance_history)
        avg_imp = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=True)
        colors  = ["#FF6B35" if v >= avg_imp.median() else "#888888" for v in avg_imp]
        axes[1].barh(avg_imp.index, avg_imp.values, color=colors, edgecolor="none")
        axes[1].set_title("Feature importances moyennes (XGBoost)", fontsize=11)
        axes[1].set_xlabel("Importance moyenne")

    plt.tight_layout(); _save(fname)


def plot_metrics_table(metrics_df: pd.DataFrame, fname: str = "metrics.png") -> None:
    cols  = {"Annual Return": True, "Annual Vol": False, "Sharpe": True,
             "Sortino": True, "Max Drawdown": False, "Calmar": True}
    avail = [c for c in cols if c in metrics_df.columns]
    sub   = metrics_df[avail]
    norm  = pd.DataFrame(index=sub.index, columns=sub.columns, dtype=float)
    for col, up in cols.items():
        if col not in sub.columns: continue
        col_min, col_max = sub[col].min(), sub[col].max()
        v = (sub[col] - col_min) / (col_max - col_min) if col_max > col_min else pd.Series(0.5, index=sub.index)
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
        freq_str = (f"mensuel" if REBAL_DAYS == 21 else
                    f"trimestriel" if REBAL_DAYS == 63 else
                    f"annuel" if REBAL_DAYS == 252 else f"{REBAL_DAYS}j")
        _section(f"BACKTEST  ({TRAIN_YEARS} ans train / rebal. {freq_str})")
        rets_df, met, w_hist, log_rec, imp_hist = run_backtest(prices, benchmark)
        for s in met.index:
            _log_metrics(met.loc[s].to_dict(), s)
        plot_cumulative(rets_df, "Backtest Walk-Forward — MaxDiv ML vs SPY", "backtest_cumulative.png",
                        log_records=log_rec)
        plot_metrics_table(met, "backtest_metrics.png")
        plot_rolling_metrics(rets_df)
        plot_weights_evolution(w_hist)
        plot_xgb_diagnostics(log_rec, imp_hist)

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
