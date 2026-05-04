import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

TICKERS = {
    "LMT":       "Lockheed Martin",
    "AIR.PA":    "Airbus",
    "AM.PA":     "Dassault Aviation",
    "RHM.DE":    "Rheinmetall",
    "SAAB-B.ST": "Saab",
}


def load_prices(start: str = "2000-01-01") -> pd.DataFrame:
    print(f"  Downloading {len(TICKERS)} tickers from {start} ...")
    raw = yf.download(list(TICKERS.keys()), start=start, auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=TICKERS).dropna(how="all")
    print(f"  {raw.shape[0]} jours  |  {raw.index[0].date()} -> {raw.index[-1].date()}")
    return raw


def build_features(prices: pd.DataFrame) -> dict:
    """
    Pour chaque actif : (X, y) avec y = rendement J+1.

    Features :
      ret_1, ret_5, ret_10, ret_21  : rendements/moyennes mobiles passes
      vol_5, vol_21                  : volatilite realisee
      mom_63                         : momentum trimestriel
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

    return datasets


def run_ols(datasets: dict) -> dict:
    """OLS statsmodels, erreurs robustes HC3. Retourne stats + prediction J+1."""
    results = {}
    for name, (X, y) in datasets.items():
        X_c   = sm.add_constant(X)
        model = sm.OLS(y, X_c).fit(cov_type="HC3")
        results[name] = {
            "r2":        model.rsquared,
            "r2_adj":    model.rsquared_adj,
            "f_stat":    model.fvalue,
            "f_pval":    model.f_pvalue,
            "n_obs":     int(model.nobs),
            "params":    model.params.drop("const"),
            "tvalues":   model.tvalues.drop("const"),
            "pvalues":   model.pvalues.drop("const"),
            "last_pred": float(model.predict(X_c.iloc[[-1]])),
        }
        sig = "[sig]" if model.f_pvalue < 0.05 else "[ns] "
        print(f"  {name:<25}  R2={model.rsquared:.4f}  F-pval={model.f_pvalue:.4f} {sig}")
    return results


def run_xgboost(datasets: dict, n_splits: int = 5) -> dict:
    """XGBoost avec TimeSeriesSplit. Retourne OOF R², RMSE, importance, prediction J+1."""
    results = {}
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    params  = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, n_jobs=-1)

    for name, (X, y) in datasets.items():
        oof = np.full(len(y), np.nan)
        for tr, val in tscv.split(X):
            m = xgb.XGBRegressor(**params, verbosity=0)
            m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
            oof[val] = m.predict(X.iloc[val])

        valid  = ~np.isnan(oof)
        r2     = r2_score(y.values[valid], oof[valid])
        rmse   = np.sqrt(mean_squared_error(y.values[valid], oof[valid]))

        final  = xgb.XGBRegressor(**params, verbosity=0)
        final.fit(X, y)

        results[name] = {
            "r2":         r2,
            "rmse":       rmse,
            "importance": pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False),
            "last_pred":  float(final.predict(X.iloc[[-1]])),
        }
        print(f"  {name:<25}  OOF R2={r2:.4f}  RMSE={rmse:.6f}")

    return results


def ml_weights(predictions: dict) -> dict:
    """Poids proportionnels aux rendements predits positifs. Fallback equal weight."""
    clipped = {k: max(v, 0.0) for k, v in predictions.items()}
    total   = sum(clipped.values())
    if total == 0:
        n = len(predictions)
        return {k: 1 / n for k in predictions}
    return {k: v / total for k, v in clipped.items()}
