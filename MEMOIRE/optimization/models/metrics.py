import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class PortfolioMetrics:

    def __init__(self, returns: pd.DataFrame, weights: dict | np.ndarray, rf: float = 0.02):
        if isinstance(weights, dict):
            weights = np.array([weights.get(c, 0) for c in returns.columns])
        self.r  = (returns * weights).sum(axis=1)
        self.rf = rf

    def annual_return(self):   return float(self.r.mean() * 252)
    def annual_vol(self):      return float(self.r.std() * np.sqrt(252))
    def sharpe(self):          return (self.annual_return() - self.rf) / self.annual_vol()

    def sortino(self):
        down = self.r[self.r < self.rf / 252]
        dv   = float(np.sqrt(np.mean(down ** 2)) * np.sqrt(252))
        return (self.annual_return() - self.rf) / dv if dv > 0 else 0

    def max_drawdown(self):
        cum = (1 + self.r).cumprod()
        return float(((cum - cum.expanding().max()) / cum.expanding().max()).min())

    def calmar(self):
        dd = abs(self.max_drawdown())
        return self.annual_return() / dd if dd > 0 else 0

    def var95(self):  return float(np.percentile(self.r, 5))
    def cvar95(self): return float(self.r[self.r <= self.var95()].mean())

    def all(self) -> dict:
        return {
            "Annual Return":   self.annual_return(),
            "Annual Vol":      self.annual_vol(),
            "Sharpe":          self.sharpe(),
            "Sortino":         self.sortino(),
            "Max Drawdown":    self.max_drawdown(),
            "Calmar":          self.calmar(),
            "VaR 95%":         self.var95(),
            "CVaR 95%":        self.cvar95(),
            "Skewness":        float(skew(self.r)),
            "Kurtosis":        float(kurtosis(self.r)),
            "Win Rate":        float((self.r > 0).mean()),
        }


def compare_metrics(results: dict, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = PortfolioMetrics(returns, res["weights"]).all()
        m["Strategy"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Strategy")
    return df
