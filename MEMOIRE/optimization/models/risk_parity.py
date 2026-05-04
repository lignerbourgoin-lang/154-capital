import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:

    def __init__(self, returns: pd.DataFrame, rf: float = 0.02):
        self.returns  = returns
        self.names    = returns.columns.tolist()
        self.mu       = returns.mean().values
        self.cov      = returns.cov().values
        self.n        = len(self.names)
        self.rf       = rf

    def _vol(self, w):
        return float(np.sqrt(w @ self.cov @ w))

    def _rc(self, w):
        vol = self._vol(w)
        return w * (self.cov @ w) / vol if vol > 0 else np.zeros(self.n)

    def optimize(self) -> dict:
        def objective(w):
            rc     = self._rc(w)
            target = self._vol(w) / self.n
            return float(np.sum((rc - target) ** 2))

        result = minimize(objective, np.ones(self.n) / self.n,
                          method="SLSQP",
                          bounds=[(0, 1)] * self.n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                          options={"maxiter": 1000, "ftol": 1e-9})

        w   = result.x
        ret = float(self.mu @ w)
        vol = self._vol(w)
        rc  = self._rc(w)

        return {
            "weights":            dict(zip(self.names, w)),
            "return":             ret,
            "volatility":         vol,
            "sharpe":             (ret - self.rf) / vol if vol > 0 else 0,
            "risk_contributions": dict(zip(self.names, rc)),
            "rc_pct":             dict(zip(self.names, rc / vol * 100 if vol > 0 else np.zeros(self.n))),
        }
