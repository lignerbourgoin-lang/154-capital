import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzOptimizer:

    def __init__(self, returns: pd.DataFrame):
        self.returns      = returns
        self.names        = returns.columns.tolist()
        self.mean_returns = returns.mean().values
        self.cov_matrix   = returns.cov().values
        self.n            = len(self.names)

    def _portfolio_perf(self, weights):
        ret = float(self.mean_returns @ weights)
        vol = float(np.sqrt(weights @ self.cov_matrix @ weights))
        return ret, vol

    def min_variance(self) -> dict:
        def objective(w): return np.sqrt(w @ self.cov_matrix @ w)
        result = minimize(objective, np.ones(self.n) / self.n,
                          method="SLSQP",
                          bounds=[(0, 1)] * self.n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        ret, vol = self._portfolio_perf(result.x)
        return {"weights": dict(zip(self.names, result.x)), "return": ret, "volatility": vol}

    def max_sharpe(self, rf: float = 0.02) -> dict:
        def objective(w):
            ret, vol = self._portfolio_perf(w)
            return -(ret - rf) / vol if vol > 0 else 1e10
        result = minimize(objective, np.ones(self.n) / self.n,
                          method="SLSQP",
                          bounds=[(0, 1)] * self.n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        ret, vol = self._portfolio_perf(result.x)
        return {"weights": dict(zip(self.names, result.x)),
                "return": ret, "volatility": vol,
                "sharpe": (ret - rf) / vol}

    def efficient_frontier(self, n_points: int = 50) -> dict:
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), n_points)
        vols, rets, weights_list = [], [], []

        for target in target_returns:
            def objective(w): return np.sqrt(w @ self.cov_matrix @ w)
            result = minimize(objective, np.ones(self.n) / self.n,
                              method="SLSQP",
                              bounds=[(0, 1)] * self.n,
                              constraints=[
                                  {"type": "eq", "fun": lambda w: w.sum() - 1},
                                  {"type": "eq", "fun": lambda w: self.mean_returns @ w - target},
                              ])
            if result.success:
                _, vol = self._portfolio_perf(result.x)
                vols.append(vol)
                rets.append(float(target))
                weights_list.append(result.x)

        return {"volatilities": vols, "returns": rets, "weights": weights_list}
