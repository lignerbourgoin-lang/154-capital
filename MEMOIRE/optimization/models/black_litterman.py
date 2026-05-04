import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BlackLittermanOptimizer:

    def __init__(self, returns: pd.DataFrame, rf: float = 0.02):
        self.returns = returns
        self.names   = returns.columns.tolist()
        self.mu      = returns.mean().values
        self.cov     = returns.cov().values
        self.n       = len(self.names)
        self.rf      = rf

        w_mkt            = np.ones(self.n) / self.n
        mkt_var          = float(w_mkt @ self.cov @ w_mkt)
        lam              = (float(self.mu @ w_mkt) - rf) / mkt_var
        self.implied_mu  = rf + lam * (self.cov @ w_mkt)
        self.w_mkt       = w_mkt
        self.views       = []

    def add_views(self, views: list[dict]) -> None:
        """
        views = [
            {"assets": ["Airbus"], "expected_return": 0.08, "confidence": 0.8},
            ...
        ]
        """
        self.views = views

    def _posterior(self, tau: float = 0.025) -> np.ndarray:
        if not self.views:
            return self.implied_mu.copy()

        P = np.array([self._p_vector(v) for v in self.views])
        Q = np.array([v["expected_return"] for v in self.views])
        Omega = np.diag([
            (v["confidence"] ** -1) * tau * float(self.cov[np.ix_(self._idx(v), self._idx(v))].sum())
            for v in self.views
        ])

        term   = tau * P @ self.cov @ P.T + Omega
        adj    = self.cov @ P.T @ np.linalg.pinv(term) @ (Q - P @ self.implied_mu)
        return self.implied_mu + tau * adj

    def _p_vector(self, view: dict) -> np.ndarray:
        p = np.zeros(self.n)
        for asset in view["assets"]:
            if asset in self.names:
                p[self.names.index(asset)] = 1 / len(view["assets"])
        return p

    def _idx(self, view: dict) -> list:
        return [self.names.index(a) for a in view["assets"] if a in self.names]

    def optimize(self, tau: float = 0.025) -> dict:
        mu_post = self._posterior(tau)

        def objective(w):
            ret = float(mu_post @ w)
            vol = float(np.sqrt(w @ self.cov @ w))
            return -(ret - self.rf) / vol if vol > 0 else 1e10

        result = minimize(objective, self.w_mkt.copy(),
                          method="SLSQP",
                          bounds=[(0, 1)] * self.n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1})

        w   = result.x
        ret = float(mu_post @ w)
        vol = float(np.sqrt(w @ self.cov @ w))

        return {
            "weights":          dict(zip(self.names, w)),
            "return":           ret,
            "volatility":       vol,
            "sharpe":           (ret - self.rf) / vol if vol > 0 else 0,
            "posterior_mu":     dict(zip(self.names, mu_post)),
            "implied_mu":       dict(zip(self.names, self.implied_mu)),
            "historical_mu":    dict(zip(self.names, self.mu)),
        }
