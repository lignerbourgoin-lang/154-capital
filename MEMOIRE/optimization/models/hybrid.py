import numpy as np
import pandas as pd
from scipy.optimize import minimize


class HybridOptimizer:
    """Risk Parity + Black-Litterman : maximise Sharpe avec penalite risk parity."""

    def __init__(self, returns: pd.DataFrame, rf: float = 0.02):
        self.returns = returns
        self.names   = returns.columns.tolist()
        self.mu      = returns.mean().values
        self.cov     = returns.cov().values
        self.n       = len(self.names)
        self.rf      = rf

        w_mkt           = np.ones(self.n) / self.n
        mkt_var         = float(w_mkt @ self.cov @ w_mkt)
        lam             = (float(self.mu @ w_mkt) - rf) / mkt_var
        self.implied_mu = rf + lam * (self.cov @ w_mkt)
        self.w_mkt      = w_mkt
        self.views      = []

    def add_views(self, views: list[dict]) -> None:
        self.views = views

    def _posterior(self, tau: float) -> np.ndarray:
        if not self.views:
            return self.implied_mu.copy()

        P = np.zeros((len(self.views), self.n))
        Q = np.zeros(len(self.views))
        diag = []

        for i, v in enumerate(self.views):
            for asset in v["assets"]:
                if asset in self.names:
                    P[i, self.names.index(asset)] = 1 / len(v["assets"])
            Q[i] = v["expected_return"]
            diag.append((v["confidence"] ** -1) * tau * float(P[i] @ self.cov @ P[i]))

        Omega = np.diag(diag)
        term  = tau * P @ self.cov @ P.T + Omega
        adj   = self.cov @ P.T @ np.linalg.pinv(term) @ (Q - P @ self.implied_mu)
        return self.implied_mu + tau * adj

    def _vol(self, w): return float(np.sqrt(w @ self.cov @ w))

    def _rc(self, w):
        vol = self._vol(w)
        return w * (self.cov @ w) / vol if vol > 0 else np.zeros(self.n)

    def optimize(self, tau: float = 0.025, penalty: float = 100) -> dict:
        mu_post = self._posterior(tau)

        def objective(w):
            ret    = float(mu_post @ w)
            vol    = self._vol(w)
            sharpe = -(ret - self.rf) / vol if vol > 0 else 1e10
            rc     = self._rc(w)
            target = vol / self.n
            rp_pen = penalty * float(np.sum((rc - target) ** 2))
            return sharpe + rp_pen

        result = minimize(objective, self.w_mkt.copy(),
                          method="SLSQP",
                          bounds=[(0, 1)] * self.n,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                          options={"maxiter": 2000, "ftol": 1e-10})

        w   = result.x
        ret = float(mu_post @ w)
        vol = self._vol(w)
        rc  = self._rc(w)

        return {
            "weights":       dict(zip(self.names, w)),
            "return":        ret,
            "volatility":    vol,
            "sharpe":        (ret - self.rf) / vol if vol > 0 else 0,
            "rc_pct":        dict(zip(self.names, rc / vol * 100 if vol > 0 else np.zeros(self.n))),
            "posterior_mu":  dict(zip(self.names, mu_post)),
        }
