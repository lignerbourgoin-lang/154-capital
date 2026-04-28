import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cvxpy as cp

from config import TICKERS, LAYOUT, CONFIG, CHARTS_QUANTITATIVE, DATA_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(fig: go.Figure, path: Path) -> None:
    fig.write_html(str(path), config=CONFIG, include_plotlyjs="cdn")


def _write_html(title: str, body: str, path: Path) -> None:
    css = """
    <style>
      body  { background:#282828; color:#e0e0e0;
              font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
              padding:24px; margin:0; }
      h2    { color:#aaaaff; margin-bottom:16px; font-size:0.95rem;
              letter-spacing:.1em; text-transform:uppercase; }
      table { width:100%; border-collapse:collapse; font-size:0.82rem; }
      th    { text-align:left; padding:8px 12px; border-bottom:2px solid #444;
              color:#7b9cff; font-weight:normal; letter-spacing:.08em; }
      td    { padding:8px 12px; border-bottom:1px solid #333; }
      .val  { text-align:right; font-variant-numeric:tabular-nums; }
      .pos  { color:#adff2f; }
      .neg  { color:#ff4c5e; }
      .neu  { color:#e0e0e0; }
    </style>
    """
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body><h2>{title}</h2>{body}</body></html>"
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MarkowitzOptimizer  (adapted from basic_optim.py — same logic, Plotly output)
# ---------------------------------------------------------------------------

class MarkowitzOptimizer:

    def __init__(self, returns: pd.DataFrame):
        self.returns      = returns
        self.names        = returns.columns.tolist()
        self.mean_returns = returns.mean().values
        cov               = returns.cov().values
        self.cov_matrix   = (cov + cov.T) / 2
        self.n            = len(self.names)

    def min_variance(self) -> dict:
        w          = cp.Variable(self.n)
        objective  = cp.Minimize(cp.quad_form(w, self.cov_matrix, assume_PSD=True))
        problem    = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
        problem.solve(solver=cp.CLARABEL)
        weights    = w.value
        return {
            "weights":    weights,
            "rendement":  float(self.mean_returns @ weights),
            "volatilite": float(np.sqrt(weights @ self.cov_matrix @ weights)),
        }

    def max_sharpe(self, rf: float = 0.0) -> dict:
        y          = cp.Variable(self.n, nonneg=True)
        kappa      = cp.Variable(nonneg=True)
        objective  = cp.Maximize((self.mean_returns - rf) @ y)
        constraints = [
            cp.sum(y) == kappa,
            cp.quad_form(y, self.cov_matrix, assume_PSD=True) <= 1,
        ]
        problem    = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        weights    = y.value / kappa.value
        rendement  = float(self.mean_returns @ weights)
        volatilite = float(np.sqrt(weights @ self.cov_matrix @ weights))
        return {
            "weights":    weights,
            "rendement":  rendement,
            "volatilite": volatilite,
            "sharpe":     (rendement - rf) / volatilite,
        }

    def efficient_frontier(self, n_points: int = 50) -> dict:
        w              = cp.Variable(self.n)
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), n_points)
        volatilities, returns_out, all_weights = [], [], []

        for target in target_returns:
            objective  = cp.Minimize(cp.quad_form(w, self.cov_matrix, assume_PSD=True))
            problem    = cp.Problem(objective, [cp.sum(w) == 1, w >= 0, self.mean_returns @ w == target])
            problem.solve(solver=cp.CLARABEL)
            if problem.status == "optimal":
                volatilities.append(float(np.sqrt(problem.value)))
                returns_out.append(float(target))
                all_weights.append(w.value)

        return {"volatilities": volatilities, "returns": returns_out, "weights": all_weights}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_pie(result: dict, names: list[str], label: str, filename: str) -> None:
    weights = np.array(result["weights"])
    mask    = weights > 0.001

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(
        title_text    = label,
        paper_bgcolor = "rgb(40,40,40)",
        plot_bgcolor  = "rgb(40,40,40)",
        legend        = dict(orientation="v", x=1.02, y=0.5),
    )
    fig.add_pie(
        labels    = [n for n, m in zip(names, mask) if m],
        values    = [round(w * 100, 2) for w, m in zip(weights, mask) if m],
        hole      = 0.35,
        textinfo  = "label+percent",
        hovertemplate = "%{label}<br>%{value:.2f}%<extra></extra>",
    )
    _write(fig, CHARTS_QUANTITATIVE / filename)


def build_frontier(frontier: dict, mv: dict, ms: dict, names: list[str]) -> None:
    vols = [v * 100 for v in frontier["volatilities"]]
    rets = [r * 100 for r in frontier["returns"]]

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(
        title_text  = "Frontière efficiente — Defence Basket",
        xaxis_title = "Volatilité journalière (%)",
        yaxis_title = "Rendement journalier (%)",
        hovermode   = "closest",
    )

    # Frontier curve
    fig.add_scatter(x=vols, y=rets, mode="lines", name="Frontière efficiente",
                    line=dict(color="#7b2cff", width=2))

    # Min variance point
    fig.add_scatter(
        x=[mv["volatilite"] * 100], y=[mv["rendement"] * 100],
        mode="markers+text", name="Variance minimale",
        text=["Min Var"], textposition="top right",
        marker=dict(size=12, color="limegreen", symbol="diamond"),
    )

    # Max sharpe point
    fig.add_scatter(
        x=[ms["volatilite"] * 100], y=[ms["rendement"] * 100],
        mode="markers+text", name="Sharpe maximum",
        text=["Max Sharpe"], textposition="top right",
        marker=dict(size=12, color="#ff9f1c", symbol="star"),
    )

    _write(fig, CHARTS_QUANTITATIVE / "optim_frontier.html")


def build_metrics(mv: dict, ms: dict, names: list[str]) -> None:

    def _color(v: float) -> str:
        return "pos" if v >= 0 else "neg"

    def _row(label: str, val_mv: str, val_ms: str, cls: str = "neu") -> str:
        return (f"<tr><td>{label}</td>"
                f"<td class='val {cls}'>{val_mv}</td>"
                f"<td class='val {cls}'>{val_ms}</td></tr>")

    ret_mv  = mv["rendement"]  * 100
    ret_ms  = ms["rendement"]  * 100
    vol_mv  = mv["volatilite"] * 100
    vol_ms  = ms["volatilite"] * 100
    sh_mv   = ret_mv / vol_mv if vol_mv else 0
    sh_ms   = ms["sharpe"]
    ann_mv  = sh_mv  * np.sqrt(252)
    ann_ms  = sh_ms  * np.sqrt(252)

    rows = [
        _row("Rendement journalier",  f"{ret_mv:+.4f}%",  f"{ret_ms:+.4f}%",  _color(ret_mv)),
        _row("Volatilité journalière", f"{vol_mv:.4f}%",   f"{vol_ms:.4f}%",   "neu"),
        _row("Sharpe (journalier)",   f"{sh_mv:.4f}",     f"{sh_ms:.4f}",     "neu"),
        _row("Sharpe (annualisé)",    f"{ann_mv:.4f}",    f"{ann_ms:.4f}",    "neu"),
    ]

    # Weights table
    weight_rows = "".join(
        f"<tr><td>{n}</td>"
        f"<td class='val'>{mv['weights'][i]*100:.1f}%</td>"
        f"<td class='val'>{ms['weights'][i]*100:.1f}%</td></tr>"
        for i, n in enumerate(names)
    )

    body = f"""
    <table>
      <tr><th>Métrique</th><th>Variance min</th><th>Sharpe max</th></tr>
      {"".join(rows)}
    </table>
    <br>
    <table>
      <tr><th>Entreprise</th><th>Poids (Min Var)</th><th>Poids (Max Sharpe)</th></tr>
      {weight_rows}
    </table>
    """
    _write_html("Métriques d'optimisation — Defence Basket", body,
                CHARTS_QUANTITATIVE / "optim_metrics.html")


def build_nav_optimised(returns: pd.DataFrame, mv: dict, ms: dict) -> None:
    """Portfolio cumulative return: Min Var vs Max Sharpe vs Equal Weight."""
    eq_weights = np.ones(len(returns.columns)) / len(returns.columns)

    def _cum(weights):
        port_ret = (returns * weights).sum(axis=1)
        return (1 + port_ret).cumprod() * 100

    df = pd.DataFrame({
        "Variance minimale": _cum(mv["weights"]),
        "Sharpe maximum":    _cum(ms["weights"]),
        "Equal weight":      _cum(eq_weights),
    })

    fig = go.Figure(layout=LAYOUT)
    fig.update_layout(
        title_text  = "NAV base 100 — Portfolios optimisés (dernier mois)",
        yaxis_title = "Base 100",
        hovermode   = "x unified",
    )
    for col in df.columns:
        dash = "dash" if col == "Equal weight" else "solid"
        fig.add_scatter(x=df.index, y=df[col], name=col, line=dict(dash=dash))

    _write(fig, CHARTS_QUANTITATIVE / "nav_optimised.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    prices = pd.read_excel(DATA_DIR / "prices.xlsx", index_col=0, parse_dates=True)

    # Last ~30 trading days
    returns = prices.pct_change(fill_method=None).dropna().iloc[-30:]

    print(f"  Optimisation — {len(returns)} jours ({returns.index[0].date()} to {returns.index[-1].date()})")

    names     = list(returns.columns)
    optimizer = MarkowitzOptimizer(returns)

    print("  Variance minimale ...")
    mv = optimizer.min_variance()

    print("  Sharpe maximum ...")
    ms = optimizer.max_sharpe()

    print("  Frontiere efficiente ...")
    frontier = optimizer.efficient_frontier()

    build_pie(mv, names, "Allocation — Variance minimale", "optim_pie_minvar.html")
    build_pie(ms, names, "Allocation — Sharpe maximum",    "optim_pie_sharpe.html")
    build_frontier(frontier, mv, ms, names)
    build_metrics(mv, ms, names)
    build_nav_optimised(returns, mv, ms)

    print("  Done — optimisation charts written.")


if __name__ == "__main__":
    run()
