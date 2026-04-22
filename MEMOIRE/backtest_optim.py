import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from basic_optim import univers, MarkowitzOptimizer

Path = r"C:\Eliott\154-capital\MEMOIRE\DATA\univers.xlsx"

WINDOW = 252      # fenêtre d'estimation : 1 an
REBAL  = 21       # rebalancement : mensuel


def min_variance_weights(returns_window):
    opt = MarkowitzOptimizer(returns_window)
    result = opt.optimization()
    return result['weights']


def equal_weights(n):
    return np.ones(n) / n


def run_backtest(returns):
    dates = returns.index
    n = returns.shape[1]

    records = []

    for i in range(WINDOW, len(dates), REBAL):
        estimation = returns.iloc[i - WINDOW:i]
        hold_end   = min(i + REBAL, len(dates))
        oos        = returns.iloc[i:hold_end]

        if oos.empty:
            break

        w_mv = min_variance_weights(estimation)
        w_ew = equal_weights(n)

        for j, date in enumerate(oos.index):
            r = oos.iloc[j].values
            records.append({
                'date': date,
                'MV':   float(w_mv @ r),
                'EW':   float(w_ew @ r),
            })

    df = pd.DataFrame(records).set_index('date')
    cumul = (1 + df).cumprod()
    return df, cumul


def metrics(returns_series):
    r   = returns_series.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol
    cumul   = (1 + r).cumprod()
    drawdown = cumul / cumul.cummax() - 1
    max_dd  = drawdown.min()
    return {
        'Rendement annualisé': f"{ann_ret*100:.2f}%",
        'Volatilité annualisée': f"{ann_vol*100:.2f}%",
        'Sharpe':               f"{sharpe:.2f}",
        'Max Drawdown':         f"{max_dd*100:.2f}%",
    }


if __name__ == "__main__":
    print("Chargement des données...")
    assets, dates, returns = univers(Path)
    print(f"  {len(assets)} actifs | {dates[0].date()} → {dates[-1].date()}")

    print("Backtest en cours...")
    daily_ret, cumul = run_backtest(returns)
    print("  Terminé.")

    print("\n--- Métriques ---")
    for strat in ['MV', 'EW']:
        print(f"\n{strat}")
        for k, v in metrics(daily_ret[strat]).items():
            print(f"  {k}: {v}")

    cumul.plot(figsize=(12, 6), title="Backtest — Minimum Variance vs Equally Weighted (2015–2026)")
    plt.xlabel("Date")
    plt.ylabel("Valeur cumulée (base 1)")
    plt.legend(["Min Variance", "Equally Weighted"])
    plt.tight_layout()
    plt.show()
