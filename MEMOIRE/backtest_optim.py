import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cvxpy as cp
import yfinance as yf

from basic_optim import univers, MarkowitzOptimizer

Path = r"C:\Eliott\154-capital\MEMOIRE\DATA\univers.xlsx"

WINDOW    = 252
REBAL     = 21
BENCHMARK = "SPY"

COLORS = {
    'MV':   '#2563EB',
    'EW':   '#16A34A',
    'SPY':  '#DC2626',
    'UNIV': '#F59E0B',
}


def min_variance_weights(returns_window):
    opt = MarkowitzOptimizer(returns_window)
    result = opt.optimization()
    return result['weights']


def equal_weights(n):
    return np.ones(n) / n


def load_spy(start, end):
    spy = yf.download(BENCHMARK, start=start, end=end, progress=False, auto_adjust=True)
    r = spy['Close'].pct_change().dropna()
    r.index = pd.to_datetime(r.index).tz_localize(None)
    return r


def run_backtest(returns):
    dates = returns.index
    n     = returns.shape[1]

    spy_raw   = load_spy(dates[0], dates[-1])
    spy_align = spy_raw.reindex(dates).fillna(0)

    records = []

    for i in range(WINDOW, len(dates), REBAL):
        estimation = returns.iloc[i - WINDOW:i]
        hold_end   = min(i + REBAL, len(dates))
        oos        = returns.iloc[i:hold_end]

        if oos.empty:
            break

        w_mv   = min_variance_weights(estimation)
        w_ew   = equal_weights(n)
        w_univ = equal_weights(n)  # benchmark univers = EW figé, sans rebalancement dynamique

        for j, date in enumerate(oos.index):
            r = oos.iloc[j].values
            records.append({
                'date': date,
                'MV':   float(w_mv @ r),
                'EW':   float(w_ew @ r),
                'SPY':  float(spy_align.loc[date]) if date in spy_align.index else 0.0,
                'UNIV': float(w_univ @ r),
            })

    df    = pd.DataFrame(records).set_index('date')
    cumul = (1 + df).cumprod()
    return df, cumul


def metrics(returns_series):
    r       = returns_series.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol
    cumul   = (1 + r).cumprod()
    drawdown = cumul / cumul.cummax() - 1
    max_dd  = drawdown.min()
    return {
        'Rendement annualisé':  f"{ann_ret*100:.2f}%",
        'Volatilité annualisée': f"{ann_vol*100:.2f}%",
        'Sharpe':               f"{sharpe:.2f}",
        'Max Drawdown':         f"{max_dd*100:.2f}%",
    }


def plot_matrices(returns, daily_ret):
    opt      = MarkowitzOptimizer(returns.iloc[-WINDOW:])
    res      = opt.optimization()
    w        = res['weights']
    active   = w > 1e-4
    tickers  = returns.columns[active]
    ret_sub  = returns[tickers].iloc[-WINDOW:]
    cov_mat  = ret_sub.cov() * 252
    corr_mat = ret_sub.corr()

    print(f"\n--- Actifs sélectionnés par Min Variance : {len(tickers)} / {len(returns.columns)} ---")
    for t, ww in sorted(zip(tickers, w[active]), key=lambda x: -x[1]):
        print(f"  {t:<20} {ww*100:.2f}%")

    print("\n--- Matrice de Covariance annualisée (actifs MV) ---")
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(cov_mat.to_string())

    print("\n--- Matrice de Corrélation (actifs MV) ---")
    print(corr_mat.to_string())

    # Corrélation entre stratégies
    strat_labels = {'MV': 'Min Variance', 'EW': 'Equal Weight', 'SPY': 'S&P 500', 'UNIV': 'Univers EW'}
    corr_strats  = daily_ret[['MV', 'EW', 'SPY', 'UNIV']].rename(columns=strat_labels).corr()

    print("\n--- Corrélation entre stratégies ---")
    print(corr_strats.to_string())

    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#0F172A')

    for ax, matrix, title, fmt in zip(
        axes,
        [cov_mat, corr_mat],
        ["Matrice de Covariance (actifs MV, annualisée)", "Matrice de Corrélation (actifs MV)"],
        [".3f", ".2f"],
    ):
        ax.set_facecolor('#1E293B')
        sns.heatmap(
            matrix, ax=ax, annot=len(tickers) <= 20, fmt=fmt,
            cmap='RdYlGn_r' if 'Corr' in title else 'YlOrRd',
            linewidths=0.3, linecolor='#0F172A',
            xticklabels=tickers, yticklabels=tickers,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.tick_params(colors='#CBD5E1', labelsize=7, rotation=45)

    fig.suptitle(f"Analyse des {len(tickers)} actifs sélectionnés par Min Variance", color='white', fontsize=13)
    plt.tight_layout()
    plt.show()

    # Corrélation entre stratégies

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    fig2.patch.set_facecolor('#0F172A')
    ax2.set_facecolor('#1E293B')
    import seaborn as sns
    sns.heatmap(
        corr_strats, ax=ax2, annot=True, fmt=".3f",
        cmap='RdYlGn', vmin=-1, vmax=1,
        linewidths=1, linecolor='#0F172A',
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax2.set_title("Corrélation entre stratégies", color='white', fontsize=13, pad=12)
    ax2.tick_params(colors='#CBD5E1', labelsize=10, rotation=30)
    plt.tight_layout()
    plt.show()


def plot_backtest(cumul):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor('#0F172A')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1', labelsize=9)
        ax.spines[:].set_color('#334155')

    labels = {'MV': 'Min Variance', 'EW': 'Equally Weighted', 'SPY': 'S&P 500 (SPY)', 'UNIV': 'Univers EW'}
    styles = {'MV': '-', 'EW': '--', 'SPY': '-.', 'UNIV': ':'}

    for col in ['MV', 'EW', 'SPY', 'UNIV']:
        ax1.plot(cumul.index, cumul[col], color=COLORS[col],
                 lw=1.8, ls=styles[col], label=labels[col])

    ax1.set_title("Backtest — Performance cumulée", color='white', fontsize=14, pad=12)
    ax1.set_ylabel("Valeur cumulée (base 1)", color='#94A3B8', fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=10)
    ax1.grid(color='#334155', linewidth=0.5, linestyle='--')
    ax1.axhline(1, color='#475569', lw=0.8, ls=':')

    for col in ['MV', 'EW', 'SPY', 'UNIV']:
        dd = cumul[col] / cumul[col].cummax() - 1
        ax2.fill_between(cumul.index, dd, 0, color=COLORS[col], alpha=0.3)
        ax2.plot(cumul.index, dd, color=COLORS[col], lw=1)

    ax2.set_ylabel("Drawdown", color='#94A3B8', fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax2.grid(color='#334155', linewidth=0.5, linestyle='--')
    ax2.axhline(0, color='#475569', lw=0.8)

    plt.xlabel("Date", color='#94A3B8', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Chargement des données...")
    assets, dates, returns = univers(Path)
    print(f"  {len(assets)} actifs | {dates[0].date()} → {dates[-1].date()}")

    print("Backtest en cours...")
    daily_ret, cumul = run_backtest(returns)
    print("  Terminé.")

    print("\n--- Métriques ---")
    for strat in ['MV', 'EW', 'SPY', 'UNIV']:
        print(f"\n{strat}")
        for k, v in metrics(daily_ret[strat]).items():
            print(f"  {k}: {v}")

    plot_backtest(cumul)
    plot_matrices(returns, daily_ret)
