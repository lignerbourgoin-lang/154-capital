"""
Script principal d'exécution complète des 3 stratégies:
1. Risk Parity
2. Black-Litterman
3. Hybrid (Risk Parity + Black-Litterman)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from risk_parity_optimizer import RiskParityOptimizer
from black_litterman_optimizer import BlackLittermanOptimizer
from hybrid_optimizer import HybridRiskParityBlackLitterman
from portfolio_metrics import PortfolioMetrics, compare_portfolio_metrics
from visualization import PortfolioVisualizer


def load_universe_from_excel(file_path):
    """Charge l'univers d'actifs depuis le fichier Excel."""
    print(f"\nChargement des donnees depuis: {file_path}")
    
    # Lire le fichier Excel
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Convertir la date en index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Prendre les dernieres donnees disponibles (moins de NaN)
    # Supprimer les colonnes avec trop de NaN
    df = df.dropna(axis=1, thresh=len(df) * 0.7)  # Garder si >70% de donnees
    
    # Calculer les rendements quotidiens
    returns = df.pct_change().dropna()
    
    print(f"[OK] Donnees chargees: {len(returns)} jours de trading")
    print(f"[OK] Nombre d'actifs: {len(returns.columns)}")
    print(f"[OK] Periode: {returns.index[0].date()} a {returns.index[-1].date()}")
    
    return returns


def download_real_data(tickers, period='1y'):
    """Télécharge les données réelles depuis Yahoo Finance."""
    print(f"\nTéléchargement des données pour {', '.join(tickers)}...")
    data = yf.download(tickers, period=period, progress=False)['Adj Close']
    
    # Calculer les rendements quotidiens
    returns = data.pct_change().dropna()
    
    print(f"Données téléchargées: {len(returns)} jours de trading")
    return returns


def create_synthetic_data(num_assets=5, num_periods=252):
    """Crée des données synthétiques pour les tests."""
    np.random.seed(42)
    
    assets = [f'ASSET_{i+1}' for i in range(num_assets)]
    returns_data = pd.DataFrame(
        np.random.randn(num_periods, num_assets) * 0.02 + 0.0005,
        columns=assets,
        index=pd.date_range('2023-01-01', periods=num_periods)
    )
    
    return returns_data


def run_risk_parity_analysis(returns_data):
    """Exécute l'analyse Risk Parity."""
    print("\n" + "=" * 80)
    print("1. RISK PARITY ANALYSIS")
    print("=" * 80)
    
    optimizer = RiskParityOptimizer(returns_data)
    result = optimizer.optimize_risk_parity()
    
    print(f"\nOptimisation reussie: {result['optimizer_success']}")
    print(f"  Rendement: {result['return']:.4%}")
    print(f"  Volatilite: {result['volatility']:.4%}")
    print(f"  Ratio Sharpe: {result['sharpe_ratio']:.4f}")
    
    print("\n  Poids du portefeuille:")
    for asset, weight in result['weights'].items():
        print(f"    {asset}: {weight:.4%}")
    
    print("\n  Contribution au risque (should be ~equal):")
    weights_array = np.array([result['weights'][asset] for asset in returns_data.columns])
    analysis = optimizer.get_risk_parity_contribution_analysis(weights_array)
    for _, row in analysis.iterrows():
        print(f"    {row['Asset']}: {row['Risk_Contribution_%']:.2f}% (target: {100/len(returns_data.columns):.2f}%)")
    
    # Comparaison avec Equal Weight
    comparison, _ = optimizer.compare_with_equal_weight()
    print("\n  Comparaison vs Equal Weight:")
    for _, row in comparison.iterrows():
        print(f"    {row['Metric']}: RP={row['Risk_Parity']:.4f}, EW={row['Equal_Weight']:.4f}")
    
    return result, optimizer


def run_black_litterman_analysis(returns_data):
    """Exécute l'analyse Black-Litterman."""
    print("\n" + "=" * 80)
    print("2. BLACK-LITTERMAN ANALYSIS")
    print("=" * 80)
    
    optimizer = BlackLittermanOptimizer(returns_data, risk_free_rate=0.02)
    
    print("\n  Rendements implicites d'équilibre (reverse optimization):")
    for asset, ret in zip(returns_data.columns, optimizer.implied_returns):
        print(f"    {asset}: {ret:.4%}")
    
    # Définir des views
    assets_list = returns_data.columns.tolist()
    views = {}
    
    # View 1: Optimistic sur les premiers 2 actifs
    views['bullish_view'] = {
        'type': 'absolute',
        'assets': [assets_list[0], assets_list[1]],
        'expected_return': 0.08,
        'confidence': 0.85
    }
    
    # View 2: Pessimistic sur le 3e actif
    views['bearish_view'] = {
        'type': 'absolute',
        'assets': [assets_list[2]],
        'expected_return': 0.02,
        'confidence': 0.6
    }
    
    # View 3: Neutral sur les autres
    if len(assets_list) > 3:
        views['neutral_view'] = {
            'type': 'absolute',
            'assets': assets_list[3:],
            'expected_return': 0.04,
            'confidence': 0.5
        }
    
    optimizer.add_views(views)
    
    print("\n  Views ajoutées:")
    for view in optimizer.views_list:
        print(f"    {view['name']}: confiance={view['confidence']:.1%}")
    
    # Optimisation
    result = optimizer.full_optimization(tau=0.05, optimization_type='sharpe')
    
    print(f"\nOptimisation reussie: {result['success']}")
    print(f"  Rendement: {result['return']:.4%}")
    print(f"  Volatilite: {result['volatility']:.4%}")
    print(f"  Ratio Sharpe: {result['sharpe_ratio']:.4f}")
    
    print("\n  Poids du portefeuille:")
    for asset, weight in result['weights'].items():
        print(f"    {asset}: {weight:.4%}")
    
    print("\n  Comparaison des rendements:")
    print("    Asset | Historical | Implied | Posterior")
    for asset in returns_data.columns:
        hist = result['historical_returns'][asset]
        impl = result['implied_returns'][asset]
        post = result['posterior_returns'][asset]
        print(f"    {asset:6} | {hist:10.4%} | {impl:7.4%} | {post:9.4%}")
    
    return result, optimizer


def run_hybrid_analysis(returns_data):
    """Exécute l'analyse Hybrid (Risk Parity + Black-Litterman)."""
    print("\n" + "=" * 80)
    print("3. HYBRID OPTIMIZATION (Risk Parity + Black-Litterman)")
    print("=" * 80)
    
    optimizer = HybridRiskParityBlackLitterman(returns_data, risk_free_rate=0.02)
    
    # Ajouter les mêmes views que B-L
    assets_list = returns_data.columns.tolist()
    views = {}
    views['bullish_view'] = {
        'type': 'absolute',
        'assets': [assets_list[0], assets_list[1]],
        'expected_return': 0.08,
        'confidence': 0.85
    }
    views['bearish_view'] = {
        'type': 'absolute',
        'assets': [assets_list[2]],
        'expected_return': 0.02,
        'confidence': 0.6
    }
    if len(assets_list) > 3:
        views['neutral_view'] = {
            'type': 'absolute',
            'assets': assets_list[3:],
            'expected_return': 0.04,
            'confidence': 0.5
        }
    
    optimizer.add_views(views)
    
    # Optimiser avec différents poids de pénalité
    print("\n  Optimisation avec balance Risk Parity / Sharpe Ratio:")
    
    results = {}
    for penalty in [10, 100, 1000]:
        result = optimizer.optimize_hybrid(tau=0.05, penalty_weight=penalty)
        results[penalty] = result
        
        print(f"\n    Penalty={penalty}:")
        print(f"      Rendement: {result['return']:.4%}")
        print(f"      Volatilité: {result['volatility']:.4%}")
        print(f"      Ratio Sharpe: {result['sharpe_ratio']:.4f}")
        
        # Vérifier la déviation Risk Parity
        rc_pcts = list(result['risk_contributions_pct'].values())
        target = 100 / len(returns_data.columns)
        avg_deviation = np.mean(np.abs(np.array(rc_pcts) - target))
        print(f"      Déviation avg RC: {avg_deviation:.2f}% (lower = plus risk parity)")
    
    # Utiliser le résultat avec penalty=500 (bon équilibre)
    result = optimizer.optimize_hybrid(tau=0.05, penalty_weight=500)
    
    print("\n  RÉSULTAT FINAL (penalty=500, équilibre optimal):")
    print(f"    Rendement: {result['return']:.4%}")
    print(f"    Volatilité: {result['volatility']:.4%}")
    print(f"    Ratio Sharpe: {result['sharpe_ratio']:.4f}")
    
    print("\n    Poids:")
    for asset, weight in result['weights'].items():
        print(f"      {asset}: {weight:.4%}")
    
    print("\n    Contribution au risque:")
    print("    Asset | Weight | RC | RC_%")
    for asset in returns_data.columns:
        weight = result['weights'][asset]
        rc = result['risk_contributions'][asset]
        rc_pct = result['risk_contributions_pct'][asset]
        target = 100 / len(returns_data.columns)
        print(f"    {asset:6} | {weight:6.2%} | {rc:6.4f} | {rc_pct:6.2f}% (target: {target:.2f}%)")
    
    return result, optimizer, results


def create_comparison_table(rp_result, bl_result, hybrid_result):
    """Crée un tableau de comparaison complète."""
    print("\n" + "=" * 80)
    print("TABLEAU COMPARATIF: Risk Parity vs Black-Litterman vs Hybrid")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Strategy': ['Risk Parity', 'Black-Litterman', 'Hybrid (RP+BL)'],
        'Return': [rp_result['return'], bl_result['return'], hybrid_result['return']],
        'Volatility': [rp_result['volatility'], bl_result['volatility'], hybrid_result['volatility']],
        'Sharpe Ratio': [rp_result['sharpe_ratio'], bl_result['sharpe_ratio'], hybrid_result['sharpe_ratio']]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Ranking
    print("\n  Ranking:")
    print(f"    Best Sharpe Ratio: {comparison.loc[comparison['Sharpe Ratio'].idxmax(), 'Strategy']}")
    print(f"    Lowest Volatility: {comparison.loc[comparison['Volatility'].idxmin(), 'Strategy']}")
    print(f"    Highest Return: {comparison.loc[comparison['Return'].idxmax(), 'Strategy']}")
    
    return comparison


def main():
    """Exécution principale."""
    print("\n" + "=" * 80)
    print("OPTIMISATION DE PORTEFEUILLE: RISK PARITY + BLACK-LITTERMAN")
    print("=" * 80)
    
    # Charger l'univers d'actifs
    univers_path = r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx"
    
    try:
        returns_data = load_universe_from_excel(univers_path)
    except Exception as e:
        print(f"[ERROR] Erreur de chargement de l'univers: {e}")
        print("Utilisation de donnees synthetiques...")
        returns_data = create_synthetic_data(5, 252)
    
    # Afficher les statistiques des données
    print("\nSTATISTIQUES DES DONNEES:")
    print(f"  Periode: {returns_data.index[0].date()} a {returns_data.index[-1].date()}")
    print(f"  Nombre d'actifs: {len(returns_data.columns)}")
    print(f"  Nombre de jours: {len(returns_data)}")
    print("\n  Rendements annualises:")
    for col in returns_data.columns:
        annual_return = returns_data[col].mean() * 252
        print(f"    {col}: {annual_return:.4%}")
    
    # 1. Risk Parity
    rp_result, rp_optimizer = run_risk_parity_analysis(returns_data)
    
    # 2. Black-Litterman
    bl_result, bl_optimizer = run_black_litterman_analysis(returns_data)
    
    # 3. Hybrid
    hybrid_result, hybrid_optimizer, hybrid_variants = run_hybrid_analysis(returns_data)
    
    # Tableau comparatif
    comparison = create_comparison_table(rp_result, bl_result, hybrid_result)
    
    # MÉTRIQUES AVANCÉES
    print("\n" + "=" * 80)
    print("METRIQUES AVANCEES")
    print("=" * 80)
    
    results_for_metrics = {
        'Risk Parity': rp_result,
        'Black-Litterman': bl_result,
        'Hybrid (RP+BL)': hybrid_result
    }
    
    df_metrics = compare_portfolio_metrics(results_for_metrics, returns_data)
    print("\n" + df_metrics.to_string())
    
    # Afficher les metriques detaillees du Hybrid (meilleur)
    print("\n" + "=" * 80)
    print("DETAIL DES METRIQUES - HYBRID (Meilleure strategie)")
    print("=" * 80)
    
    hybrid_weights = np.array([hybrid_result['weights'].get(col, 0) for col in returns_data.columns])
    metrics_hybrid = PortfolioMetrics(returns_data, hybrid_weights)
    metrics_hybrid.print_metrics()
    
    # VISUALISATIONS
    print("\n" + "=" * 80)
    print("GENERATION DES VISUALISATIONS")
    print("=" * 80)
    
    visualizer = PortfolioVisualizer(figsize=(14, 8))
    visualizer.create_summary_report(returns_data, results_for_metrics, 
                                    output_dir='./portfolio_analysis')
    
    print("\n" + "=" * 80)
    print("[DONE] ANALYSE COMPLETE TERMINEE")
    print("=" * 80)
    print("\nResultats et graphiques disponibles dans le dossier: ./portfolio_analysis/")
    
    return {
        'returns_data': returns_data,
        'risk_parity': {'result': rp_result, 'optimizer': rp_optimizer},
        'black_litterman': {'result': bl_result, 'optimizer': bl_optimizer},
        'hybrid': {'result': hybrid_result, 'optimizer': hybrid_optimizer},
        'comparison': comparison,
        'metrics': df_metrics
    }


if __name__ == '__main__':
    analysis = main()
