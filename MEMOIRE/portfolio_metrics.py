"""
Métriques avancées de portefeuille: Sortino, Information Ratio, Maximum Drawdown, etc.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class PortfolioMetrics:
    """Calcule des métriques avancées pour un portefeuille."""
    
    def __init__(self, returns_data, portfolio_weights, risk_free_rate=0.02):
        """
        Args:
            returns_data: DataFrame avec les rendements historiques
            portfolio_weights: dict ou array avec les poids du portefeuille
            risk_free_rate: Taux sans risque
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        
        # Convertir les poids en array
        if isinstance(portfolio_weights, dict):
            self.weights = np.array([portfolio_weights[col] for col in returns_data.columns])
        else:
            self.weights = np.array(portfolio_weights)
        
        # Calculer les rendements du portefeuille
        self.portfolio_returns = (returns_data * self.weights).sum(axis=1)
        
        # Statistiques de base
        self.num_periods = len(returns_data)
        self.num_assets = returns_data.shape[1]
    
    def annual_return(self):
        """Rendement annualisé (252 jours de trading)."""
        return self.portfolio_returns.mean() * 252
    
    def annual_volatility(self):
        """Volatilité annualisée."""
        return self.portfolio_returns.std() * np.sqrt(252)
    
    def sharpe_ratio(self):
        """Ratio de Sharpe."""
        excess_return = self.annual_return() - self.risk_free_rate
        annual_vol = self.annual_volatility()
        return excess_return / annual_vol if annual_vol > 0 else 0
    
    def sortino_ratio(self, target_return=None):
        """
        Ratio de Sortino: comme Sharpe mais seulement pénalise la volatilité à la baisse.
        
        Args:
            target_return: Rendement cible (default: risk_free_rate)
        """
        if target_return is None:
            target_return = self.risk_free_rate
        
        excess_returns = self.portfolio_returns - (target_return / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        downside_volatility = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        excess_annual = self.annual_return() - target_return
        return excess_annual / downside_volatility if downside_volatility > 0 else 0
    
    def calmar_ratio(self):
        """
        Ratio de Calmar: rendement annuel / max drawdown absolu.
        Mesure rendement par unité de risque de baisse.
        """
        max_dd = abs(self.max_drawdown())
        annual_ret = self.annual_return()
        return annual_ret / max_dd if max_dd > 0 else 0
    
    def max_drawdown(self):
        """
        Maximum Drawdown: plus grande baisse cumulée du portefeuille.
        Exprimé en négatif (ex: -0.25 = -25%).
        """
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def recovery_time(self):
        """
        Temps moyen (en jours) pour se rétablir après le max drawdown.
        """
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        min_idx = drawdown.idxmin()
        min_value = drawdown.loc[min_idx]
        
        # Trouver quand le portefeuille retrouve le même niveau
        recovery_dates = cumulative_returns[cumulative_returns.index > min_idx][
            cumulative_returns >= cumulative_returns.loc[min_idx]
        ]
        
        if len(recovery_dates) > 0:
            recovery_idx = recovery_dates.index[0]
            days_to_recovery = (recovery_idx - min_idx).days
            return days_to_recovery
        else:
            return np.nan
    
    def var_95(self):
        """Value at Risk 95%: perte maximale avec 95% de confiance."""
        return np.percentile(self.portfolio_returns, 5)
    
    def cvar_95(self):
        """Conditional Value at Risk 95%: moyenne des 5% pires jours."""
        var_95 = self.var_95()
        return self.portfolio_returns[self.portfolio_returns <= var_95].mean()
    
    def skewness(self):
        """Asymétrie: mesure la dissymétrie de la distribution."""
        return skew(self.portfolio_returns)
    
    def kurtosis(self):
        """Kurtosis: mesure les queues épaisses (risque de crash)."""
        return kurtosis(self.portfolio_returns)
    
    def win_rate(self):
        """Pourcentage de jours avec rendement positif."""
        positive_days = (self.portfolio_returns > 0).sum()
        return positive_days / len(self.portfolio_returns) if len(self.portfolio_returns) > 0 else 0
    
    def best_day(self):
        """Meilleur jour."""
        return self.portfolio_returns.max()
    
    def worst_day(self):
        """Pire jour."""
        return self.portfolio_returns.min()
    
    def best_month(self):
        """Meilleur mois."""
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.max()
    
    def worst_month(self):
        """Pire mois."""
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.min()
    
    def information_ratio(self, benchmark_returns):
        """
        Information Ratio: rendement excédentaire vs benchmark / tracking error.
        
        Args:
            benchmark_returns: Series avec les rendements du benchmark
        """
        excess_returns = self.portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        excess_return_annual = excess_returns.mean() * 252
        return excess_return_annual / tracking_error if tracking_error > 0 else 0
    
    def get_all_metrics(self, benchmark_returns=None):
        """Retourne toutes les métriques dans un dictionnaire."""
        metrics = {
            'Annual Return': self.annual_return(),
            'Annual Volatility': self.annual_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Recovery Time (days)': self.recovery_time(),
            'VaR 95%': self.var_95(),
            'CVaR 95%': self.cvar_95(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            'Win Rate': self.win_rate(),
            'Best Day': self.best_day(),
            'Worst Day': self.worst_day(),
            'Best Month': self.best_month(),
            'Worst Month': self.worst_month()
        }
        
        if benchmark_returns is not None:
            metrics['Information Ratio'] = self.information_ratio(benchmark_returns)
        
        return metrics
    
    def print_metrics(self, benchmark_returns=None):
        """Affiche les metriques de maniere formatee."""
        metrics = self.get_all_metrics(benchmark_returns)
        
        print("\n" + "=" * 70)
        print("METRIQUES AVANCEES DU PORTEFEUILLE")
        print("=" * 70)
        
        print("\nRendement & Risque:")
        print(f"  Rendement annualise: {metrics['Annual Return']:>12.4%}")
        print(f"  Volatilite annualisee: {metrics['Annual Volatility']:>11.4%}")
        print(f"  Win Rate: {metrics['Win Rate']:>28.2%}")
        
        print("\nRatios:")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:>20.4f}")
        print(f"  Sortino Ratio: {metrics['Sortino Ratio']:>19.4f}")
        print(f"  Calmar Ratio: {metrics['Calmar Ratio']:>20.4f}")
        if 'Information Ratio' in metrics:
            print(f"  Information Ratio: {metrics['Information Ratio']:>16.4f}")
        
        print("\nRisque de baisse:")
        print(f"  Max Drawdown: {metrics['Max Drawdown']:>22.4%}")
        print(f"  Recovery Time: {metrics['Recovery Time (days)']:>19.0f} jours")
        print(f"  VaR 95%: {metrics['VaR 95%']:>29.4%}")
        print(f"  CVaR 95%: {metrics['CVaR 95%']:>28.4%}")
        
        print("\nDistribution:")
        print(f"  Skewness: {metrics['Skewness']:>27.4f}")
        print(f"  Kurtosis: {metrics['Kurtosis']:>27.4f}")
        
        print("\nExtremes:")
        print(f"  Meilleur jour: {metrics['Best Day']:>21.4%}")
        print(f"  Pire jour: {metrics['Worst Day']:>25.4%}")
        print(f"  Meilleur mois: {metrics['Best Month']:>21.4%}")
        print(f"  Pire mois: {metrics['Worst Month']:>25.4%}")


def compare_portfolio_metrics(results_dict, returns_data):
    """
    Compare les métriques de plusieurs portefeuilles.
    
    Args:
        results_dict: dict avec {nom_stratégie: {'weights': ..., ...}}
        returns_data: DataFrame avec les rendements historiques
    
    Returns:
        DataFrame avec toutes les métriques comparées
    """
    metrics_list = []
    
    for strategy_name, result in results_dict.items():
        weights = result.get('weights', {})
        
        # Convertir en array
        if isinstance(weights, dict):
            weights_array = np.array([weights.get(col, 0) for col in returns_data.columns])
        else:
            weights_array = np.array(weights)
        
        # Calculer les métriques
        metrics_calc = PortfolioMetrics(returns_data, weights_array)
        metrics = metrics_calc.get_all_metrics()
        
        metrics['Strategy'] = strategy_name
        metrics_list.append(metrics)
    
    # Créer DataFrame
    df_metrics = pd.DataFrame(metrics_list)
    
    # Réorganiser les colonnes
    cols = df_metrics.columns.tolist()
    cols.remove('Strategy')
    df_metrics = df_metrics[['Strategy'] + sorted(cols)]
    
    return df_metrics


def main():
    """Exemple d'utilisation."""
    np.random.seed(42)
    
    # Données simulées
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    returns_data = pd.DataFrame(
        np.random.randn(252, 5) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Portefeuille égal poids
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Calculer les métriques
    metrics = PortfolioMetrics(returns_data, weights)
    metrics.print_metrics()
    
    # Comparaison de plusieurs portefeuilles
    print("\n" + "=" * 70)
    print("COMPARAISON MULTI-PORTEFEUILLE")
    print("=" * 70)
    
    results = {
        'Equal Weight': {'weights': {asset: 0.2 for asset in assets}},
        'Risk Parity': {'weights': {asset: 0.25 for asset in assets[:4]} | {assets[-1]: 0}},
        'Tech Heavy': {'weights': {asset: 0.4 for asset in assets[:2]} | {asset: 0.2 for asset in assets[2:]}},
    }
    
    df_comparison = compare_portfolio_metrics(results, returns_data)
    print("\n" + df_comparison.to_string())


if __name__ == '__main__':
    main()
