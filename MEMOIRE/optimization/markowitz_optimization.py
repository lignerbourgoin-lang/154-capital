import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class MarkowitzOptimizer:
    """Optimiseur de portefeuille selon la théorie de Markowitz."""
    
    def __init__(self, returns_data):
        """
        Initialise l'optimiseur avec les données de rendements.
        
        Args:
            returns_data: DataFrame pandas avec les rendements des actifs (colonnes = actifs, lignes = périodes)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
    
    def portfolio_performance(self, weights):
        """Calcule le rendement et la volatilité du portefeuille."""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
    
    def negative_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """Calcule le ratio de Sharpe négatif (pour minimisation)."""
        p_return, p_volatility = self.portfolio_performance(weights)
        return -(p_return - risk_free_rate) / p_volatility
    
    def minimize_variance(self, weights):
        """Calcule la volatilité (pour minimisation de risque)."""
        return self.portfolio_performance(weights)[1]
    
    def optimize_portfolio(self, optimization_type='sharpe', risk_free_rate=0.02):
        """
        Optimise le portefeuille.
        
        Args:
            optimization_type: 'sharpe' (max Sharpe) ou 'variance' (min variance)
            risk_free_rate: Taux sans risque pour le ratio de Sharpe
            
        Returns:
            dict avec les poids optimaux et les performances
        """
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        if optimization_type == 'sharpe':
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(risk_free_rate,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:  # variance minimale
            result = minimize(
                self.minimize_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        opt_return, opt_volatility = self.portfolio_performance(result.x)
        opt_sharpe = (opt_return - risk_free_rate) / opt_volatility
        
        return {
            'weights': dict(zip(self.returns.columns, result.x)),
            'return': opt_return,
            'volatility': opt_volatility,
            'sharpe_ratio': opt_sharpe
        }
    
    def efficient_frontier(self, risk_free_rate=0.02, num_points=100):
        """Génère la frontière efficiente."""
        min_vol = self.minimize_variance(np.array([1 / self.num_assets] * self.num_assets))
        max_return = self.mean_returns.max()
        
        target_returns = np.linspace(min_vol, max_return, num_points)
        frontier_volatility = []
        frontier_returns = []
        
        for target_return in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
            )
            bounds = tuple((0, 1) for _ in range(self.num_assets))
            initial_weights = np.array([1 / self.num_assets] * self.num_assets)
            
            result = minimize(
                self.minimize_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                _, volatility = self.portfolio_performance(result.x)
                frontier_volatility.append(volatility)
                frontier_returns.append(target_return)
        
        return frontier_returns, frontier_volatility


def main():
    """Exemple d'utilisation."""
    np.random.seed(42)
    
    # Génération de données de rendement simulées
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    returns_data = pd.DataFrame(
        np.random.randn(252, 4) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Création de l'optimiseur
    optimizer = MarkowitzOptimizer(returns_data)
    
    # Optimisation pour maximum Sharpe
    optimal_portfolio = optimizer.optimize_portfolio(optimization_type='sharpe')
    print("Portefeuille optimal (Max Sharpe):")
    print(f"  Rendement: {optimal_portfolio['return']:.4f}")
    print(f"  Volatilité: {optimal_portfolio['volatility']:.4f}")
    print(f"  Ratio Sharpe: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("  Poids:")
    for asset, weight in optimal_portfolio['weights'].items():
        print(f"    {asset}: {weight:.4f}")
    
    # Optimisation pour variance minimale
    min_var_portfolio = optimizer.optimize_portfolio(optimization_type='variance')
    print("\nPortefeuille de variance minimale:")
    print(f"  Rendement: {min_var_portfolio['return']:.4f}")
    print(f"  Volatilité: {min_var_portfolio['volatility']:.4f}")
    print(f"  Ratio Sharpe: {min_var_portfolio['sharpe_ratio']:.4f}")
    
    # Génération et affichage de la frontière efficiente
    frontier_returns, frontier_volatility = optimizer.efficient_frontier()
    
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_volatility, frontier_returns, 'b-', label='Frontière Efficiente')
    plt.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                color='red', s=200, marker='*', label='Max Sharpe', zorder=5)
    plt.scatter(min_var_portfolio['volatility'], min_var_portfolio['return'], 
                color='green', s=100, marker='o', label='Min Variance', zorder=5)
    plt.xlabel('Volatilité (Écart-type)')
    plt.ylabel('Rendement attendu')
    plt.title('Frontière Efficiente de Markowitz')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    print("\nGraphique sauvegardé: efficient_frontier.png")


if __name__ == '__main__':
    main()
