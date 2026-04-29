import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class RiskParityOptimizer:
    """Optimiseur Risk Parity: chaque actif contribue ÉGALEMENT au risque total."""
    
    def __init__(self, returns_data):
        """
        Initialise l'optimiseur Risk Parity.
        
        Args:
            returns_data: DataFrame pandas avec les rendements des actifs (colonnes = actifs, lignes = périodes)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
    
    def portfolio_variance(self, weights):
        """Calcule la variance du portefeuille."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def portfolio_volatility(self, weights):
        """Calcule la volatilité (écart-type) du portefeuille."""
        return np.sqrt(self.portfolio_variance(weights))
    
    def marginal_contribution_to_risk(self, weights):
        """
        Calcule la contribution marginale au risque (MCR) pour chaque actif.
        MCR_i = (Cov_matrix @ weights)_i / volatilité_portfolio
        """
        portfolio_vol = self.portfolio_volatility(weights)
        if portfolio_vol == 0:
            return np.zeros(self.num_assets)
        mcr = np.dot(self.cov_matrix, weights) / portfolio_vol
        return mcr
    
    def risk_contribution(self, weights):
        """
        Calcule la contribution au risque de chaque actif.
        RC_i = poids_i * MCR_i
        """
        mcr = self.marginal_contribution_to_risk(weights)
        rc = weights * mcr
        return rc
    
    def risk_parity_objective(self, weights):
        """
        Fonction objectif à minimiser: variance de la contribution au risque.
        Objectif: chaque actif contribue pour 1/N du risque total.
        
        On minimise: sum((RC_i - risque_total/N)^2)
        """
        rc = self.risk_contribution(weights)
        portfolio_vol = self.portfolio_volatility(weights)
        target_rc = portfolio_vol / self.num_assets
        
        # Variance des contributions
        objective = np.sum((rc - target_rc) ** 2)
        return objective
    
    def optimize_risk_parity(self):
        """
        Optimise le portefeuille selon la stratégie Risk Parity.
        
        Returns:
            dict avec les poids Risk Parity et les performances
        """
        # Initialisation avec poids égaux
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # Contraintes: somme des poids = 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bornes: poids entre 0 et 1 (pas de short selling)
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Optimisation
        result = minimize(
            self.risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        optimal_weights = result.x
        
        # Calculs de performance
        portfolio_vol = self.portfolio_volatility(optimal_weights)
        portfolio_return = np.sum(self.mean_returns * optimal_weights)
        rc = self.risk_contribution(optimal_weights)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'risk_contributions': dict(zip(self.asset_names, rc)),
            'total_risk_contribution': np.sum(rc),
            'optimizer_success': result.success,
            'optimizer_message': result.message
        }
    
    def get_risk_parity_contribution_analysis(self, weights):
        """Analyse détaillée des contributions au risque."""
        rc = self.risk_contribution(weights)
        portfolio_vol = self.portfolio_volatility(weights)
        target_rc = portfolio_vol / self.num_assets
        
        analysis = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': weights,
            'Risk_Contribution': rc,
            'Risk_Contribution_%': (rc / portfolio_vol) * 100 if portfolio_vol > 0 else 0,
            'Target_RC_%': (100 / self.num_assets),
            'Deviation_from_target': rc - target_rc
        })
        
        return analysis
    
    def compare_with_equal_weight(self):
        """Compare Risk Parity avec un portefeuille égal poids (1/N)."""
        equal_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # Risk Parity
        rp_result = self.optimize_risk_parity()
        rp_weights = np.array([rp_result['weights'][asset] for asset in self.asset_names])
        
        # Comparaison
        ew_vol = self.portfolio_volatility(equal_weights)
        ew_return = np.sum(self.mean_returns * equal_weights)
        ew_sharpe = (ew_return - 0.02) / ew_vol if ew_vol > 0 else 0
        
        comparison = pd.DataFrame({
            'Metric': ['Rendement', 'Volatilité', 'Ratio Sharpe'],
            'Equal_Weight': [ew_return, ew_vol, ew_sharpe],
            'Risk_Parity': [rp_result['return'], rp_result['volatility'], rp_result['sharpe_ratio']],
            'Difference': [
                rp_result['return'] - ew_return,
                rp_result['volatility'] - ew_vol,
                rp_result['sharpe_ratio'] - ew_sharpe
            ]
        })
        
        return comparison, rp_result


def main():
    """Exemple d'utilisation de Risk Parity."""
    np.random.seed(42)
    
    # Simulation de données
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    returns_data = pd.DataFrame(
        np.random.randn(252, 5) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Créer l'optimiseur
    optimizer = RiskParityOptimizer(returns_data)
    
    # Optimiser
    rp_portfolio = optimizer.optimize_risk_parity()
    
    print("=" * 60)
    print("RISK PARITY OPTIMIZATION")
    print("=" * 60)
    print(f"\nRendement: {rp_portfolio['return']:.4%}")
    print(f"Volatilite: {rp_portfolio['volatility']:.4%}")
    print(f"Ratio Sharpe: {rp_portfolio['sharpe_ratio']:.4f}")
    print(f"\nOptimisation reussie: {rp_portfolio['optimizer_success']}")
    
    print("\nPoids du portefeuille:")
    for asset, weight in rp_portfolio['weights'].items():
        print(f"  {asset}: {weight:.4%}")
    
    print("\nContribution au risque par actif:")
    print("Asset | Weight | Risk_Contribution | Risk_Contribution_% | Target_%")
    print("-" * 70)
    
    weights_array = np.array([rp_portfolio['weights'][asset] for asset in assets])
    analysis = optimizer.get_risk_parity_contribution_analysis(weights_array)
    for idx, row in analysis.iterrows():
        print(f"{row['Asset']:5} | {row['Weight']:6.2%} | {row['Risk_Contribution']:17.4f} | {row['Risk_Contribution_%']:19.2f} | {row['Target_RC_%']:7.2f}")
    
    # Comparaison avec Equal Weight
    comparison, _ = optimizer.compare_with_equal_weight()
    print("\n" + "=" * 60)
    print("COMPARAISON: RISK PARITY vs EQUAL WEIGHT")
    print("=" * 60)
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
