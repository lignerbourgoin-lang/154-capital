import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt


class HybridRiskParityBlackLitterman:
    """
    Optimiseur hybride: Risk Parity + Black-Litterman.
    
    Stratégie:
    1. Utiliser les rendements postérieurs de Black-Litterman
    2. Appliquer les contraintes Risk Parity
    3. Maximiser le ratio Sharpe sous ces contraintes
    """
    
    def __init__(self, returns_data, market_weights=None, risk_free_rate=0.02):
        """
        Initialise l'optimiseur hybride.
        
        Args:
            returns_data: DataFrame avec les rendements historiques
            market_weights: dict ou array avec les poids du marché
            risk_free_rate: Taux sans risque
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
        self.risk_free_rate = risk_free_rate
        
        # Poids du marché
        if market_weights is None:
            self.market_weights = np.array([1 / self.num_assets] * self.num_assets)
        else:
            if isinstance(market_weights, dict):
                self.market_weights = np.array([market_weights.get(asset, 1/self.num_assets) 
                                                 for asset in self.asset_names])
            else:
                self.market_weights = np.array(market_weights)
        
        self.market_weights = self.market_weights / self.market_weights.sum()
        
        # Rendements implicites
        market_return = np.dot(self.market_weights, self.mean_returns)
        market_variance = np.dot(self.market_weights.T, 
                                 np.dot(self.cov_matrix, self.market_weights))
        lambda_param = (market_return - self.risk_free_rate) / market_variance
        
        self.implied_returns = self.risk_free_rate + lambda_param * np.dot(self.cov_matrix, 
                                                                             self.market_weights)
        
        # Rendements postérieurs (sera mis à jour via Black-Litterman)
        self.posterior_returns = self.implied_returns.copy()
        
        self.views_list = []
    
    def add_views(self, views_dict):
        """Ajoute les opinions de l'investisseur."""
        self.views_list = []
        
        for view_name, view_details in views_dict.items():
            view_type = view_details.get('type', 'absolute')
            assets = view_details.get('assets', [])
            expected_return = view_details.get('expected_return', 0)
            confidence = view_details.get('confidence', 0.5)
            
            p_vector = np.zeros(self.num_assets)
            for asset in assets:
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    p_vector[idx] = 1 / len(assets)
            
            self.views_list.append({
                'name': view_name,
                'p_vector': p_vector,
                'expected_return': expected_return,
                'type': view_type,
                'confidence': confidence
            })
    
    def calculate_posterior_returns(self, tau=0.025):
        """Calcule les rendements postérieurs Black-Litterman."""
        if not self.views_list:
            self.posterior_returns = self.implied_returns.copy()
            return self.posterior_returns
        
        P = np.array([view['p_vector'] for view in self.views_list])
        Q = np.array([view['expected_return'] for view in self.views_list])
        
        Omega = np.diag([
            (view['confidence'] ** -1) * tau * np.dot(view['p_vector'], 
                                                       np.dot(self.cov_matrix, view['p_vector']))
            for view in self.views_list
        ])
        
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self.cov_matrix)
        
        term1 = tau * np.dot(P, np.dot(self.cov_matrix, P.T))
        term2 = term1 + Omega
        
        try:
            inv_term2 = np.linalg.inv(term2)
        except np.linalg.LinAlgError:
            inv_term2 = np.linalg.pinv(term2)
        
        adjustment = np.dot(self.cov_matrix, np.dot(P.T, np.dot(inv_term2, 
                                                                  Q - np.dot(P, self.implied_returns))))
        
        self.posterior_returns = self.implied_returns + tau * adjustment
        
        return self.posterior_returns
    
    def portfolio_volatility(self, weights):
        """Calcule la volatilité du portefeuille."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def marginal_contribution_to_risk(self, weights):
        """MCR pour chaque actif."""
        portfolio_vol = self.portfolio_volatility(weights)
        if portfolio_vol == 0:
            return np.zeros(self.num_assets)
        mcr = np.dot(self.cov_matrix, weights) / portfolio_vol
        return mcr
    
    def risk_contribution(self, weights):
        """RC pour chaque actif."""
        mcr = self.marginal_contribution_to_risk(weights)
        rc = weights * mcr
        return rc
    
    def negative_sharpe_ratio(self, weights, expected_returns):
        """Objectif: maximiser Sharpe (minimiser -Sharpe)."""
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_vol = self.portfolio_volatility(weights)
        
        if portfolio_vol == 0:
            return 1e10
        
        return -(portfolio_return - self.risk_free_rate) / portfolio_vol
    
    def risk_parity_penalty(self, weights, penalty_weight=100):
        """
        Pénalité si les contributions au risque ne sont pas égales.
        
        Permet une optimisation "soft" du risk parity pendant qu'on maximise Sharpe.
        """
        rc = self.risk_contribution(weights)
        portfolio_vol = self.portfolio_volatility(weights)
        target_rc = portfolio_vol / self.num_assets
        
        # Variance des contributions
        deviation = np.sum((rc - target_rc) ** 2)
        
        return penalty_weight * deviation
    
    def hybrid_objective(self, weights, expected_returns, penalty_weight=100):
        """
        Objectif hybride: maximiser Sharpe + minimiser déviations Risk Parity.
        
        Args:
            weights: array des poids
            expected_returns: rendements postérieurs de B-L
            penalty_weight: poids du terme de pénalité (balance risque/return vs parity)
                - Petit (ex: 1): privilégier Sharpe ratio
                - Grand (ex: 1000): privilégier Risk Parity
        """
        sharpe_term = self.negative_sharpe_ratio(weights, expected_returns)
        rp_penalty = self.risk_parity_penalty(weights, penalty_weight)
        
        return sharpe_term + rp_penalty
    
    def optimize_hybrid(self, tau=0.025, penalty_weight=100, debug=False):
        """
        Optimise le portefeuille hybride.
        
        Args:
            tau: paramètre Black-Litterman
            penalty_weight: balance entre Sharpe ratio et Risk Parity
            debug: affiche les détails de l'optimisation
        
        Returns:
            dict avec poids et performances
        """
        # Calculer les rendements postérieurs
        posterior_returns = self.calculate_posterior_returns(tau=tau)
        
        # Initialisation
        initial_weights = self.market_weights.copy()
        
        # Contraintes
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Optimisation
        result = minimize(
            self.hybrid_objective,
            initial_weights,
            args=(posterior_returns, penalty_weight),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        
        optimal_weights = result.x
        
        # Calculs de performance
        portfolio_return = np.sum(posterior_returns * optimal_weights)
        portfolio_vol = self.portfolio_volatility(optimal_weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Analyse Risk Parity
        rc = self.risk_contribution(optimal_weights)
        rc_pct = (rc / portfolio_vol * 100) if portfolio_vol > 0 else np.zeros(self.num_assets)
        
        return {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'risk_contributions': dict(zip(self.asset_names, rc)),
            'risk_contributions_pct': dict(zip(self.asset_names, rc_pct)),
            'posterior_returns': dict(zip(self.asset_names, posterior_returns)),
            'implied_returns': dict(zip(self.asset_names, self.implied_returns)),
            'optimizer_success': result.success,
            'optimizer_message': result.message
        }
    
    def compare_all_strategies(self, tau=0.025, penalty_weight=100):
        """Comparaison complète: Equal Weight vs Risk Parity vs B-L vs Hybrid."""
        
        # Equal Weight
        ew_weights = np.array([1 / self.num_assets] * self.num_assets)
        ew_return = np.sum(self.mean_returns * ew_weights)
        ew_vol = self.portfolio_volatility(ew_weights)
        ew_sharpe = (ew_return - self.risk_free_rate) / ew_vol if ew_vol > 0 else 0
        
        # Black-Litterman seul
        posterior_returns = self.calculate_posterior_returns(tau=tau)
        bl_result = self._optimize_markowitz(posterior_returns)
        
        # Risk Parity seul
        rp_result = self._optimize_risk_parity()
        
        # Hybrid
        hybrid_result = self.optimize_hybrid(tau=tau, penalty_weight=penalty_weight)
        
        # Créer le dataframe de comparaison
        comparison = pd.DataFrame({
            'Strategy': ['Equal Weight', 'Risk Parity', 'Black-Litterman', 'Hybrid (RP+BL)'],
            'Return': [ew_return, rp_result['return'], bl_result['return'], hybrid_result['return']],
            'Volatility': [ew_vol, rp_result['volatility'], bl_result['volatility'], hybrid_result['volatility']],
            'Sharpe Ratio': [ew_sharpe, rp_result['sharpe_ratio'], bl_result['sharpe_ratio'], hybrid_result['sharpe_ratio']]
        })
        
        return comparison, {
            'equal_weight': {'weights': dict(zip(self.asset_names, ew_weights))},
            'risk_parity': rp_result,
            'black_litterman': bl_result,
            'hybrid': hybrid_result
        }
    
    def _optimize_markowitz(self, expected_returns):
        """Optimisation Markowitz standard."""
        def negative_sharpe(weights):
            ret = np.sum(expected_returns * weights)
            vol = self.portfolio_volatility(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 1e10
        
        initial_weights = self.market_weights.copy()
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret = np.sum(expected_returns * weights)
        vol = self.portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        return {
            'weights': dict(zip(self.asset_names, weights)),
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def _optimize_risk_parity(self):
        """Optimisation Risk Parity."""
        def rp_objective(weights):
            rc = self.risk_contribution(weights)
            portfolio_vol = self.portfolio_volatility(weights)
            target_rc = portfolio_vol / self.num_assets if portfolio_vol > 0 else 1/self.num_assets
            return np.sum((rc - target_rc) ** 2)
        
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        result = minimize(
            rp_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret = np.sum(self.mean_returns * weights)
        vol = self.portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        rc = self.risk_contribution(weights)
        
        return {
            'weights': dict(zip(self.asset_names, weights)),
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'risk_contributions': dict(zip(self.asset_names, rc))
        }


def main():
    """Exemple complet de l'optimisation hybride."""
    np.random.seed(42)
    
    # Simulation de données
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    returns_data = pd.DataFrame(
        np.random.randn(252, 5) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Créer l'optimiseur hybride
    optimizer = HybridRiskParityBlackLitterman(returns_data, risk_free_rate=0.02)
    
    # Ajouter des views
    views = {
        'optimistic_tech': {
            'type': 'absolute',
            'assets': ['AAPL', 'MSFT'],
            'expected_return': 0.08,
            'confidence': 0.85
        },
        'pessimistic_tsla': {
            'type': 'absolute',
            'assets': ['TSLA'],
            'expected_return': 0.02,
            'confidence': 0.6
        },
        'neutral_googl': {
            'type': 'absolute',
            'assets': ['GOOGL'],
            'expected_return': 0.05,
            'confidence': 0.7
        }
    }
    
    optimizer.add_views(views)
    
    print("=" * 80)
    print("HYBRID OPTIMIZATION: RISK PARITY + BLACK-LITTERMAN")
    print("=" * 80)
    
    # Comparaison de tous les strategies
    comparison, all_results = optimizer.compare_all_strategies(tau=0.05, penalty_weight=500)
    
    print("\n" + "=" * 80)
    print("COMPARAISON DES STRATEGIES")
    print("=" * 80)
    print(comparison.to_string(index=False))
    
    # Détails du portefeuille hybride
    hybrid = all_results['hybrid']
    
    print("\n" + "=" * 80)
    print("DETAIL: PORTEFEUILLE HYBRIDE (Risk Parity + Black-Litterman)")
    print("=" * 80)
    print(f"\nRendement: {hybrid['return']:.4%}")
    print(f"Volatilité: {hybrid['volatility']:.4%}")
    print(f"Ratio Sharpe: {hybrid['sharpe_ratio']:.4f}")
    
    print("\nPoids du portefeuille:")
    for asset in assets:
        weight = hybrid['weights'][asset]
        print(f"  {asset}: {weight:.4%}")
    
    print("\nContribution au risque (Risk Parity check):")
    print("Asset | Weight | Risk_Contribution | RC_% | Target_%")
    print("-" * 60)
    for asset in assets:
        weight = hybrid['weights'][asset]
        rc = hybrid['risk_contributions'][asset]
        rc_pct = hybrid['risk_contributions_pct'][asset]
        target = 100 / len(assets)
        print(f"{asset:5} | {weight:6.2%} | {rc:17.4f} | {rc_pct:5.1f} | {target:6.1f}")
    
    print("\nRendements (comparaison):")
    print("Asset | Historical | Implied | Posterior")
    print("-" * 50)
    for asset in assets:
        hist = optimizer.mean_returns[assets.index(asset)]
        impl = hybrid['implied_returns'][asset]
        post = hybrid['posterior_returns'][asset]
        print(f"{asset:5} | {hist:11.4%} | {impl:7.4%} | {post:9.4%}")


if __name__ == '__main__':
    main()
