import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class BlackLittermanOptimizer:
    """
    Optimiseur Black-Litterman: fusion entre rendements d'équilibre du marché
    et opinions subjectives sur les actifs.
    """
    
    def __init__(self, returns_data, market_weights=None, risk_free_rate=0.02):
        """
        Initialise l'optimiseur Black-Litterman.
        
        Args:
            returns_data: DataFrame pandas avec les rendements historiques
            market_weights: dict ou array avec les poids du marché (si None, poids égaux)
            risk_free_rate: Taux sans risque pour le calcul du rendement implicite
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
        self.risk_free_rate = risk_free_rate
        
        # Poids du marché (par défaut, poids égaux)
        if market_weights is None:
            self.market_weights = np.array([1 / self.num_assets] * self.num_assets)
        else:
            if isinstance(market_weights, dict):
                self.market_weights = np.array([market_weights.get(asset, 1/self.num_assets) 
                                                 for asset in self.asset_names])
            else:
                self.market_weights = np.array(market_weights)
        
        # Normaliser les poids
        self.market_weights = self.market_weights / self.market_weights.sum()
        
        # Rendement implicite du marché (reverse optimization)
        self.market_return = self.calculate_market_return()
        
        # Rendements implicites d'équilibre
        self.implied_returns = self.calculate_implied_returns()
    
    def calculate_market_return(self):
        """Calcule le rendement implicite du portefeuille de marché."""
        return np.dot(self.market_weights, self.mean_returns)
    
    def calculate_implied_returns(self):
        """
        Calcule les rendements implicites d'équilibre.
        
        Formule:
        r_impliques = r_free + lambda * Cov * w_marche
        où lambda = (r_marche - r_free) / variance_marche
        """
        market_variance = np.dot(self.market_weights.T, 
                                 np.dot(self.cov_matrix, self.market_weights))
        
        lambda_param = (self.market_return - self.risk_free_rate) / market_variance
        
        implied = self.risk_free_rate + lambda_param * np.dot(self.cov_matrix, self.market_weights)
        
        return implied
    
    def add_views(self, views_dict, confidence_matrix=None, view_uncertainty=None):
        """
        Ajoute les opinions de l'investisseur.
        
        Args:
            views_dict: dict {
                'view_name': {
                    'type': 'absolute' ou 'relative',
                    'assets': ['AAPL'] ou ['AAPL', 'GOOGL'],
                    'expected_return': 0.05 ou 0.02,
                    'confidence': 0.8  # entre 0 et 1
                }
            }
            confidence_matrix: matrice de confiance personnalisée (sinon calculée auto)
            view_uncertainty: écart-type de l'erreur de la view (sinon calculé auto)
        """
        self.views_list = []
        self.view_type = []  # 'absolute' ou 'relative'
        self.view_confidence = []
        
        for view_name, view_details in views_dict.items():
            view_type = view_details.get('type', 'absolute')
            assets = view_details.get('assets', [])
            expected_return = view_details.get('expected_return', 0)
            confidence = view_details.get('confidence', 0.5)
            
            # Créer la matrice de view (P matrix)
            p_vector = np.zeros(self.num_assets)
            for asset in assets:
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    if view_type == 'absolute':
                        p_vector[idx] = 1
                    elif view_type == 'relative':
                        # Pour les views relatives (ex: AAPL surperforme GOOGL de 2%)
                        p_vector[idx] = 1 / len(assets)
            
            self.views_list.append({
                'name': view_name,
                'p_vector': p_vector,
                'expected_return': expected_return,
                'type': view_type,
                'confidence': confidence
            })
            self.view_type.append(view_type)
            self.view_confidence.append(confidence)
        
        self.num_views = len(self.views_list)
    
    def calculate_posterior_returns(self, tau=0.025):
        """
        Calcule les rendements postérieurs (après fusion des views).
        
        tau: paramètre d'incertitude (scaling factor)
        - Petite tau (ex: 0.05): moins de confiance dans les views
        - Grande tau (ex: 0.5): plus de confiance dans les views
        """
        if not hasattr(self, 'views_list'):
            print("Erreur: Aucune view ajoutée. Utiliser add_views() d'abord.")
            return self.implied_returns
        
        # Construire les matrices P et Q
        P = np.array([view['p_vector'] for view in self.views_list])  # (num_views, num_assets)
        Q = np.array([view['expected_return'] for view in self.views_list])  # (num_views,)
        
        # Matrice d'incertitude des views (Omega)
        Omega = np.diag([
            (view['confidence'] ** -1) * tau * np.dot(view['p_vector'], 
                                                       np.dot(self.cov_matrix, view['p_vector']))
            for view in self.views_list
        ])
        
        # Calcul inverse de covariance
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            print("Attention: Matrice de covariance singulière, utilisation de pseudo-inverse")
            inv_cov = np.linalg.pinv(self.cov_matrix)
        
        # Formule Black-Litterman pour les rendements postérieurs
        # r_posterieur = r_implique + tau * Cov * P.T * inv(P*tau*Cov*P.T + Omega) * (Q - P*r_implique)
        term1 = tau * np.dot(P, np.dot(self.cov_matrix, P.T))
        term2 = term1 + Omega
        
        try:
            inv_term2 = np.linalg.inv(term2)
        except np.linalg.LinAlgError:
            print("Attention: Matrice singulière, utilisation de pseudo-inverse")
            inv_term2 = np.linalg.pinv(term2)
        
        adjustment = np.dot(self.cov_matrix, np.dot(P.T, np.dot(inv_term2, Q - np.dot(P, self.implied_returns))))
        
        posterior_returns = self.implied_returns + tau * adjustment
        
        return posterior_returns
    
    def optimize_with_posterior_returns(self, posterior_returns, optimization_type='sharpe'):
        """
        Optimise le portefeuille avec les rendements postérieurs (Markowitz classique).
        
        Args:
            posterior_returns: array avec les rendements postérieurs
            optimization_type: 'sharpe' (max) ou 'variance' (min)
        
        Returns:
            dict avec poids et performances
        """
        def portfolio_return(weights):
            return np.sum(posterior_returns * weights)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        def minimize_var(weights):
            return portfolio_volatility(weights)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = self.market_weights.copy()
        
        if optimization_type == 'sharpe':
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
        else:
            result = minimize(
                minimize_var,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
        
        weights = result.x
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        return {
            'weights': dict(zip(self.asset_names, weights)),
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
    
    def full_optimization(self, tau=0.025, optimization_type='sharpe'):
        """
        Optimise complètement: views + Black-Litterman + Markowitz.
        
        Returns:
            dict avec les résultats de l'optimisation
        """
        posterior_returns = self.calculate_posterior_returns(tau=tau)
        
        result = self.optimize_with_posterior_returns(posterior_returns, optimization_type)
        
        result['posterior_returns'] = dict(zip(self.asset_names, posterior_returns))
        result['implied_returns'] = dict(zip(self.asset_names, self.implied_returns))
        result['historical_returns'] = dict(zip(self.asset_names, self.mean_returns))
        result['market_weights'] = dict(zip(self.asset_names, self.market_weights))
        
        return result


def main():
    """Exemple d'utilisation de Black-Litterman."""
    np.random.seed(42)
    
    # Simulation de données
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    returns_data = pd.DataFrame(
        np.random.randn(252, 4) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Créer l'optimiseur
    optimizer = BlackLittermanOptimizer(returns_data, risk_free_rate=0.02)
    
    print("=" * 70)
    print("BLACK-LITTERMAN FRAMEWORK")
    print("=" * 70)
    
    print("\n1. RENDEMENTS HISTORIQUES:")
    for asset, ret in zip(assets, optimizer.mean_returns):
        print(f"   {asset}: {ret:.4%}")
    
    print("\n2. RENDEMENTS IMPLICITES D'ÉQUILIBRE (Reverse Optimization):")
    for asset, ret in zip(assets, optimizer.implied_returns):
        print(f"   {asset}: {ret:.4%}")
    
    # Ajouter des views
    views = {
        'view_1': {
            'type': 'absolute',
            'assets': ['AAPL'],
            'expected_return': 0.08,
            'confidence': 0.9
        },
        'view_2': {
            'type': 'absolute',
            'assets': ['GOOGL'],
            'expected_return': 0.05,
            'confidence': 0.6
        },
        'view_3': {
            'type': 'relative',
            'assets': ['MSFT', 'AMZN'],
            'expected_return': 0.02,
            'confidence': 0.7
        }
    }
    
    optimizer.add_views(views)
    
    print("\n3. VIEWS AJOUTÉES:")
    for view in optimizer.views_list:
        print(f"   {view['name']}: {view['type']} - Confiance: {view['confidence']:.1%}")
    
    # Calculer les rendements postérieurs
    posterior_returns = optimizer.calculate_posterior_returns(tau=0.05)
    
    print("\n4. RENDEMENTS POSTÉRIEURS (après fusion des views):")
    for asset, ret in zip(assets, posterior_returns):
        print(f"   {asset}: {ret:.4%}")
    
    # Optimiser
    result = optimizer.full_optimization(tau=0.05, optimization_type='sharpe')
    
    print("\n5. PORTEFEUILLE OPTIMISÉ (Black-Litterman + Markowitz):")
    print(f"   Rendement: {result['return']:.4%}")
    print(f"   Volatilité: {result['volatility']:.4%}")
    print(f"   Ratio Sharpe: {result['sharpe_ratio']:.4f}")
    
    print("\n   Poids:")
    for asset, weight in result['weights'].items():
        print(f"      {asset}: {weight:.4%}")
    
    # Comparaison
    print("\n6. COMPARAISON DES POIDS:")
    print("Asset | Market Weights | Posterior Returns | Optimized Weights")
    print("-" * 65)
    for asset in assets:
        market_w = result['market_weights'][asset]
        post_ret = result['posterior_returns'][asset]
        opt_w = result['weights'][asset]
        print(f"{asset:5} | {market_w:14.2%} | {post_ret:17.4%} | {opt_w:17.2%}")


if __name__ == '__main__':
    main()
