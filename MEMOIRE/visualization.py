"""
Visualisations avancées pour les portefeuilles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


class PortfolioVisualizer:
    """Génère des visualisations pour les portefeuilles."""
    
    def __init__(self, figsize=(14, 8), style='seaborn-v0_8-darkgrid'):
        """
        Args:
            figsize: Taille des figures (width, height)
            style: Style matplotlib
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_portfolio_weights(self, results_dict, save_path=None):
        """
        Affiche les poids des portefeuilles côte à côte.
        
        Args:
            results_dict: dict {nom: {'weights': {...}, ...}}
            save_path: Chemin pour sauvegarder l'image
        """
        num_strategies = len(results_dict)
        fig, axes = plt.subplots(1, num_strategies, figsize=(5*num_strategies, 6))
        
        if num_strategies == 1:
            axes = [axes]
        
        for idx, (strategy_name, result) in enumerate(results_dict.items()):
            weights = result['weights']
            
            # Convertir dict en values
            assets = list(weights.keys())
            values = [weights[asset] for asset in assets]
            
            # Filtrer les poids nuls pour une meilleure visualisation
            non_zero_idx = [i for i, v in enumerate(values) if v > 0.001]
            assets_display = [assets[i] for i in non_zero_idx]
            values_display = [values[i] for i in non_zero_idx]
            
            colors = sns.color_palette("husl", len(assets_display))
            
            # Pie chart
            ax = axes[idx]
            wedges, texts, autotexts = ax.pie(
                values_display,
                labels=assets_display,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 10}
            )
            
            # Améliorer les pourcentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax.set_title(strategy_name, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_risk_contributions(self, strategy_name, risk_contributions_pct, save_path=None):
        """
        Affiche la contribution au risque de chaque actif.
        
        Args:
            strategy_name: Nom de la stratégie
            risk_contributions_pct: dict {asset: percentage}
            save_path: Chemin pour sauvegarder
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        assets = list(risk_contributions_pct.keys())
        values = list(risk_contributions_pct.values())
        target = 100 / len(assets)
        
        colors = ['green' if abs(v - target) < 5 else 'orange' if abs(v - target) < 15 else 'red' 
                  for v in values]
        
        bars = ax.bar(assets, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Ligne de target
        ax.axhline(y=target, color='blue', linestyle='--', linewidth=2, label=f'Target ({target:.1f}%)')
        
        ax.set_ylabel('Risk Contribution (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
        ax.set_title(f'Risk Contribution by Asset - {strategy_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_cumulative_returns(self, returns_data, results_dict, save_path=None):
        """
        Affiche les rendements cumulés de différentes stratégies.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            results_dict: dict {nom: {'weights': {...}, ...}}
            save_path: Chemin pour sauvegarder
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for strategy_name, result in results_dict.items():
            weights = result['weights']
            
            # Convertir dict en array
            weights_array = np.array([weights.get(col, 0) for col in returns_data.columns])
            
            # Rendements du portefeuille
            portfolio_returns = (returns_data * weights_array).sum(axis=1)
            
            # Rendements cumulés
            cumulative = (1 + portfolio_returns).cumprod()
            
            ax.plot(cumulative.index, (cumulative - 1) * 100, 
                   linewidth=2.5, label=strategy_name, marker='o', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_drawdown(self, returns_data, results_dict, save_path=None):
        """
        Affiche le drawdown (baisse) de chaque stratégie.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            results_dict: dict {nom: {'weights': {...}, ...}}
            save_path: Chemin pour sauvegarder
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for strategy_name, result in results_dict.items():
            weights = result['weights']
            weights_array = np.array([weights.get(col, 0) for col in returns_data.columns])
            
            portfolio_returns = (returns_data * weights_array).sum(axis=1)
            cumulative = (1 + portfolio_returns).cumprod()
            
            # Calculer le drawdown
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=strategy_name)
            ax.plot(drawdown.index, drawdown, linewidth=2, label=None)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_returns_distribution(self, returns_data, results_dict, save_path=None):
        """
        Affiche la distribution des rendements quotidiens.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            results_dict: dict {nom: {'weights': {...}, ...}}
            save_path: Chemin pour sauvegarder
        """
        num_strategies = len(results_dict)
        fig, axes = plt.subplots(1, num_strategies, figsize=(6*num_strategies, 5))
        
        if num_strategies == 1:
            axes = [axes]
        
        for idx, (strategy_name, result) in enumerate(results_dict.items()):
            weights = result['weights']
            weights_array = np.array([weights.get(col, 0) for col in returns_data.columns])
            
            portfolio_returns = (returns_data * weights_array).sum(axis=1)
            
            ax = axes[idx]
            
            # Histogramme
            ax.hist(portfolio_returns * 100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Ajouter les stats
            mean_ret = portfolio_returns.mean() * 100
            std_ret = portfolio_returns.std() * 100
            
            ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.3f}%')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            ax.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{strategy_name}\nμ={mean_ret:.3f}%, σ={std_ret:.3f}%', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_efficient_frontier(self, returns_data, results_dict, save_path=None):
        """
        Affiche la frontière efficiente avec les portefeuilles.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            results_dict: dict {nom: {'return': ..., 'volatility': ..., ...}}
            save_path: Chemin pour sauvegarder
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extracting data
        strategies = list(results_dict.keys())
        returns = [results_dict[s].get('return', 0) * 100 for s in strategies]
        volatilities = [results_dict[s].get('volatility', 0) * 100 for s in strategies]
        sharpes = [results_dict[s].get('sharpe_ratio', 0) for s in strategies]
        
        # Normaliser les Sharpe ratios pour les couleurs
        sharpe_normalized = [(s - min(sharpes)) / (max(sharpes) - min(sharpes) + 0.001) 
                            for s in sharpes]
        
        # Scatter plot avec couleurs basées sur Sharpe
        scatter = ax.scatter(volatilities, returns, s=300, c=sharpe_normalized, 
                           cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Ajouter les labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (volatilities[i], returns[i]), 
                       fontsize=10, fontweight='bold', ha='center', va='center')
        
        # Ajouter la capital allocation line (CAL)
        risk_free_rate = 0.02 * 100
        if max(volatilities) > 0:
            max_sharpe_idx = np.argmax(sharpes)
            max_vol = volatilities[max_sharpe_idx]
            max_ret = returns[max_sharpe_idx]
            
            if max_vol > 0:
                slope = (max_ret - risk_free_rate) / max_vol
                cal_x = np.array([0, max(volatilities) * 1.2])
                cal_y = risk_free_rate + slope * cal_x
                ax.plot(cal_x, cal_y, 'r--', linewidth=2, label='Capital Allocation Line')
        
        ax.axhline(y=risk_free_rate, color='gray', linestyle=':', linewidth=1.5, 
                  label=f'Risk-free Rate ({risk_free_rate:.2f}%)')
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Efficient Frontier - Portfolio Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio (normalized)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self, returns_data, save_path=None):
        """
        Affiche la matrice de corrélation des actifs.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            save_path: Chemin pour sauvegarder
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation = returns_data.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                   linewidths=1, linecolor='gray')
        
        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        return fig
    
    def create_summary_report(self, returns_data, results_dict, output_dir='.'):
        """
        Crée un rapport complet avec tous les graphiques.
        
        Args:
            returns_data: DataFrame avec les rendements quotidiens
            results_dict: dict {nom: {'weights': {...}, ...}}
            output_dir: Répertoire pour sauvegarder les graphiques
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GÉNÉRATION DU RAPPORT VISUEL")
        print("="*70)
        
        # 1. Poids
        self.plot_portfolio_weights(results_dict, 
                                   f"{output_dir}/01_portfolio_weights.png")
        
        # 2. Rendements cumulés
        self.plot_cumulative_returns(returns_data, results_dict,
                                    f"{output_dir}/02_cumulative_returns.png")
        
        # 3. Drawdown
        self.plot_drawdown(returns_data, results_dict,
                          f"{output_dir}/03_drawdown.png")
        
        # 4. Distribution des rendements
        self.plot_returns_distribution(returns_data, results_dict,
                                      f"{output_dir}/04_returns_distribution.png")
        
        # 5. Frontière efficiente
        self.plot_efficient_frontier(returns_data, results_dict,
                                    f"{output_dir}/05_efficient_frontier.png")
        
        # 6. Corrélation
        self.plot_correlation_heatmap(returns_data,
                                     f"{output_dir}/06_correlation_heatmap.png")
        
        # 7. Risk Contribution (si disponible)
        for strategy_name, result in results_dict.items():
            if 'risk_contributions_pct' in result:
                self.plot_risk_contributions(strategy_name, 
                                            result['risk_contributions_pct'],
                                            f"{output_dir}/07_risk_contribution_{strategy_name.lower().replace(' ', '_')}.png")
        
        print("\n✓ Tous les graphiques ont été générés!")


def main():
    """Exemple d'utilisation."""
    np.random.seed(42)
    
    # Données
    dates = pd.date_range('2023-01-01', periods=252)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    returns_data = pd.DataFrame(
        np.random.randn(252, 5) * 0.02 + 0.0005,
        columns=assets,
        index=dates
    )
    
    # Résultats simulés
    results = {
        'Equal Weight': {
            'weights': {asset: 0.2 for asset in assets},
            'return': 0.001262,
            'volatility': 0.009074,
            'sharpe_ratio': -2.0650
        },
        'Risk Parity': {
            'weights': {asset: 0.2 for asset in assets},
            'return': 0.001262,
            'volatility': 0.009074,
            'sharpe_ratio': -2.0650
        },
        'Hybrid': {
            'weights': {'AAPL': 0.498, 'GOOGL': 0.502, 'MSFT': 0, 'AMZN': 0, 'TSLA': 0},
            'return': 0.038119,
            'volatility': 0.013580,
            'sharpe_ratio': 1.3342,
            'risk_contributions_pct': {asset: 20 for asset in assets}
        }
    }
    
    # Générer les visualisations
    visualizer = PortfolioVisualizer()
    visualizer.create_summary_report(returns_data, results, output_dir='./portfolio_analysis')


if __name__ == '__main__':
    main()
