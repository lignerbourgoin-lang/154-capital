import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Path = r"C:\Eliott\154-capital\MEMOIRE\DATA\univers.xlsx"
Path_onedrive = r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx"

def univers(Path):
    df = pd.read_excel(Path,index_col=0, parse_dates=True)
    df= df.dropna(axis=1, thresh=int(len(df)*0.7))
    asset = df.columns.to_list()
    dates = df.index
    returns_data = df.pct_change().dropna()
        
    
    return asset , dates, returns_data

class MarkowitzOptimizer: 
    
    def __init__(self,returns_data):
        self.returns = returns_data
        self.mean_returns = returns_data.mean().values
        cov = returns_data.cov().values
        self.cov_matrix = (cov + cov.T) / 2
        self.num_assets = len(returns_data.columns)
        self.initial_weights = np.ones(self.num_assets) / self.num_assets
        
        
    def portfolio_fonda(self, weights):
        
        portfolio_returns = np.sum(self.mean_returns* weights)
        portfolio_volatility = np.sqrt(weights.T* self.cov_matrix*weights)
        
        return portfolio_returns, portfolio_volatility
    
    
    def optimization(self):
         
        w = cp.Variable(self.num_assets)
        objectif = cp.Minimize(cp.quad_form(w, self.cov_matrix, assume_PSD=True))
        contraintes = [cp.sum(w)==1 , w >= 0 ]
        probleme = cp.Problem(objectif,contraintes)
        probleme.solve(solver=cp.CLARABEL)
                
        return {
            'weights': w.value,
            'variance': probleme.value,
            'rendement': float(self.mean_returns @ w.value)
        }

    def optimize_sharpe(self, rf=0.0):
        y = cp.Variable(self.num_assets, nonneg=True)
        kappa = cp.Variable(nonneg=True)
        objectif = cp.Maximize((self.mean_returns - rf) @ y)
        contraintes = [
            cp.sum(y) == kappa,
            cp.quad_form(y, self.cov_matrix, assume_PSD=True) <= 1,
        ]
        probleme = cp.Problem(objectif, contraintes)
        probleme.solve(solver=cp.CLARABEL)
        w = y.value / kappa.value
        rendement = float(self.mean_returns @ w)
        volatilite = float(np.sqrt(w @ self.cov_matrix @ w))
        return {
            'weights': w,
            'rendement': rendement,
            'volatilite': volatilite,
            'sharpe': (rendement - rf) / volatilite
        }

    def frontiere_effcient(self):
        volatilities = []
        rendements = []
        all_weights = []
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), 50)
        w = cp.Variable(self.num_assets)
        for targets_returns in target_returns:
            objectif = cp.Minimize(cp.quad_form(w, self.cov_matrix, assume_PSD=True))
            contraintes = [cp.sum(w)==1, w >= 0, self.mean_returns @ w == targets_returns]
            problem = cp.Problem(objectif, contraintes)
            problem.solve(solver=cp.CLARABEL)
            volatilities.append(np.sqrt(problem.value))
            rendements.append(targets_returns)
            all_weights.append(w.value)
        return {
            'target_returns' : rendements,
            'weights' : all_weights,
            'volatilities': volatilities
        }
                      
if __name__ == "__main__":
    print("Chargement des données...")
    assets, dates, returns = univers(Path)
    print(f"Univers : {len(assets)} actifs | {len(returns)} observations | {dates[0].date()} → {dates[-1].date()}")

    print("\nFiltrage sur janvier 2026...")
    returns_jan2026 = returns["2026-01-01":"2026-01-31"]
    print(f"  {len(returns_jan2026)} observations")

    print("\nInitialisation de l'optimiseur...")
    optimizer = MarkowitzOptimizer(returns_jan2026)

    print("Optimisation variance minimale...")
    result = optimizer.optimization()
    print(f"  Rendement : {result['rendement']*100:.4f}%")
    print(f"  Volatilité : {np.sqrt(result['variance'])*100:.4f}%")

    print("\nOptimisation Sharpe maximum...")
    sharpe_result = optimizer.optimize_sharpe()
    print(f"  Rendement : {sharpe_result['rendement']*100:.4f}%")
    print(f"  Volatilité : {sharpe_result['volatilite']*100:.4f}%")
    print(f"  Sharpe (journalier) : {sharpe_result['sharpe']:.4f}")
    print(f"  Sharpe (annualisé)  : {sharpe_result['sharpe'] * np.sqrt(252):.4f}")

    print("\nCalcul de la frontière efficiente (50 points)...")
    frontiere = optimizer.frontiere_effcient()
    print("  Frontière calculée.")
    print(f"  Volatilité min : {min(frontiere['volatilities'])*100:.4f}%")
    print(f"  Volatilité max : {max(frontiere['volatilities'])*100:.4f}%")
    print("\nTerminé.")
     
        
        
        
        
        
        
    
        
        
    
    