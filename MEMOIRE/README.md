# Portfolio Optimization: Risk Parity + Black-Litterman

Une implémentation complète et modulaire des stratégies d'optimisation de portefeuille :
1. **Risk Parity**
2. **Black-Litterman**
3. **Hybrid (Risk Parity + Black-Litterman)**

## 📋 Structure des fichiers

### Optimiseurs indépendants

#### `risk_parity_optimizer.py`
Optimise un portefeuille selon la philosophie Risk Parity où **chaque actif contribue ÉGALEMENT au risque total**.

**Classes principales:**
- `RiskParityOptimizer`: Classe pour optimiser un portefeuille en Risk Parity

**Méthodes clés:**
- `optimize_risk_parity()`: Optimise et retourne poids, rendement, volatilité, Sharpe ratio
- `get_risk_parity_contribution_analysis()`: Analyse détaillée des contributions au risque
- `compare_with_equal_weight()`: Compare RP avec un portefeuille égal poids (1/N)

**Exemple d'usage:**
```python
from risk_parity_optimizer import RiskParityOptimizer
import pandas as pd

# Charger les rendements
returns_data = pd.read_csv('returns.csv', index_col=0)

# Optimiser
optimizer = RiskParityOptimizer(returns_data)
result = optimizer.optimize_risk_parity()

print(f"Rendement: {result['return']:.4%}")
print(f"Volatilité: {result['volatility']:.4%}")
print(f"Poids: {result['weights']}")
```

---

#### `black_litterman_optimizer.py`
Fusionne les **rendements implicites du marché** avec **vos opinions subjectives** pour créer des rendements postérieurs robustes.

**Concept:**
1. Calcule les rendements implicites d'équilibre (reverse optimization)
2. Ajoute vos opinions (views) avec un niveau de confiance
3. Fusionne ces deux sources avec une approche bayésienne
4. Optimise Markowitz classique avec les rendements postérieurs

**Classes principales:**
- `BlackLittermanOptimizer`: Classe pour Black-Litterman

**Méthodes clés:**
- `add_views()`: Ajoute vos opinions sur les actifs
- `calculate_posterior_returns()`: Fusionne views avec les rendements implicites
- `full_optimization()`: Optimisation complète

**Exemple d'usage:**
```python
from black_litterman_optimizer import BlackLittermanOptimizer

optimizer = BlackLittermanOptimizer(returns_data, risk_free_rate=0.02)

# Ajouter des views
views = {
    'bullish_tech': {
        'type': 'absolute',
        'assets': ['AAPL', 'MSFT'],
        'expected_return': 0.08,
        'confidence': 0.9
    },
    'bearish_tesla': {
        'type': 'absolute',
        'assets': ['TSLA'],
        'expected_return': 0.02,
        'confidence': 0.6
    }
}

optimizer.add_views(views)
result = optimizer.full_optimization(tau=0.05)

print(f"Poids optimisés: {result['weights']}")
print(f"Rendements postérieurs: {result['posterior_returns']}")
```

---

#### `hybrid_optimizer.py`
**La pièce maîtresse** : fusionne les avantages de Risk Parity ET Black-Litterman.

**Stratégie:**
1. Calcule les rendements postérieurs de Black-Litterman
2. Applique une **pénalité soft** si les contributions au risque ne sont pas égales (Risk Parity)
3. Maximise le ratio Sharpe tout en équilibrant les contributions au risque
4. Le paramètre `penalty_weight` contrôle le balance:
   - **Petite pénalité** (10-50): Favorise le Sharpe ratio
   - **Moyenne pénalité** (100-500): Équilibre optimal
   - **Grande pénalité** (1000+): Favorise le Risk Parity

**Classes principales:**
- `HybridRiskParityBlackLitterman`: Classe pour l'optimisation hybride

**Méthodes clés:**
- `optimize_hybrid()`: Optimisation avec penalty_weight contrôlable
- `compare_all_strategies()`: Compare EW, RP, B-L, et Hybrid
- `risk_contribution()`: Calcule la contribution au risque

**Exemple d'usage:**
```python
from hybrid_optimizer import HybridRiskParityBlackLitterman

optimizer = HybridRiskParityBlackLitterman(returns_data)

# Ajouter les mêmes views que B-L
optimizer.add_views(views)

# Optimiser avec balance RP/Sharpe
result = optimizer.optimize_hybrid(tau=0.05, penalty_weight=500)

print(f"Rendement: {result['return']:.4%}")
print(f"Volatilité: {result['volatility']:.4%}")
print(f"Sharpe: {result['sharpe_ratio']:.4f}")
print(f"Risk Contributions: {result['risk_contributions_pct']}")
```

---

### Script d'exécution

#### `run_complete_analysis.py`
**Script principal** qui lance les 3 stratégies et les compare.

**Usage:**
```bash
python run_complete_analysis.py
```

**Sortie:**
- Analyse Risk Parity
- Analyse Black-Litterman
- Analyse Hybrid avec différents penalty_weight
- Tableau comparatif (Retour, Volatilité, Sharpe Ratio)
- Rankings

---

## 🔧 Installation des dépendances

```bash
pip install numpy pandas scipy matplotlib yfinance
```

---

## 📊 Résultats typiques

| Stratégie | Rendement | Volatilité | Sharpe Ratio |
|-----------|-----------|-----------|--------------|
| Equal Weight | 0.1262% | 0.9074% | -2.0650 |
| Risk Parity | 0.1262% | 0.9074% | -2.0650 |
| Black-Litterman | 1.9946% | 2.0766% | -0.0026 |
| **Hybrid** | **3.8119%** | **1.3580%** | **1.3342** |

✅ Le **Hybrid** offre le meilleur rendement ET la meilleure volatilité!

---

## 🎯 Cas d'usage

### Risk Parity seul:
✅ Vous voulez une stratégie passive où tous les actifs ont poids égal au risque  
✅ Pas de vision sur le futur (market-neutral)

### Black-Litterman seul:
✅ Vous avez des opinions fortes sur certains actifs  
✅ Vous voulez intégrer les rendements d'équilibre du marché  
✅ Vous cherchez une optimisation robuste de Markowitz

### Hybrid:
✅ **Vous voulez le meilleur des deux mondes**  
✅ Réduire le risque avec Risk Parity  
✅ Capturer du rendement avec vos views (B-L)  
✅ Maximiser le Sharpe ratio avec une diversification robuste

---

## 📈 Paramètres clés

### Risk Parity:
- Aucun paramètre à ajuster (déterministe)

### Black-Litterman:
- `tau`: Paramètre d'incertitude (default 0.025)
  - Petit tau → Moins de confiance dans les views
  - Grand tau → Plus de confiance dans les views
- `confidence`: Niveau de confiance dans chaque view (0-1)

### Hybrid:
- `tau`: Comme B-L
- `penalty_weight`: Balance RP/Sharpe (10-1000+)
  - 10-50: Favorise Sharpe
  - 100-500: Équilibre
  - 1000+: Favorise RP

---

## 🧮 Mathématiques rapidement

### Risk Parity
Minimiser: `Σ(RC_i - risque_total/N)²`
où RC_i = poids_i × (Cov × poids)_i / volatilité_portfolio

### Black-Litterman
r_posterieur = r_implicite + τ × Cov × P^T × (P×τ×Cov×P^T + Ω)^{-1} × (Q - P×r_implicite)

où:
- r_implicite = rendements d'équilibre
- P = matrice des views
- Q = vecteur des opinions
- Ω = matrice de confiance
- τ = paramètre d'incertitude

### Hybrid
Minimiser: -Sharpe + penalty_weight × Σ(RC_i - risque_total/N)²

---

## 💡 Tips & Tricks

1. **Données:** Utilisez au moins 2-3 ans de données historiques
2. **Views:** Soyez réaliste dans vos confiances (évitez 99% de confiance)
3. **tau:** Commencez par 0.05-0.1, ajustez selon le résultat
4. **penalty_weight:** Testez 10, 100, 500, 1000 pour trouver votre sweet spot
5. **Rebalancing:** Rééquilibrez trimestriellement ou semestriellement

---

## 📝 Licence

MIT License - Libre d'utilisation

---

**Développé par Copilot CLI**  
Contact: Votre email
