import pandas as pd
import numpy as np

df = pd.read_excel('outputs/backtest_weights.xlsx')
print('Shape:', df.shape)
print('Colonnes:', df.columns.tolist())
print()

# Distribution des poids
print('=== Distribution des poids ===')
print(df['weight'].describe())
print()

# Poids très faibles
tiny = df[df['weight'] < 0.005]
print(f'Poids < 0.5% : {len(tiny)} lignes ({len(tiny)/len(df)*100:.1f}%)')
print()

# Exemple d'une date avec des poids bizarres
sample_date = tiny['date'].iloc[0] if not tiny.empty else df['date'].iloc[0]
sub = df[df['date'] == sample_date].sort_values('weight', ascending=False)
print(f'Exemple date {sample_date}:')
print(sub.to_string(index=False))
print(f'Somme des poids: {sub["weight"].sum():.4f}')
print(f'Nombre actifs: {len(sub)}')
