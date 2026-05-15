import pandas as pd, glob, os, numpy as np

files = sorted(glob.glob('outputs/backtest_weights_*.xlsx'))
fname = files[-1] if files else 'outputs/backtest_weights.xlsx'
print("File:", fname)
df = pd.read_excel(fname)
print('Rows:', len(df))
print(df['weight'].describe())
stuck = df[abs(df['weight'] - 1/15) < 0.001]
print(f'\nEqual weights (1/15): {len(stuck)}/{len(df)} = {len(stuck)/len(df)*100:.1f}%')

non_eq = df[abs(df['weight'] - 1/15) >= 0.001]
if not non_eq.empty:
    sample_dates = non_eq['date'].unique()[:3]
    for d in sample_dates:
        sub = df[df['date'] == d].sort_values('weight', ascending=False)
        print(f'\nDate {d}:')
        print(sub.to_string(index=False))
