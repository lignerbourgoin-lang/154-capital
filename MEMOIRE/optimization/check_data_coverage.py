import pandas as pd
import numpy as np

UNIVERSE_FILE = r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx"
MAX_NAN_PCT   = 0.20

df = pd.read_excel(UNIVERSE_FILE, parse_dates=["Date"]).set_index("Date")
df.index = pd.to_datetime(df.index)
df = df.sort_index()

raw    = df.drop(columns=["URTH", "SPY"], errors="ignore")
raw    = raw.loc[:, raw.isna().mean() <= MAX_NAN_PCT]
prices = raw.ffill().dropna(how="all")
prices = prices.dropna(thresh=int(len(prices.columns) * 0.80))

start = prices.index[0]
end   = prices.index[-1]
n_days = len(prices)

print(f"Données : {start.date()} -> {end.date()}  ({n_days} jours)")
print(f"Actifs  : {prices.shape[1]}")
print()

REBAL_DAYS = 21
for train_years in [3, 4, 5, 6, 7, 8]:
    n_train      = train_years * 252
    n_test_days  = n_days - n_train
    n_periods    = n_test_days // REBAL_DAYS
    first_test   = prices.index[n_train] if n_train < n_days else None
    years_bt     = n_periods * REBAL_DAYS / 252
    print(f"TRAIN_YEARS={train_years}  ->  {n_periods:>4} périodes  "
          f"({years_bt:.1f} ans de backtest)  "
          f"[début test : {first_test.date() if first_test else 'N/A'}]")
