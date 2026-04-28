import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
from config import TICKERS, START, END, DATA_DIR

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

print("Downloading prices …")

raw = yf.download(
    tickers      = list(TICKERS.keys()),
    start        = START,
    end          = END,
    auto_adjust  = True,
    progress     = False,
)["Close"]

# Rename columns from ticker to full company name
raw = raw.rename(columns=TICKERS)
raw.index.name = "Date"

# Keep only trading days that have at least one valid price
raw = raw.dropna(how="all")

print(raw.tail())
print(f"\nShape : {raw.shape}  ({raw.shape[0]} days × {raw.shape[1]} companies)")

# ---------------------------------------------------------------------------
# Export — wide format
# ---------------------------------------------------------------------------

out = DATA_DIR / "prices.xlsx"
raw.to_excel(out)
print(f"\nSaved : {out}")
