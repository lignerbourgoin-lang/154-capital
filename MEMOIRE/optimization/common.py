"""
Utilitaires partagés — chargement univers, métriques, logging.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

UNIVERSE_FILE = Path(r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx")
BENCHMARK     = "SPY"
RF            = 0.02
MAX_NAN_PCT   = 0.20

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, colorize=True,
               format="<green>{time:HH:mm:ss}</green> │ <level>{level: <8}</level> │ {message}",
               level="INFO")
    logger.add(OUT / "run.log",
               format="{time:YYYY-MM-DD HH:mm:ss} │ {level: <8} │ {message}",
               level="DEBUG", rotation="10 MB", encoding="utf-8")


def load_universe() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Chargement : {UNIVERSE_FILE.name}")
    df = pd.read_excel(UNIVERSE_FILE, parse_dates=["Date"]).set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    benchmark = df[BENCHMARK].ffill() if BENCHMARK in df.columns else None
    raw       = df.drop(columns=[BENCHMARK], errors="ignore")

    excluded = raw.columns[raw.isna().mean() > MAX_NAN_PCT].tolist()
    if excluded:
        logger.warning(f"Exclus (>{MAX_NAN_PCT*100:.0f}% NaN) : {excluded}")

    prices = raw.drop(columns=excluded).ffill().dropna(how="all")
    prices = prices.dropna(thresh=int(len(prices.columns) * 0.80))

    logger.success(f"{prices.shape[1]} actifs  |  {prices.shape[0]} jours  |  "
                   f"{prices.index[0].date()} → {prices.index[-1].date()}")
    return prices, benchmark


def log_results(results: dict) -> None:
    for name, r in results.items():
        sharpe = r.get("sharpe", 0)
        logger.info(f"  {name:<22}  ret={r['return']*100:>+6.2f}%  "
                    f"vol={r['volatility']*100:>5.2f}%  sharpe={sharpe:>6.3f}")
