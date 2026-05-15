import yfinance as yf

missing = ['ADYEN.AS', 'ALC.SW', 'ANSS', 'DHER.DE', 'PDD']
for t in missing:
    try:
        info = yf.Ticker(t).info
        s = info.get('sector', info.get('sectorDisp', 'Unknown'))
        print(f"  {t:<15} {s}")
    except Exception as e:
        print(f"  {t:<15} ERROR: {e}")
