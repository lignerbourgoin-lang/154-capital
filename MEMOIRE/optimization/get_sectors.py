import yfinance as yf
import pandas as pd

tickers = [
    '0700.HK','AAPL','ABBV','AIR.PA','ALV.DE','AMZN','ASML.AS','AXP','AZN',
    'BABA','BAS.DE','BLK','BNP.PA','BRK-B','C','CAP.PA','DBK.DE','DHR','DIS',
    'EDEN.PA','FAST','GOOGL','GS','HD','HDFCBANK.NS','HSBC','HUBS','ICICIBANK.NS',
    'IEX','INFY.NS','INGA.AS','JD','JNJ','JPM','KER.PA','KNEBV.HE','KO','LULU',
    'MC.PA','MCD','META','MRK','MS','MSFT','NESN.SW','NKE','NOVN.SW','NVDA','NVS',
    'ODFL','OR.PA','PAYC','PEP','PFE','PG','RELIANCE.NS','RMS.PA','RNO.PA','ROG.SW',
    'ROK','SAN.PA','SAP.DE','SBUX','SIE.DE','SU.PA','TCS.NS','TMO','TSM','TT',
    'TTE.PA','UMI.BR','UNH','V','WAT','WLN.PA','ZAL.DE'
]

sectors = {}
for t in tickers:
    try:
        info = yf.Ticker(t).info
        s = info.get('sector', info.get('sectorDisp', 'Unknown'))
        sectors[t] = s if s else 'Unknown'
        print(f"  {t:<20} {sectors[t]}")
    except Exception as e:
        sectors[t] = 'Unknown'
        print(f"  {t:<20} ERROR: {e}")

df = pd.DataFrame.from_dict(sectors, orient='index', columns=['Sector'])
print("\n=== Par secteur ===")
print(df.groupby('Sector').size().sort_values(ascending=False).to_string())
df.to_csv('outputs/sectors.csv')
print("\nSauvegardé -> outputs/sectors.csv")
