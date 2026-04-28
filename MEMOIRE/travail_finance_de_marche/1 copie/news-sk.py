import yfinance as yf
import pandas as pd
import par84

images = par84.img_path + 'news-'

tickers = ['AAPL','TSLA','BTC-USD']
news_per_ticker = 3

rows = [{
    "Title": n.get("title",""),
    "Link": f'<a href="{url}" target="_blank">Open</a>'
}
for t in tickers
for n in yf.Search(t, news_count=news_per_ticker).news[:news_per_ticker]
for url in [n.get("link","")]]

html = pd.DataFrame(rows).to_html(index=False, border=0,  escape=False, header=False)
html = html.replace("<table", '<table style="color:white;"')
with open(images + "1.html","w",encoding="utf-8") as f: f.write(html)


