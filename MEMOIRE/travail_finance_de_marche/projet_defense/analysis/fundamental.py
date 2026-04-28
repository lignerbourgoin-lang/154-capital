import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd

from config import TICKERS, CHARTS_FUNDAMENTAL

# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  body  { background:#282828; color:#e0e0e0;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          padding: 16px; margin: 0; }
  h2    { color:#aaaaff; margin-bottom: 10px; font-size: 1rem; }
  table { width:100%; border-collapse:collapse; font-size:0.82rem; }
  th    { text-align:left; padding:6px 10px; border-bottom:1px solid #444;
          color:#888; font-weight:normal; }
  td    { padding:6px 10px; border-bottom:1px solid #333; vertical-align:top; }
  a     { color:#7b9cff; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .date { color:#888; white-space:nowrap; }
  .pub  { color:#aaa; }
</style>
"""


def _news_table(rows: list[dict]) -> str:
    """Build an HTML <table> from a list of news dicts."""
    if not rows:
        return "<p>No news available.</p>"

    lines = ['<table><tr><th>Date</th><th>Headline</th><th>Source</th></tr>']
    for r in rows:
        lines.append(
            f'<tr>'
            f'<td class="date">{r["date"]}</td>'
            f'<td><a href="{r["link"]}" target="_blank">{r["title"]}</a></td>'
            f'<td class="pub">{r["publisher"]}</td>'
            f'</tr>'
        )
    lines.append("</table>")
    return "\n".join(lines)


def _parse_news(items: list, limit: int) -> list[dict]:
    out = []
    for n in items[:limit]:
        pub_ts = n.get("providerPublishTime", 0)
        date   = pd.to_datetime(pub_ts, unit="s").strftime("%Y-%m-%d") if pub_ts else ""
        out.append(dict(
            date      = date,
            title     = n.get("title",     ""),
            link      = n.get("link",      "#"),
            publisher = n.get("publisher", ""),
        ))
    return out


def _write_html(title: str, body: str, path: Path) -> None:
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body><h2>{title}</h2>{body}</body></html>"
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# News per company
# ---------------------------------------------------------------------------

def build_company_news(ticker: str, name: str, limit: int = 15) -> None:
    items = yf.Search(ticker, news_count=limit).news[:limit]
    rows  = _parse_news(items, limit)
    body  = _news_table(rows)
    slug  = name.lower().replace(" ", "_")
    _write_html(f"{name} — Latest News", body, CHARTS_FUNDAMENTAL / f"news_{slug}.html")


# ---------------------------------------------------------------------------
# Global macro / economic news
# (sampled from major macro proxies: SPY, TLT, GLD, DXY, VIX)
# ---------------------------------------------------------------------------

MACRO_TICKERS = {
    "^GSPC":  "S&P 500",
    "^VIX":   "VIX",
    "TLT":    "US Long Bonds (TLT)",
    "GLD":    "Gold (GLD)",
    "DX-Y.NYB": "US Dollar Index",
}


def build_global_news(limit_per_source: int = 5) -> None:
    rows = []
    for ticker, label in MACRO_TICKERS.items():
        try:
            items = yf.Search(ticker, news_count=limit_per_source).news[:limit_per_source]
            for r in _parse_news(items, limit_per_source):
                r["publisher"] = f"{label} / {r['publisher']}"
                rows.append(r)
        except Exception:
            pass

    rows.sort(key=lambda r: r["date"], reverse=True)
    body = _news_table(rows)
    _write_html("Global Economic News", body, CHARTS_FUNDAMENTAL / "news_global.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    for ticker, name in TICKERS.items():
        print(f"  Fundamental — {name}")
        build_company_news(ticker, name)

    print("  Fundamental — Global macro news")
    build_global_news()


if __name__ == "__main__":
    run()
    print("Done — charts written to", CHARTS_FUNDAMENTAL)
