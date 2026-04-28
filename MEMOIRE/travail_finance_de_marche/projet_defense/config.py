from pathlib import Path
import plotly.graph_objects as go
import plotly.colors as pc

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

TICKERS = {
    "LMT":       "Lockheed Martin",
    "AIR.PA":    "Airbus",
    "AM.PA":     "Dassault Aviation",
    "RHM.DE":    "Rheinmetall",
    "SAAB-B.ST": "Saab",
}

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

START = "2011-01-01"
END   = "2026-04-28"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
CHARTS_DIR = ROOT / "charts"

CHARTS_TECHNICAL    = CHARTS_DIR / "technical"
CHARTS_QUANTITATIVE = CHARTS_DIR / "quantitative"
CHARTS_FUNDAMENTAL  = CHARTS_DIR / "fundamental"

# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

LAYOUT = go.Layout(
    template       = "plotly_dark",
    colorway       = pc.qualitative.Plotly + pc.qualitative.Dark24,
    autosize       = True,
    margin         = dict(l=0, r=0, b=0, t=40, pad=0),
    paper_bgcolor  = "rgb(40,40,40)",
    plot_bgcolor   = "rgb(40,40,40)",
    legend         = dict(
        bgcolor     = "rgba(40,40,40,0.8)",
        bordercolor = "rgba(255,255,255,0.2)",
        borderwidth = 1,
    ),
    title = dict(yanchor="top"),
)

CONFIG = {
    "displayModeBar": False,
    "scrollZoom":     True,
}
