import plotly.graph_objects as go
import plotly.colors as pc

img_path="../../img/1/"

layout=go.Layout(
    template="plotly_dark",
    title_yanchor = "top",
        title=dict(
        yanchor="top"
    ),
    colorway=pc.qualitative.Plotly + pc.qualitative.Dark24,
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=40, pad=0),
    paper_bgcolor='rgb(40,40,40)',
    plot_bgcolor='rgb(40,40,40)'
)

config = {
    'displayModeBar': False,
    'scrollZoom': True
}
