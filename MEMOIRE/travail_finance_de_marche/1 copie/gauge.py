import par84
import plotly.graph_objects as go
from plotly.offline import plot

images = par84.img_path+'gauge-'

value = 420
minvalue = 0
maxvalue = 500
reference = 400

fig = go.Figure(layout=par84.layout)

fig.add_indicator(
    mode="gauge+number+delta",
    value=value,
    domain={'x': [0, 1], 'y': [0, 1]},
    delta={
        'reference': reference,
        'increasing': {'color': '#f92672'}
    },
    gauge={
        'axis': {
            'range': [None, maxvalue],
            'tickwidth': 1,
            'tickcolor': '#aaaaaa'
        },
        'bar': {'color': '#7b2cff'},
        'bgcolor': 'rgba(10, 10, 20, 1)',
        'borderwidth': 3,
        'bordercolor': 'rgba(123, 44, 255, 0.8)',
        'steps': [
            {'range': [minvalue, maxvalue * 0.6], 'color': 'rgba(123, 44, 255, 0.10)'},
            {'range': [maxvalue * 0.6, maxvalue * 0.85], 'color': 'rgba(123, 44, 255, 0.25)'},
            {'range': [maxvalue * 0.85, maxvalue], 'color': 'rgba(249, 38, 114, 0.5)'}
        ],
        'threshold': {
            'line': {'color': '#ff2cff', 'width': 6},
            'thickness': 0.8,
            'value': 490
        }
    }
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': 'white', 'family': 'Helvetica'}
)

plot(fig, filename=images+'1.html', config=par84.config, auto_open=False)

