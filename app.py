"""
Dash application

styling links
https://github.com/thomaspark/bootswatch/tree/master/dist/
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
"""

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True
