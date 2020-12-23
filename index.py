"""
Execution of dash application.
"""

import os

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from src.apps.page_button import PageButton
from src.apps.page_main import PageMain
from src.apps.page_sample_tab import PageSampleTab
from src.global_constants import ENV_DASH_DEBUG_MODE, ENV_DASH_DOCKER

PAGES = [
    PageButton(),
    PageMain(),
    PageSampleTab()
]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    for page in PAGES:
        if page.get_url() == pathname:
            return page.create_layout()


if __name__ == "__main__":
    print(f"\nThe docker is set to: {os.environ.get(ENV_DASH_DOCKER)}")
    print(f"The dash debug mode is set to: {os.environ.get(ENV_DASH_DEBUG_MODE)}")

    if os.environ.get(ENV_DASH_DOCKER) == "True":
        print("Running dash in docker.\n")
        app.run_server(debug=True, host="0.0.0.0", port="8050")
    else:
        print("Running dash locally.\n")
        app.run_server(debug=True, host="127.0.0.1", port="8050")
