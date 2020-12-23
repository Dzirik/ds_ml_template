"""
Page
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_html_components import Div, H6, Br

from app import app
from src.apps.template_dash_page import TemplateDashPage, PageConfig
from src.utils.config import Config


class PageButton(TemplateDashPage):  # type:ignore
    """
    Creates an example page with a button.
    """

    def __init__(self) -> None:
        TemplateDashPage.__init__(self)
        self.page = PageConfig("/PageButton", "PageButton")

    def create_content_list(self) -> Div:
        return [
            H6("Change the value in the text box to see callbacks in action!"),
            Div(["Input: ",
                 dcc.Input(id='my-input', value='initial value', type='text')]),
            Br(),
            Div(id='my-output'),
            dcc.Link("Go to Main Page", href="/"),
            html.Img(src=Config().get().dash.sett.path_to_image, style={"height": "10%", "width": "10%"}),
        ]


@app.callback(  # type:ignore
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value: str) -> str:
    """
    Update output div.
    :param input_value: str. Value to be returned.
    :return: str. Value.
    """
    return 'Output: {}'.format(input_value)
