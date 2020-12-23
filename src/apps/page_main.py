"""
Page
"""
from typing import List, Any

import dash_core_components as dcc
from dash_html_components import H6

from src.apps.template_dash_page import TemplateDashPage, PageConfig


class PageMain(TemplateDashPage):  # type:ignore
    """
    Creates an example page with a link.
    """

    def __init__(self) -> None:
        TemplateDashPage.__init__(self)
        self.page = PageConfig("/", "PageMain")

    def create_content_list(self) -> List[Any]:
        return [
            H6("Main page" * 10),
            H6("Main page"),
            H6("Main page"),
            H6("Main page"),
            H6("Main page"),
            H6("Main page"),
            H6("Main page"),
            H6("Main page"),
            dcc.Link("Go to Button Page", href="/PageButton")
        ]
