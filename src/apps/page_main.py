"""
Page
"""
from typing import List, Any

from dash import dcc
from dash.html import H6

from src.apps.template_dash_page import TemplateDashPage, PageConfig


class PageMain(TemplateDashPage):  # type:ignore
    """
    Creates an example page with a link.
    """

    def __init__(self) -> None:
        TemplateDashPage.__init__(self)
        self._page = PageConfig("/", "PageMain")

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
