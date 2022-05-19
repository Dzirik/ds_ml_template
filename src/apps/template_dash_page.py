"""
Dash application
"""
from abc import abstractmethod
from typing import List, Any
from typing import NamedTuple

from dash.html import Div

from src.apps.block_header import BlockHeader
from src.apps.block_sidebar import BlockSidebar
from src.utils.config import Config


class PageConfig(NamedTuple):
    """
    Configuration tuple for the page.
    """
    url: str
    name: str


class TemplateDashPage:
    """
    Base class for a separate page.
    """

    def __init__(self) -> None:
        self._page = PageConfig

    def get_url(self) -> str:
        """
        Returns url of the page.
        :return: str. Url of the page.
        """
        return self._page.url

    def create_layout(self) -> Div:
        """
        Creates layout of the page as a dash Div object.
        :return: Div. Layout of the page wrapped in Div dash component.
        """
        return Div([
            BlockHeader().create(),
            Div([
                BlockSidebar().create(),
                Div(
                    self.create_content_list()
                    , style=Config().get().dash.content_style
                )
            ])
        ])

    @abstractmethod
    def create_content_list(self) -> List[Any]:
        """
        Creates content of the page.
        :return: Div. Layout of the page wrapped in Div dash component.
        """
