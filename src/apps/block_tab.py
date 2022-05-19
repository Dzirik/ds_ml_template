"""
Tab component
"""
from abc import ABC
from typing import List, Any

from dash_bootstrap_components import Row
from dash.dcc import Tabs, Tab
from dash.html import Div


class BlockTab(ABC):
    """
    Creates a tab component.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def create(setting: List[Any], tabs: List[Tab]) -> List[Any]:
        """
        Creates a div for header.
        :return: Navbar. Navbar for header.
        """
        return [
            Row(
                setting
            ),
            Tabs(
                children=tabs,
                vertical=False,
            )
        ]

    @staticmethod
    def create_one_tab(tab_name: str, components: List[Any]) -> Tab:
        """
        Creates one tab.
        :param tab_name: str. Name of the tab.
        :param components: List of components to be on the tab.
        :return: Tab.
        """
        return Tab(
            label=tab_name,
            children=[Div(components)]
        )

    @staticmethod
    def create_disabled_tab() -> Tab:
        """
        Creates a disabled tab.
        :return: Tab. Empty tab.
        """
        return Tab(label="", disabled=True)
