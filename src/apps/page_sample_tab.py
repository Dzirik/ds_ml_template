"""
Page
"""
from typing import List, Any

from dash.dcc import Graph
from dash.html import H3, P

from src.apps.block_tab import BlockTab
from src.apps.template_dash_page import TemplateDashPage, PageConfig


class PageSampleTab(TemplateDashPage):  # type:ignore
    """
    Creates an example page containing tabs.
    """

    def __init__(self) -> None:
        TemplateDashPage.__init__(self)
        self._page = PageConfig("/PageSampleTab", "PageSampleTab")
        self._tab_comp = BlockTab()

    def create_content_list(self) -> List[Any]:
        content_list: List[Any]
        content_list = self._tab_comp.create(
            self.create_setting(),
            [
                self._tab_comp.create_one_tab("Graph Tab", self.create_graph_tab_components()),
                self._tab_comp.create_one_tab("Second Tab", self.create_second_tab_components()),
                self._tab_comp.create_one_tab("Third Tab", self.create_third_tab_components()),
                self._tab_comp.create_disabled_tab()
            ])
        return content_list

    @staticmethod
    def create_setting() -> List[Any]:
        """
        Creates a setting for the block of tabs.
        :return: List[Any]. List of components to be there.
        """
        return [
            H3("This is a setting.")
        ]

    @staticmethod
    def create_graph_tab_components() -> List[Any]:
        """
        Creates components for a tab.
        :return: List[Any]. Returns components for tab.
        """
        return [
            Graph(
                id="example-graph",
                figure={
                    "data": [
                        {"x": [1, 2, 3], "y": [4, 1, 2],
                         "type": "bar", "name": "SF"},
                        {"x": [1, 2, 3], "y": [2, 4, 5],
                         "type": "bar", "name": "Montréal"},
                    ],
                    "layout": {
                        "title": "Dash Data Visualization"
                    }
                }
            )
        ]

    @staticmethod
    def create_second_tab_components() -> List[Any]:
        """
        Creates components for a tab.
        :return: List[Any]. Returns components for tab.
        """
        return [
            H3("This is the content in tab 2"),
            P("A graph here would be nice!")
        ]

    @staticmethod
    def create_third_tab_components() -> List[Any]:
        """
        Creates components for a tab.
        :return: List[Any]. Returns components for tab.
        """
        return [
            H3("This is the content in tab 3"),
        ]
