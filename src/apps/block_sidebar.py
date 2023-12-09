"""
Sidebar

Link: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/dropdown_menu/
"""

from typing import Tuple, List, Union

from dash import html
from dash.html import Div
from dash_bootstrap_components import Nav, NavItem, NavLink, DropdownMenuItem, \
    DropdownMenu

from src.utils.config import Config


class BlockSidebar:
    """
    Creates a sidebar for page.

    Notes: If you delete background color in setting, the header is visible.
    """

    def __init__(self) -> None:
        self._header_config: List[Union[Tuple[str, str, str], List[Tuple[str, str, str]]]] = []

    @staticmethod
    def _add_image() -> Div:
        return Div([
            html.Img(
                src=Config().get_data().dash.sett.path_to_image,
                style={"height": "5%", "width": "50%"})
        ], style={"textAlign": "center"})

    @staticmethod
    def _create_navigation_section() -> Nav:
        """
        Creates links on the sidebar.
        :return: Nav. Navigation panel on the sidebar.
        """
        nav_bar_items = []
        for item in Config().get_data().dash.sidebar_config:
            if len(item[0]) == 1:
                nav_bar_items.append(NavItem(NavLink(Config().get_data().dash.list_of_pages[item[0][0]][1],
                                                     href=Config().get_data().dash.list_of_pages[item[0][0]][0],
                                                     style={"color": Config().get_data().dash.sett.header_link_color})))
                                                     # style={"color": Config().get_data().dash.sett.header_link_color,
                                                     #        "border": "2px solid black"})))
            else:
                dropdown_menu = []
                for sub_item in item:
                    dropdown_menu.append(DropdownMenuItem(Config().get_data().dash.list_of_pages[sub_item[0]][1],
                                                          href=Config().get_data().dash.list_of_pages[sub_item[0]][0]))
                nav_bar_items.append(
                    DropdownMenu(
                        label=item[0][1],
                        children=dropdown_menu
                    )
                )
        return Nav(
            nav_bar_items,
            navbar=True,
            vertical=True,
            pills=True,
            style={"font-weight": Config().get_data().dash.sett.header_font_weight}
            # style = {"font-weight": Config().get_data().dash.sett.header_font_weight, "backgroundColor": "#B2BEB5",
            #      "border": "2px solid black"}
        )

    def create(self) -> Div:
        """
        Creates a return the sidebar.
        :return: Navbar. Navbar for sidebar.
        """
        return Div([
            html.H2("Sidebar", className="display-4", style={"textAlign": "center"}),
            html.Hr(),
            html.P("Data Science PoC", className="lead", style={"textAlign": "center"}),
            self._create_navigation_section()
        ], style=Config().get_data().dash.sidebar_style)
