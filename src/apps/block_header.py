"""
Header
"""

from dash.html import Div, H3

HEADER_STYLE = {
    "margin-left": "0rem",  # 16 for it to be right
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa"
}


class BlockHeader:
    """
    Creates a header for every page.

    Notes: If you delete background color for sidebar in settings, the header is visible.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def create() -> Div:
        """
        Creates a div for header.
        :return: Navbar. Navbar for header.
        """
        return Div([
            Div([
                H3("A")
            ], style={"margin-left": "2rem"})
        ], style=HEADER_STYLE)
