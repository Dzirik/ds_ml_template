"""
Functions for visualisations.
"""
from datetime import datetime
from typing import Optional

from numpy import sin
from numpy.random import seed, randn
from pandas import Series


def hex_to_rgb(color: str, opacity: float = 1) -> str:
    """
    Returns a color in format rgba (red, green, blue, alpha/opacity) for the color given in hex format (#rrggbb).
    :param color: str. Color in format #rrggbb.
    :param opacity: int. Opacity value in range [0, 1].
    :return: str. String in the format rgba.
    """
    color = color.lstrip("#")
    len_color = len(color)
    rgb = tuple(int(color[i:i + len_color // 3], 16) for i in range(0, len_color, len_color // 3))
    rgb_opacity = rgb + (opacity,)

    return "rgba" + str(rgb_opacity)


def create_time_series(seed_number: int = 3872, x_multiplier: float = 1, name: Optional[str] = None) -> Series:
    """
    Generates a pandas time series, uses sin(x_multiplier * range(..)) plus some random part.
    :param seed_number: int. Seed for random part.
    :param x_multiplier: float. See the description above.
    :return: pd.Series.
    """
    n = 30

    datetime_index = []
    for i in range(1, n + 1, 1):
        datetime_index.append(datetime(2020, 1, i))

    seed(seed_number)
    data = [15 + 5 * sin(x_multiplier * x) + randn() for x in range(n)]

    return Series(data=data, index=datetime_index, name=name)
