"""
Functions for helping with date and/or time manipulation within the project.
"""

from datetime import datetime


def add_zeros_in_front_and_convert_to_string(number: int, order: int) -> str:
    """
    Adds zeros in front of the number, and return without the first digit.
    Usage: number=5, order=100 -> "05"
    :param number: int.
    :param order: int.
    :return: str.
    """
    return str(number + order)[1:]


def convert_datetime_to_string_date(now: datetime = datetime.now()) -> str:
    """
    Converts now to string format yyyy-dd-yy-hh-mm-ss
    :param now: datetime. Date in datetime format. Default value is datetime.now().
    :return: str.
    """
    year = now.year
    month = add_zeros_in_front_and_convert_to_string(now.month, 100)
    day = add_zeros_in_front_and_convert_to_string(now.day, 100)
    hour = add_zeros_in_front_and_convert_to_string(now.hour, 100)
    minute = add_zeros_in_front_and_convert_to_string(now.minute, 100)
    second = add_zeros_in_front_and_convert_to_string(now.second, 100)
    now_str = f"{year}-{month}-{day}-{hour}-{minute}-{second}"
    return now_str


def convert_string_date_to_datetime(now_str: str) -> datetime:
    """
    Converts string of the format yyyy-dd-yy-hh-mm-ss to datetime format.
    :param now_str: str. String of the format yyyy-dd-yy-hh-mm-ss.
    """
    numbers = [int(string_number) for string_number in now_str.split("-")]
    date = datetime(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5])
    return date
