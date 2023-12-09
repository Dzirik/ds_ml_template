"""
Functions for helping with date and/or time manipulation within the project.
"""

from datetime import datetime

from numpy.random import randint


def add_zeros_in_front_and_convert_to_string(number: int, order: int) -> str:
    """
    Adds zeros in front of the number, and return without the first digit.
    Usage: number=5, order=100 -> "05"
    :param number: int.
    :param order: int.
    :return: str.
    """
    return str(number + order)[1:]


def convert_datetime_to_string_date(now: datetime = datetime.now(), add_micro: bool = False, sep: str = "-") -> str:
    """
    Converts now to string format yyyy-dd-yy-hh-mm-ss<-micro>.
    :param now: datetime. Date in datetime format. Default value is datetime.now().
    :param add_micro: bool. If to add microseconds too.
    :param sep: str. Separator in between time data. Default is "-".
    :return: str.
    """
    year = now.year
    month = add_zeros_in_front_and_convert_to_string(now.month, 100)
    day = add_zeros_in_front_and_convert_to_string(now.day, 100)
    hour = add_zeros_in_front_and_convert_to_string(now.hour, 100)
    minute = add_zeros_in_front_and_convert_to_string(now.minute, 100)
    second = add_zeros_in_front_and_convert_to_string(now.second, 100)
    micro = add_zeros_in_front_and_convert_to_string(now.microsecond, 1000000)
    if add_micro:
        now_str = f"{year}{sep}{month}{sep}{day}{sep}{hour}{sep}{minute}{sep}{second}{sep}{micro}"
    else:
        now_str = f"{year}{sep}{month}{sep}{day}{sep}{hour}{sep}{minute}{sep}{second}"
    return now_str


def create_datetime_id(now: datetime = datetime.now(), add_micro: bool = False) -> str:
    """
    Creates date time id in the form yyyy-dd-mm-hh-mm-ss<-micro>_xxx where xxx is randomly generated integer.
    :param now: datetime. Date in datetime format. Default value is datetime.now().
    :param add_micro: bool. If to add microseconds too.
    :return: str.
    """
    return f"{convert_datetime_to_string_date(now, add_micro)}_" \
           f"{add_zeros_in_front_and_convert_to_string(randint(0, 999), 1000)}"


def convert_string_date_to_datetime(now_str: str) -> datetime:
    """
    Converts string of the format yyyy-dd-yy-hh-mm-ss to datetime format.
    :param now_str: str. String of the format yyyy-dd-yy-hh-mm-ss.
    """
    numbers = [int(string_number) for string_number in now_str.split("-")]
    date = datetime(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5])
    return date


if __name__ == "__main__":
    d = datetime.now()
    print(convert_datetime_to_string_date(d, False))
    print(convert_datetime_to_string_date(d, True))
