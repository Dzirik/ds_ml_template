"""
Exceptions for code development purpose.
"""

from src.exceptions.custom_exception import CustomException
from src.utils.logger import Logger


class DevelopmentException(CustomException):  # type:ignore
    """
    Exception for developing group (e.g TDD).
    """

    def __init__(self) -> None:
        CustomException.__init__(self, "Development.")


class NoProperOptionInIf(DevelopmentException):
    """
    Exception for not proper option in if clause.
    """

    def __init__(self, description: str = "Option is not present in the if statement.") -> None:
        DevelopmentException.__init__(self)
        self._code = 301
        self._description = description


if __name__ == "__main__":
    print("Before exception")
    # possible usages:
    # - raise NoProperOptionInIf
    # - raise NoProperOptionInIf("My new message")
    # - exc = NoProperOptionInIf()
    #   raise exc
    # - exc = NoProperOptionInIf("My new message")
    #   raise exc
    print("Creating exception.")
    exc = NoProperOptionInIf("This is my exception")
    print("Logging exception.")
    Logger().error(exc.get_description())
    print("Raising exception.")
    raise exc
