"""
Exceptions

SW Pattern: Strategy
"""

from typing import Optional

from src.exceptions.custom_exception import CustomException
from src.exceptions.development_exception import NoProperOptionInIf
from src.utils.envs import Envs
from src.utils.logger import Logger


class ExceptionExecutioner:
    """
    Class for executing and logging custom exceptions in unified way.

    The trouble was with running unit tests with makes - Logger does not work. Logging can be excluded from tests.
    """
    #TODO write to documentation that exceptions have to be excluded from tests

    def __init__(self, exception_class: CustomException) -> None:
        self._exception_class = exception_class

    def log_and_raise(self, description: Optional[str] = None) -> None:
        """
        Logs the exception and after raises it.
        :param description:
        """
        env = Envs()
        if description is None:
            exc = self._exception_class()
        else:
            exc = self._exception_class(description)
        if env.get_running_unit_tests() == "False":
            Logger().error(exc.get_description())
        raise exc


if __name__ == "__main__":
    ExceptionExecutioner(NoProperOptionInIf).log_and_raise()
    print("THIS SHOULD NOT BE PRINTED")
