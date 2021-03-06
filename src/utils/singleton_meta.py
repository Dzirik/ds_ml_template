"""
Singleton
"""
from typing import Dict, Any


class Singleton(type):
    """
    Singleton base class.
    """
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
