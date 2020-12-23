"""
File for storing global constants.
"""

# Environmental variables

ENV_DASH_DEBUG_MODE = "ENV_DASH_DEBUG_MODE"
ENV_DASH_DOCKER = "ENV_DASH_DOCKER"
ENV_CONFIG = "ENV_CONFIG"
DEFAULT_CONFIG = "python_repo"
ENV_LOGGER = "ENV_LOGGER"
DEFAULT_LOGGER = "logger_file_console"

# Special variables

# When a bad logger config file is put into logger reading, then a special logger
# is created and the message is added to the following file.
# The file name should be the same as in config files for logger!
SPECIAL_LOGGER_FILE_NAME = "../../reports/python_log.log"

# Folders widely used in repository

FOLDER_CONFIGURATIONS = "configurations"
