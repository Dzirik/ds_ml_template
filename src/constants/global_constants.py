"""
File for storing global constants.
"""

# Environmental variables

ENV_DASH_DEBUG_MODE = "ENV_DASH_DEBUG_MODE"
ENV_DASH_DOCKER = "ENV_DASH_DOCKER"
ENV_CONFIG = "ENV_CONFIG"
DEFAULT_CONFIG = "python_local"  # "python_repo"
ENV_LOGGER = "ENV_LOGGER"
DEFAULT_LOGGER = "logger_file_limit_console"
ENV_RUNNING_UNIT_TESTS = "ENV_RUNNING_UNIT_TESTS"
DEFAULT_RUNNING_UNIT_TESTS = "False"

# Special variables

# When a bad logger config file is put into logger reading, then a special logger
# is created and the message is added to the following file.
# The file name should be the same as in config files for logger!
SPECIAL_LOGGER_FILE_NAME = "../../logs/logger_file_limit_console.log"

# Folders widely used in repository

FOLDER_CONFIGURATIONS = "configurations"

F = "f"  # fit function in transformers
FP = "fp"  # fit_predict function in transformers
P = "p"  # predict function in transformers
I = "i"  # inverse function

# Colors for visualizations
COLORS = {
    "line": ["#0f0f0f", "#011936", "#2F3E46", "#354F52"],  # black, oxford  blue, charcoal, dask slate gray
    "fill": ["#087E8B", "#99AA38", "#F58A07", "#F7F5FB"],  # metallic seaweed (B), citron (G), dark orange, ghost white
    "error": ["#ED254E", "#C81D25"],  # red crayola, lava
    "dot": ["#99AA38", "#F9DC5C", "#ACD2ED"],  # green, yello, blue, grass "#3F681C", red crayola "#ED254E"
    "paper_background": {"color": "#000000", "opacity": 0},  # outside of the plot, black with 0 opacity
    "grid_background": {"color": "#858B97", "opacity": 0.4},
}

ATTR_DATE_TIME = "DATETIME"
