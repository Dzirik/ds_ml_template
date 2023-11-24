"""
Data structure for config.
"""
from typing import NamedTuple, List, Dict, Union


class Path(NamedTuple):
    """
    Configuration tuple for paths to data files.
    """
    archive_data: str
    auto_notebooks: str
    external_data: str
    interim_data: str
    logs_data: str
    processed_data: str
    raw_data: str
    results_data: str
    models: str


class DatabaseCredentials(NamedTuple):
    """
    Log in credentials for data base.
    """
    provider: str
    user: str
    password: str
    host: str
    port: int
    db_name: str


class DashSettings(NamedTuple):
    """
    Configuration tuple for the dash setting.
    """
    header_link_color: str
    header_font_weight: str
    navbar_background_color: str
    path_to_image: str


class Dash(NamedTuple):
    """
    Configuration tuple for the dash.
    """
    sidebar_config: List[List[List[str]]]
    list_of_pages: Dict[str, List[str]]
    sidebar_style: Dict[str, Union[int, str]]
    content_style: Dict[str, str]
    sett: DashSettings


class ParamNotebookExecution(NamedTuple):
    """
    Configuration tuple for the parameterized notebook execution.
    """
    use_default: bool
    convert_to_html: bool
    ntb_path: str
    output_folder: str
    notebook_executioner_params: List[Dict[str, Union[float, str]]]


class ConfigData(NamedTuple):
    """
    Overall configuration tuple for everything.
    """
    name: str
    path: Path
    db_cred: DatabaseCredentials
    dash: Dash
    param_ntb_execution: ParamNotebookExecution
