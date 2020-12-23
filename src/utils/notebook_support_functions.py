"""
Functions for supporting notebook functionality.
"""

from json import loads
from re import search

from IPython.display import HTML
from ipykernel.connect import get_connection_file
from notebook.notebookapp import list_running_servers
from requests import get


def get_notebook_name() -> str:
    """
    Return the full path of the jupyter notebook.
    :return: str. Notebok name.
    """
    kernel_id = search("kernel-(.*).json", get_connection_file()).group(1)  # type:ignore
    servers = list_running_servers()
    for server in servers:
        response = get(f"{server['url']}api/sessions", params={'token': server.get('token', '')})
        for load in loads(response.text):
            if load['kernel']['id'] == kernel_id:
                relative_path = load['notebook']['path']
                return str(relative_path.split("/")[-1])
    return ""


def create_button() -> HTML:
    """
    Creates a button which hides the code.
    :return: HTML. The button.
    """
    return HTML('''<script>
    code_show=true;  
    function code_toggle() { 
    if (code_show){ 
    $('div.input').hide(); 
    } else { 
    $('div.input').show(); 

    } 
    code_show = !code_show 
    }  
    $( document ).ready(code_toggle); 
    </script> 
    <form action="javascript:code_toggle()"><input type="submit" value="Show/Hide code"></form>''')
