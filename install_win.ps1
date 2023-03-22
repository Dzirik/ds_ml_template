#Setting execution policy:
# https://www.easy365manager.com/script-ps1-cannot-be-loaded-because-running-scripts-is-disabled-on-this-system/
# Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser.

Copy "make_config_template.mk" "make_config.mk"
Copy ".\configurations\python_repo.conf" ".\configurations\python_personal.conf"
Copy ".\notebooks\template\template_notebook_final.py" ".\notebooks\raw\playground_notebook.py"

# error in powershell
# make create-venv