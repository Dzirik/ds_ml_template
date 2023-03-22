########################################################################################################################
#
# Instalation script for linux platform.
# - Creates only locally used files.
# - Creates linux installation requirements from "requirements.txt" by removing windows specific libraries.
# - Creates virtual environment.
#
# Other commands that work:
# - source .venv/bin/activate; # activates the virtual environment, but doesn't work inside ().
# - make all -i; # tests the repository code base by running all tests, so make commands work
########################################################################################################################

clear

# Creating copies of the files needed for the usage, but excluded from sync with remote (.gitignore).
cp make_config_template.mk make_config.mk;
cp ./configurations/python_repo.conf ./configurations/python_personal.conf;
cp ./notebooks/template/template_notebook_final.py ./notebooks/raw/playground_notebook.py;

# Creates/rewrites "requirements_linux.txt" file by content of "requirements.txt" file with ignored windows-only 
# libraries.
# The "requirements_linux.txt" is excluded from syncing with remote, it is created for local usage only (.gitignore).
{
  (cat requirements.txt |     # outputs content of "requirements.txt" file
    grep -v pywin32     |     # rejects line with "pywin32"
    grep -v pywinpty          # rejects line with "pywinpty"
  ) > requirements_linux.txt; # creates/overwrites "requirements_linux.txt" file with filtered content
}

# installing virtual environment
python ./src/utils/make_print_documentation.py create-venv
pip install virtualenv==20.19.0
test -d .venv || virtualenv .venv
( \
     # source .venv/bin/activate;\ # it didnt work here
     . ./.venv/bin/activate;\
     pip install -r requirements_linux.txt;\
)