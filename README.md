# !!!!THIS IS THE FIRST PUBLISH VERSION WHICH IS STILL IN DEVELOPMENT!!!!

**The purpose is to serve as a repository template for machine learning and data science projects.**

**The documentation needs to be finished and the functionality has to be checked.**

# Template
Template for ML and DS development

# Configuration Files

Local copies of configuration files need to be done:
- *make_config.mk* from template *make_config_template.mk*.
- *configurations/local.conf* from *configurations/local_template.conf*.

# TODAY

Today is a nice day!
    - It is sunny!
    - It is weekend!


# TOMORROW

Tomorrow will not be that nice day.
    - It will be raining.
    
# END OF FILE

# Jupyter Notebook

## Jupytext

[Jupytext library](https://github.com/mwouts/jupytext) enables to store notebooks in plain text, markdown, instead of 
raw *.ipynb.

**All notebooks in this repository are stored in jupytext .py format instead of .ipynb.**

# Devops

## Mypy

General information:
- A static type checker for Python. 
- Reduces the execution time because types are chekced before running.

Running mypy:
- `mypy --strict your_code.py --config-file mypy.ini`
- Makes commands can be used as well, please see Make chapter.

Examples of static typing:
- Variable 
    - `x: int = 7`
- Function
    - `def my_function(num: int, dictionary: Dict[str, str]) -> str:`
- Please see [cheat sheet](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html#variables) for more options.

Following situations are excluded (please fill those examples shortly in make file documentation):
- Please fill in situations you would like to exclude and the reason.

Customisation:
- General customisation can be done in [mypy configuration file](https://mypy.readthedocs.io/en/latest/config_file.html).

Tips and tricks:
- If you don't know which type should a callable be, write `reveal_type(callable)` in the code and run mypy. The 
PowerShell prompt will specify the type that you should use.

Ignoring errors:
- Line of code: By adding `# type:ignore`
- *.py file: By adding `# type:ignore` at the beginning of file (not recommended)
- Mypy configuration file: By disabling the regarding setting of mypy such as `ignore_missing_imports=False`. However, this is 
not encouraged because it can suppress all the errors that can be helpful when debugging and also they could be caused 
by another reason than you want to ignore but gives the same error.

# Make

## Make Documentation

Please after the secion name corresponding to make command name, add @ in front of the caption of header printed
in command line. The printing function then makes it a bit nicer.

### help
@HELP

Utils:
 - make hello - Prints hello message.
Virtual Environment:
 - make create-venv: Creates/updates virtual environment named .venv.
 - .venv\Scripts\activate : Activates virtual environment. Has to be done by hand.
 - make freeze-in: Freezes libraries settings and update requirements.txt file with that. You have 
                    to be inside the virtual environment.
 - make freeze-out: Freezes libraries settings and update requirements.txt file with that. You have
                     to be in base, eg. outside the virtual environment.
                     
Source Code Quality:
 Checks * for src, tests folders.
 - make mypy: Checks mypy.
 - make lint: Checks pylint.
 - make lint-dup: Checks pylint with duplications.
 - make test: Checks pytest and doctest.
 - make all -i: Checks mypy, pylint, pytest and doctest.
 
One File Code Quality:
 Checks * for one source file FILE_NAME in FILE_FOLDER and for its pytest and 
 doctest variant in tests folders.
 - make mypy-f: Checks mypy.
 - make lint-f: Checks pylint.
 - make test-f: Checks pytest and doctest.
 - make all-f: Checks mypy, pylint, pytest and doctest.
 
Notebooks Code Quality:
 Checks * for all notebook *.py files in notebooks/final and notebooks/documentation.
 - make mypy-ntb: Checks mypy.
 - make lint-ntb: Checks pylint.
 - make all-ntb -i: Checks both mypy and pylint.
 
Coverage:
 Performs coverage.
 - make cover: Creates complete coverage report for the repository in .\coverage folder.
 - make cover-log: Saves overall coverage ratio into cover_log.csv file.
Dash
 Runs pyton dash application.
 - make run-dash: Runs the dash locally.
 - make build-run-dash-docker: Builds and runs the dash in docker. 
 - make build-dash-docker: Builds the dash docker.
 - make run-dash-docker: Runs the dash docker.

### hello
@HAIL TO YOU, HERO!!!
@CONGRATULATIONS TO YOU RUNNING YOUR FIRST MAKE COMMAND!!

### create-venv
@CREATES OR UPDATES VIRTUAL ENVIRONMENT

### freeze-out
@FREEZES REQUIREMENTS IF OUTSIDE THE VIRTUAL ENVIRONMENTS

### freeze-in
@FREEZES REQUIREMENTS IF INSIDE THE VIRTUAL ENVIRONMENTS

### mypy-no-clear
@DOES MYPY IN SRC TYPE CHECKING
MyPy type checking in src and tests folders.
Excluded from checking (for more information see documentation README.md file):
 - First excluded item ...
 - Second excluded item ...
A line can be excluded from checking by putting following in the end of the line: # type:ignore
@

### lint-no-clear
@LINTERS SRC
Lint checking in src and tests folders.
Excluded from checking (for more information see documentation README.md file):
 - Duplicate code - for that is the separate make.
A line or lines of code can be excluded from checking by:
 - `# pylint: disable=<name_of_error>` at the beginning of the code part and
 - `# pylint: enable=<name_of_error>` at the end. 
@

### lint-dup-no-clear
@LINTERS SRC WITH DUPLICATIONS
Lint checking in src and tests folders.
Excluded from checking (for more information see documentation README.md file):
 - Duplicate code - for that is the separate make.
A line or lines of code can be excluded from checking by:
 - `# pylint: disable=<name_of_error>` at the beginning of the code part and
 - `# pylint: enable=<name_of_error>` at the end.
@

### test-no-clear
@DOES PYTEST AND DOCTEST FOR ALL FILES

### mypy-f
@DOES MYPY OF FILE AND ITS TEST VERSION FOR FOLLOWING FILE

### test-f
@DOES PYTEST AND DOCTEST FOR FOLLOWING FILE

### lint-f
@LINTERS FOLLOWING FILE

### mypy-ntb-no-clear
@DOES MYPY IN NOTEBOOKS FOLDER
MyPy type checking in notebooks/final and notebooks/documentation.
Excluded from checking (for more information see documentation README.md file):
 - First excluded item ...
 - Second excluded item ...
A line can be excluded from checking ty putting following in the end of the line: # type:ignore
@

### lint-ntb-no-clear
@LINTERS NOTEBOOKS
Lint checking in notebooks/final and notebooks/documentation.
Excluded from checking (for more information see documentation README.md file):
 - Duplicate code - for that is the separate make.
A line or lines of code can be excluded from checking by:
 - `# pylint: disable=<name_of_error>` at the beginning of the code part and
 - `# pylint: enable=<name_of_error>` at the end. 
@

### cover-base
@DOES COVERAGE
Does complete coverage for the repository. The report can be found in coverage folder, which is 
excluded from repository sync.

### cover-save
@SAVES OVERALL COVERAGE RATIO INTO cover_log.csv FILE

### run-dash
@RUNS DASH LOCALLY

Dash is running on: http://127.0.0.1:8050/

### build-run-dash-docker
@BUILDS AND RUNS DASH IN DOCKER

Dash is running on: http://127.0.0.1:8050/

### build-dash-docker
@BUILDS DASH IN DOCKER

### run-dash-docker
@RUNS DASH IN DOCKER
Dash is running on: http://127.0.0.1:8050/
@

# Summary
