# ds_ml_template

Repository template for data science and machine learning projects.

<a name="table-of-content"></a>
# Table of Content
- [Introduction](#introduction)  
- [Installation Windows](#installation-windows)
    - [Prerequisites](#prerequisites-windows)
    - [Repository Set Up Step By Step](#repository-set-up-step-by-step-windows)
    - [Fast Repository Set Up](#fast-repository-set-up-windows)
- [Installation Linux](#installation-linux)  
    - [Prerequisites](#prerequisites-linux)
    - [Repository Set Up](#repository-set-up-linux) 
- [Files and Folders Structure](#files-and-folders-structure)  
- [Code Quality](#code-quality)  
    - [Mypy](#mypy)
    - [Pylint](#pylint)
    - [Pytest](#pytest)
- [Makefile](#makefile)
- [Low-Level Tools](#low-level-tools)
    - [Environment Variables](#environmental-variables)
    - [Environment Configuration](#environment-configuration)
    - [Timer](#timer)
    - [Logger](#logger)
- [Jupyter Notebooks](#jupyter-notebooks)  
    - [Jupytext Library](#jupytext-library)
    - [Jupyter Notebooks Folders](#jupyter-notebooks-folders)
    - [Jupyter Notebook Template](#jupyter-notebook-template)
    - [Parameterized Notebooks](#parameterized-notebooks)
- [Ready-to-Use Functionality](#ready-to-use-functionality)
    - [Visualisations](#visualisations)
    - [Data Frame Explorer](#data-frame-explorer)
    - [Sample Data Set](#sample-data-set)
    - [Time One Hot Encoding](#time-one-hot-encoding)
- [Dash](#dash)

<a name="introduction"></a>
# Introduction
[ToC](#table-of-content)  

This repository serves as a **template for Data Science and Machine Learning projects in Python 3.x ecosystem**. 
It contains basic functionality and tools for easy and smooth work.

<a name="installation-windows"></a>
# Installation Windows
[ToC](#table-of-content)  

<a name="prerequisites-windows"></a>
## Prerequisites
[ToC](#table-of-content) 

Working set up:
- Windows 10 operating system.
  
- Installing Anaconda.
    - Download Anaconda from the official site.
    - Follow the installation guide, **do not let Anaconda** *Add Anaconda to my PATH environment variable*.
    - When finished, follow the instruction from the video */docs/2022-01-21_anaconda_set_up.mp4*.
    - The current versions are the following:
        - Anaconda File Version: Anaconda3-2021.11-Windows-x86_64 (I have it saved).
        - `conda --version`: 4.10.3
        - `python --version`: 3.9.7

- Installing Tensorflow prerequisites for GPU usage.
    1. Install MS Visual studio version 2019.
    2. Install Cuda Toolking 11.2 (cuda_11.2.0_460.89_win10.exe).
    3. Install cuDNN compatible with 11.2, versn 8.1 was chose (cudnn-11.2-windows-x64-v8.1.0.77.zip) .
        - Extract folders *bin*, *include*, *lib* to *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2*.
    5. Add the following addresses to the Windows path:
        - *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2*
        - *C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp*
    6. The compatible TenforFlow library is in *requirements.txt* and will be automatically installed during 
       [*Repository Set Up Step By Step*](#repository-set-up-step-by-step-windows) chapter.

- Installing cygwin to be able to run make commands on Windows operating system.
    - Follow the video */docs/2022-01-21_cygwin_make_set_up.mp4*.
    - Used versions of tools:
        - Binutils: 2.37-1
        - Make: 4.2.1-2

<a name="repository-set-up-step-by-step-windows"></a>
## Repository Set Up Step By Step
[ToC](#table-of-content)  

Clone the repository, move to the main repository folder location, and do the following to set it up properly:

1. Make a copy of a file *make_config_template.mk* and give it a name *make_config.mk*.
2. Open a console in the repository folder and put there `make hello` command. A welcome and congratulation message 
will show up.
3. In the console, run `make` or `make help`; a help showing all possible make commands will display.
4. In the console, run `make create-venv` to create a virtual environment with all necessary libraries.
5. Set up repository configuration. There are two configuration files available - */configurations/python_local.conf* and
*/configurations/python_repo.conf*. To change the default configuration, change the variable 
`DEFAULT_CONFIG` in location */src/global_constants*. In general, the code in the file */src/utils/envs.py* was developed
to set up global variables (not only) for configurations. A new configuration file can be created by the following steps:
    - Make a copy of *python_repo.conf* file.
    - Give it a suitable name (one name is already excluded for syncing with GitHub and then ready for your personal 
      usage - *python_personal.conf*).
    - Change a variable `DEFAULT_CONFIG` in location */src/global_constants* to make it the default configuration file.  
   
    The default value of `DEFAULT_CONFIG` is set to `python_personal`. So for everything to work correctly, please create 
*/configurations/python_personal.conf* file.
6. In the console, activate the virtual environment `.venv\Scripts\activate`.
7. Test the repository code running `make all -i`. All tests should run without any problem.
8. Create a playground notebook */notebooks/raw/playground_notebook.py*. This one is excluded from sync, so it can be 
used for testing and development.
9. The TensorFlow installation can be checked by running the notebook 
*/notebooks/documentation/tensorflow_installation_and_testing.py*.

<a name="fast-repository-set-up-windows"></a>
## Fast Repository Set Up
[ToC](#table-of-content)  

NOTE: For detailed information, please read the 
[*Repository Set Up Step By Step*](#repository-set-up-step-by-step-windows) chapter. This version is for fast set up.

1. Run Windows PowerShell as Administrator.
2. Go to the repository folder.
3. Run `.\install_win.ps1` PowerShell script. It creates all the necessary files.
   - If you do not have a permission, set up temporal or user permission as written 
   [here](https://www.easy365manager.com/script-ps1-cannot-be-loaded-because-running-scripts-is-disabled-on-this-system/).
4. Run Anaconda prompt and move to the main repository folder.
5. Run `make create-venv` to create a virtual environment.
6. Activate the virtual environment `.venv\Scripts\activate`.
7. Test the repository code quality by running `make all -i`. All tests should run without any problem.

NOTE: The TensorFlow installation can be checked by running the notebook 
*/notebooks/documentation/tensorflow_installation_and_testing.py*.

<a name="installation-linux"></a>
# Installation Linux
[ToC](#table-of-content)  

**NOTE: This part is not complete and not properly tested, because my main development environment is Windows. So 
mainly some code snippets from testing on Ubuntu 22.04 without GPU.**

<a name="prerequisites-linux"></a>
## Prerequisites
[ToC](#table-of-content) 

Set Up:
- Ubuntu 22.04
- Anaconda of the same version as in [*Windows Installation*](installation-windows) section.
- No GPU support was tested for TensorFlow.
- Make installation: `sudo apt-get install make`.

<a name="repository-set-up-linux"></a>
## Repository Set Up
[ToC](#table-of-content) 

Clone the repository, move to the main repository folder location, and do the following to set it up properly:
1. Run `sh install_linux.sh` to:
    - Make a copy of a file *make_config_template.mk* and give it a name *make_config.mk*.
    - Create *requirements_linux.txt* by removing windows specific libraries from *requirements.txt*. The
      file is excluded from sync with remote in *.gitignore*.
    - Install the virtual environment .venv.
    - Copy default repository configuration to personal configuration (see 
    - [Windows Repository Set Up](#repository-set-up-windows) for more information).
    - Copy */notebooks/template/template_notebook_repo.py* to  */notebooks/raw/playground_notebook.py*. This file is 
      excluded from sync with remote in *.gitignore*.
2. In the console, activate the virtual environment `source .venv\bin\activate`.
3. Run `make hello` command to see a welcome and congratulation message.
4. In the console, run `make` or `make help`; a help showing all possible make commands will display.
5. Test the repository code running `make all -i`. All tests should run without any problem. 
6. The TensorFlow installation can be checked by running the notebook 
*/notebooks/documentation/tensorflow_installation_and_testing.py*. 

<a name="files-and-folders-structure"></a>
# Files and Folders Structure
[ToC](#table-of-content)

## Folders
[ToC](#table-of-content)  

- *assets*
    - A folder for storing additional content relevant to the project (pictures, ...).
- *configurations*
    - A folder for configuration files.
- *coverage*
    - A folder for coverage files. The content of this file is excluded from version control synchronisation.
- *data*
    - A folder for storing the important data. In general, **it is not a good practice to keep large data files in the
  version control system**. As a consequence, the data can be stored outside. The place for storing data can be specified
  using configuration files, but the structure and folder names are assumed to be the same as in the repository.
    - *archive*
        - A folder for important old data.
    - *external*
        - Data from external sources (dictionaries, synonyms, ...).
    - *interim*
        - Intermediate data between raw and fully processed. Not raw and also not finalized yet.
    - *logs*
        - Folder for logs, mainly from TenforFlow.
    - *processed*
        - Final data after all the preprocessing, merging, cleaning, transformation, enrichment, feature engineering etc.
    - *raw*
        - For raw data storage. This directory should be read only.
    - *results*
        - A folder for storing results from simulation and other similar activities.
    - *Note*: In pycharm mark it as an excluded folder.
- *docs*
    - For documentation. Please use [Sphingx](www.sphinx-doc.org) for this. Html versions of
  final notebooks are stored here too.
    - *Note*: In pycharm mark it as an excluded folder.
- *logs*
    - For log files.
- *models*
    - Storage for created models in binaries, pickles, ...
    - Intermediate results.
    - From long term perspective should be stored somewhere else.
    - Model names/folder should contain some metadata information - date of execution, data, size, ... see reporting.
- *notebooks*
    - Only *.py files from [jupytext](https://github.com/mwouts/jupytext/) package are held here. 
  Please see section [*Jupyter Notebooks*](#jupyter-notebooks) for more information.
    - *documentation*
        - Documentation notebooks are stored here. Html versions of notebooks are stored in */docs* directory.
    - *final* 
        - For final notebooks.
    - *raw*
        - A folder for preparation notebooks. Nothing in this folder matters with respect to the delivery.
    - *template*
        - Notebook templates are stored here.
    - *temporal*
        - All *.py*, *.ipynb*, and *.html* files in this folder are excluded from synchronisation. This folder is used for 
  keeping notebooks for simulations and its results.
- *references*
    - Manuals, explanatory materials.
    - *Note*: In pycharm mark it as an excluded folder.
- *reports*
    - Generated analysis reports (html, pdf, LaTex, docx, ...)
- *src*
    - A main folder for storing source code with folders for respective tasks and areas of interest.
    - *apps*
        - Code for Dash application.
    - *constants*
        - Folder for storing constants. One file is present from the very beginning - *global_constants.py*, which is 
          a file for storing global constants like attributes names. Please use capitals for these (eg. ATR_DATE).
    - *data*
        - Code for reading, formatting, and manipulating the data.
    - *exceptions*
        - Code for custom exceptions.
    - *external*
        - Code taken from external resources. **It is excluded from code quality check**.
    - *jobs*
        - Folder for code doing overall jobs tasks.
    - *models*
        - Code for models.
    - *pipelines*
        - Code for whole pipelines.
    - *reporting*
        - Code used for reports creation.
        - *Note*: Date and time, size, data, ...
    - *scripts*
        - A folder for scripts.
    - *scripts_production*
        - A folder for production scripts.
    - *transformations*
        - Code for data transformations.
    - *utils*
        - Code used across the whole project.
    - *visualisations*
        - Code for plots creation.
    - *Note*: In pycharm mark it as a source folder.
- *tests*
   - For storing tests to file. The structure of this folder is similar to src, but the folders start with *tests_*. 
   Test files corresponding to the main code are stored in corresponding folders.

## Files
[ToC](#table-of-content)  

- *.gitignore*
    - Excludes some files and folder. The base is standard github-based file with some additional features. 
    For more information see the file. 
- *.pylintrc*
    - Base configuration file for Python *pylint* linter. For more information see the file.
- *.pylintrc_dup*
    - Configuration file for Python *pylint* linter for duplicities check. For more information see the file.
- *.pylintrc_ntb*
    - Configuration file for Python *pylint* linter for notebooks. For more information see the file.
- *app.py*
    - File with application for Dash dashboarding tool.
- *index.py*
    - Runs Dash application.
- *Makefile*
    - For *make* commands.
- *pytest.ini*
    - Base configuration file for *pytest* library. For more information see the file.
- *README.md*
    - Documentation file.
- *requirements.txt*
    - Requirements for libraries for environment. Generated with `pip freeze > requirements.txt`.

<a name="code-quality"></a>
# Code Quality
[ToC](#table-of-content)  

<a name="mypy"></a>
## Mypy
[ToC](#table-of-content)  

Mypy is a static type checker for Python. Execution time is reduced because types are checked before running.

To run mypy:
- Run the following command in command line `mypy --strict your_code.py --config-file mypy.ini`.
- See [*Makefile*](#makefile) for available commands.
    
A [mypy configuration file](https://mypy.readthedocs.io/en/latest/config_file.html) allows you to modify the setting for
 mypy. The file for this repository is */mypy.ini*.

Examples of static typing:
- Variable 
    - `x: str = "test"`
- Function
    - `def stringify(num: int) -> str: return str(num)`

See the [cheat sheet](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html#variables) for more annotation examples
 and options.
 
 *Tip: If you don't know which type should a callable be, write* `reveal_type(callable)` *in the code 
 and run mypy. The prompt will specify the type that you should use.*
 
 It is possible to ignore errors in 3 different ways:
 * On a line of code: By adding `# type:ignore` on that line.
 * On a .py file: By adding `# type:ignore` to the beginning of the file.
 * On config file: Disabling the settings of mypy such as `ignore_missing_imports=False`. However,
 ignoring in the file is not encouraged because it can suppress all the errors that can be helpful when debugging and also 
 they could be caused by another reason you want to ignore but give the same error.

**The following situations are excluded from checking. For more information, see the description below.**
- Pytest fixtures *Untyped decorator makes function "test_density" untyped :* in files *tests_**. 
It is excluded using:
    - `@pytest.mark.parametrize("df, type_of_normalisation", # type:ignore`
- Pytest fixture *Class cannot subclass 'BaseTransformer' (has type 'Any')*. 
It is excluded in the following style in the child class:
    - `class MyTransformer(BaseTransformer): # type:ignore`

### Excluded Situations
[ToC](#table-of-content)  
 
 Two types of error are ignored by adding `# type:ignore` to the regarding the line of code
 * Untyped decorator makes function "test_density" untyped: This is caused by the missing annotation 
  while using pytest fixtures or parametrization. It is impossible to use static typing for these; therefore,
  they cause a missing annotation when mypy is used. For example:
  `@pytest.mark.parametrize("transformation_type", [FP, P])  # type:ignore`
  * Class cannot subclass 'BaseTransformer' (has type 'Any') : This is possibly caused by the inheritance 
  of a different class in a different directory. Additional information is gathered by using `disallow_any_unimported = true`
   which gives "Base type BaseTransformer becomes "Any" due to an unfollowed import". Refer 
   [follow imports by mypy](https://mypy.readthedocs.io/en/latest/running_mypy.html#follow-imports). 
   However, the real reason is not found. Several possible solutions that didn't solve the issue applied  
  are as follows:
    - Replace usage of `Any` with other built-in types.
    - Moving the inherited class into the same file.
    - Changing parameters from `Any` to other known ones.
    - Changing the number of parameters in the abstract file to be compatible with the overwritten file.
    - Disabling some of the error checking using config files such as `follow_imports=False`, 
    `follow_imports_for_stubs=False,True`, `disallow_untyped_defs=True,False`.  
    - Disabling all the inheritances in both existing files and inherited files.
    - Related materials can be found in the following links: [1](https://github.com/python/typeshed/issues/1544), 
    [2](https://github.com/python/typeshed/pull/1492), [3](https://github.com/python/typeshed/issues/1446), 
    [4](https://github.com/python/mypy/issues/4180), 
    [5](https://stackoverflow.com/questions/49888155/class-cannot-subclass-qobject-has-type-any-using-mypy)
    
    This issue is solved by adding `# type:ignore` to corresponding lines of code. For example:
    `class MyTransformer(BaseTransformer): # type:ignore`.

<a name="pylint"></a>
## Pylint
[ToC](#table-of-content)  

Pylint is a static code analysis tool that:
- looks for programming errors,
- helps enforce a coding standard,
- sniffs for code smells,
- offers simple refactoring suggestions.

To run Pylint:
- Run the following command in command line `pylint your_code.py --rcfile .pylintrc`.
- See [*Makefile*](#makefile) for available commands.

It can also be executed within Python:

    import pylint.lint 
    pylint_opts = ['--version'] # --rcfile for using the configuration file, see user guide.   
    pylint.lint.Run(pylint_opts)

It has the option to do parallel execution to speed up the performance of Pylint.

See the [user guide](http://pylint.pycqa.org/en/latest/user_guide/run.html) for more information. 

*Tip: To disable a warning for one single line (or block) you can type in the line or at the 
indentation level (e.g. a method or class) that you wish to disable the following 
comment:* `# pylint: disable=Warning-name-or-code`.

*Tip: When a warning is disabled by using `#pylint: disable=Warning-name-or-code.`, that particular type of checking
becomes disabled for the rest of the code starting from that line. Therefore, it is recommended to enable the same warning
after it is used for a specific part of code by using the command `#pylint: disable=Warning-name-or-code.`*

**The following situations were excluded from checking. For more information, see the description below.**
- Duplicate code. There are separate makes for that. Excluded from the main one.
- Parameters differ from overridden methods fit, fit_predict, predict. Because of problem in Base classes.  
    ```
    # pylint: disable=arguments-differ
    def fit(... 
        ... 
    def fit_predict(...  
        ...  
    def predict(...  
        ...  
        return ...
    # pylint: enable=arguments-differ
    ```
  
### Excluded Situations
[ToC](#table-of-content)  

Overview
- too-many-arguments: If a number of arguments are bigger than five, then pylint gives an error. This type of
error is suppressed by `#pylint: disable=too-many-arguments` because sometimes it is necessary to use more 
than five arguments, and it is a design choice.
-too-few-public-methods: If a number of functions in a class is less than 2, then pylint gives an error.
This error is suppressed by using `#pylint: disable=too-few-public-methods` because it is a design choice.
- arguments-differ: Some abstract function's parameters are overridden in other classes while overriding 
the functions. When the number of parameters differs from the inherited function, pylint gives 
arguments-differ error. No solution was found for such an error therefore errors are suppressed by
`# pylint: disable=arguments-differ` and enabled checking after function by `# pylint: enable=arguments-differ`.

**Note:** Both _...disable..._ at the beginning of the code block, and _...enable..._ at the end of the code block
are used within the minimal possible scope.

**IMPORTANT NOTE**: duplicate-code: Pylint gives an error for duplicate code. However, this code
can be written on purpose that way, and it is sometimes a design choice. Therefore, these errors can be
suppressed. However, disabling the duplicate-code error is not possible locally for pylint. 
Refer [here](https://github.com/PyCQA/pylint/issues/214) for the issue. It is solved by disabling 
duplicate-code errors globally. It is needed to be careful about next time usages. At the same time, another 
make command together with a separate pylint configuration file was created to check the code for duplicities.

<a name="pytest"></a>
## Pytest
[ToC](#table-of-content)  

Pytest is a framework that makes building simple and scalable tests easy.

To execute Pytest there are three ways:
- Test specific file by running `python -m pytest .\path\test_*.py`.
- See [*Makefile*](#makefile) for available commands for testing both the whole codebase and single file.
- (PyCharm) Through Python Integrated Tools (PIT):
    - Go to File -> Settings -> Tools -> PIT and change default tester to pytest.
    - Select the configuration pytest and run. 
    
There is an example in the repository there. There are three files:
- A file *./src/utils/leap_year.py* with a function that determines if the input is a leap year or not.
A file *./tests/tests_utils/test_leap_year.py* with functions that test the leap year function.
- A doctest file *./tests/tests_utils/test_leap_year.txt* which tests the execution.

After execution of the pytest in command prompt by `python -m pytest .\tests\tests_utils\test_leap_year.py`, 
the following message is displayed.

```
================================= test session starts ====================================

platform win32 -- Python 3.8.3, pytest-6.0.1, py-1.9.0, pluggy-0.13.1
rootdir: C:\<name_of_repository_file>, inifile: pytest.ini`<br/>
plugins: dash-1.16.0, cov-2.10.1, typeguard-2.12.0
collected 6 items

tests\test_leap_year.py ......

================================== 6 passed in 0.02s =====================================
```

In this case, all four tests were passed.

### Coverage 
[ToC](#table-of-content)  

A coverage is a metric that indicates the number of code lines covered by tests.  
To implement code coverage, the following possibilities were taken into account:
- Python [coverage](https://coverage.readthedocs.io/en/coverage-5.0.3/) library.
- Python [pytest-cov](https://pypi.org/project/pytest-cov/) library.
- Commercial solutions [sealights](https://www.sealights.io/agile-testing/test-metrics/python-code-coverage/).
To read more about coverage, please see the Wikipedia page for  
[modified condition/decision coverage](https://en.wikipedia.org/wiki/Modified_condition/decision_coverage).

Of all the options above, it was decided to use Pytest's coverage extension, pytest-cov.
- [pytest-cov](https://pypi.org/project/pytest-cov/). The main reason for this choice is the already use of Pytest 
within the repository, implying that a fast implementation of coverage was possible. 
Another reason for this choice is that pytest-cov can additionally calculate the branch coverage.
The used coverage within this repository has been set up to use branch coverage, which checks every line of code and
all possibilities of conditional statements if they were tested.
- See [*Makefile*](#makefile) for available commands.

Usage:
- To view a specific coverage run for a test, open the html file corresponding to the test, which is located in the */coverage* 
folder. This file will open in the browser and allow the user to specifically see which lines are being used and which are not.

<a name="makefile"></a>
# Makefile
[ToC](#table-of-content)  

Several make commands were created to make the high level work easier. Please follow
this chapter, because the documentation here is print together with the execution
of the respective make command. After the section name corresponding to make command name, 
please add @ in front of the caption of header printed in command line. The printing function 
then makes it a bit nicer.

## Make Documentation

### help
@HELP

Utils:
 - make hello: Prints hello message.

Virtual Environment:
 - make create-venv: Creates/updates virtual environment named .venv.
 - .venv\Scripts\activate : Activates virtual environment. It has to be done by hand.
 - make freeze-in: Freezes libraries settings and update requirements.txt file with that. You have 
                    to be inside the virtual environment.
 - make freeze-out: Freezes libraries settings and update requirements.txt file with that. You have
                     to be in base, e.g. outside the virtual environment.
                     
Source Code Quality:
 Checks * for src and tests folders.
 - make mypy: Runs mypy.
 - make lint: Runs pylint.
 - make lint-dup: Runs pylint with duplications.
 - make test: Runs pytest and doctest.
 - make all -i: Runs mypy, pylint, pytest and doctest.
 
One File Code Quality:
 Checks * for one source file FILE_NAME in FILE_FOLDER and its pytest and 
 doctest variant in tests folders. The FILE_NAME and FILE_FOLDER constants are specified in /make_config.mk file.
 - make mypy-f: Runs mypy.
 - make lint-f: Runs pylint.
 - make test-f: Runs pytest and doctest.
 - make all-f: Runs mypy, pylint, pytest and doctest.
 
Notebooks Code Quality:
 Checks * for all notebook *.py files in notebooks/final and notebooks/documentation.
 - make mypy-ntb: Runs mypy.
 - make lint-ntb: Runs pylint.
 - make all-ntb -i: Runs both mypy and pylint.
 
Coverage:
 Performs coverage.
 - make cover: Creates complete coverage report for the repository in .\coverage folder.
 - make cover-log: Saves overall coverage ratio into cover_log.csv file.

Dash
 Runs Python dash application.
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
Does complete coverage for the repository. The report can be found in the coverage folder placed in the repository 
folder. This folder is excluded from repository sync in gitignore.

### cover-save
@SAVES OVERALL COVERAGE RATIO INTO reports/cover_log.csv FILE

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


<a name="low-level-tools"></a>
# Low-Level Tools
[ToC](#table-of-content)  

A set of tools for working with the repository on the low level was created to allow different setups and monitoring. 

<a name="environment-variables"></a>
## Environment Variables
[ToC](#table-of-content)  

Environmental variables control the behaviour of the repository. The class for doing that is placed in 
*/src/utils/envs.py*. Its primary function is to set and get the variables that control the behaviour of the repository.
The methods are the following:  
- `set_config(value: str)`. Sets the name of the environment configuration file to be used.
- `get_config() -> str`. Gets the name of the environment configuration file to be used. If none was set before, 
the default configuration file name is used. The default name is stored in the variable `DEFAULT_CONFIG`, 
which can be found in the file */src/constants/global_constants.py.
- `set_logger(value: str)`. Sets the name of the logger configuration file to be used.
- `get_logger(value: str) -> str`. Gets the name of the logger file to be used. If none was set before, 
the default logger file name is used. The default name is stored in the variable `DEFAULT_LOGGER`, 
which can be found in the file /src/constants/global_constants.py.
- `set_running_unit_tests(value: str = “True”)`. Sets the indicator if in running unit tests mode.
- `get_running_unit_tests() -> str`. Gets the indicator if in running test mode. If none was set before, 
the default value is used. The default name is stored in the variable `DEFAULT_RUNNING_UNIT_TESTS`, 
which can be found in the file */src/constants/global_constants.py*.

For more information, please see the class itself and the 
*notebook /notebooks/documentation/low_level_tools_documentation.py* or 
*/docs/ low_level_tools_documentation.html*.


<a name="environmental-configuration"></a>
## Environmental Configuration
[ToC](#table-of-content)  

During the development, different stages of the project need different set up; for example development, testing , 
and production. Environment configurations allow different setups for different tasks. At this point, 
the following can be configured:
- Paths for storing the data.
- Parameters for parametric notebook execution.
- Configuration for dash.  
The class for reading the configurations is stored in the file */src/utils/config.py*. Singleton pattern was used 
- for the implementation. To get the configuration named tuple, use `config().get()` method. 


For more information, please see the class itself and the 
*notebook /notebooks/documentation/low_level_tools_documentation.py* or 
*/docs/ low_level_tools_documentation.html*.

<a name="timer"></a>
## Timer
[ToC](#table-of-content)  

The timer is a class for measuring execution time. It is placed in /src/utils/timer.py. The logic of the timer is based 
on start, meantimes, and the end. First of all, the timer has to be started. Next, a series of meantimes can be set. 
Each meantime represents the time from the start/previous meantime. Finally, the timer has to be ended. 
The collected data are stored and can be returned. The main methods are the following:  
- `set_results_printing(self, print_results: bool)`. This method sets the option for printing during the execution. 
The default value is True.
- `start()`. Starts the timer.
- `get_meantime(self, label: Union[str, None] = None) -> Tuple[float, float]`. Gets the meantime as duration in seconds 
and minutes. The label of the interval is optional and can be used to put a description of the process.
- `set_meantime(self, label: Union[str, None] = None) -> None`. Sets the meantime and stores it into internal variable.
- `get_end(self, label: Union[str, None]) -> Tuple[float, float, float, float]`. Ends the timer by setting the last 
meantime and returns the last meantime in seconds and minutes and the over duration in seconds and minutes.
- `end(self, label: Union[str, None] = None) -> None`. Ends the timer by setting the last meantime and stores the values 
into internal variable.
- `get_start_time(self) -> float`. Returns the start of the measurement.
- `get_end_time(self) -> float`. Retunrs the end of the measurement.
- `get_data(self) -> Tuple[List[float], List[float], List[str]]`. Returns the collected data.

For more information, please see the class itself and the 
*notebook /notebooks/documentation/low_level_tools_documentation.py* or 
*/docs/ low_level_tools_documentation.html*.

<a name="logger"></a>
## Logger
[ToC](#table-of-content)  

A logger was created for code monitoring. The standard logging library is used as a base. The class Logger uses a singleton pattern. There are three pre-defined logger configurations. Please see the Environmental variables section to see how to set them up. The configuration files are stored in /configurations.
- *logger_file.conf*. Logged information is saved only into a file.
- *logger_file_console.conf*. Logged information is saved into the file and, at the same time, printed into to console.
- *logger_file_console_dash.conf*. There is a separate logger configuration for the dash.
The methods are the following:
- `debug(self, message: str) -> None`.
- `info(self, message: str) -> None`.
- `warning(self, message: str) -> None`.
- `error(self, message: str) -> None`.
- `critical(self, message: str) -> None`.
In addition, the logger uses the timer to monitor the time complexity of the tasks. For more information, please see the Timer section; the methods are named accordingly.
- `start_timer(self, process_name: str) -> None`.
- `set_meantime(self, message: str) -> None`.
- `end_timer(self) -> None`.

For more information, please see the class itself and the 
*notebook /notebooks/documentation/low_level_tools_documentation.py* or 
*/docs/ low_level_tools_documentation.html*.

<a name="jupyter-notebooks"></a>
# Jupyter Notebooks
[ToC](#table-of-content)  

Jupyter notebooks are the center point of the project. They represent the workflow of the analysis, 
documentation of the process, simulations, they contain results. Their main use is as a client-server pattern. 
There is not important code stored in the notebooks there. The main code is outside in a separate codebase 
(client) and the notebooks are only using this code (server) to produce the outcome.

<a name="jupytext-library"></a>
## Jupytext Library
[ToC](#table-of-content)  

The Jupyter notebook native format is *.ipynb, but they can be converted into different formats as *.html 
and *.pdf files. The native *.ipynb format is not suitable for usage within the version control system, 
because except the code itself, it contains a lot of metadata that makes it hard to follow the changes. 
That is the reason for using [Jupytext](https://jupytext.readthedocs.io/en/latest/) library. 
This library allows storing the notebooks as plain *.py files, so that the tracking of changes is much easier.

To use Jupytext in general, do the following:
- Open an existing Jupyter notebook notebook or add a new one.
- Go to Jupyter notebook settings Edit/Edit Notebook Metadata/.
- Update the metadata to the following format.
    ```
    {
        "jupytext": {"formats": "ipynb,py:light"},
        "kernelspec": {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python"
    },
    ```
- Two parallel files are then available; *.ipynb, *.py. 

Within the repository, in the folder for Jupyter notebooks (notebooks), 
the *.ipynb are excluded from synchronisation with the server. This is specified in .gitignore file.

<a name="jupyter-notebooks-folders"></a>
## Jupyter Notebooks Folders
[ToC](#table-of-content)  

There is a short description in [*Files and Folders Structure*](#files-and-folders-structure) chapter there. It will be discussed in more 
details here to understand the flow and importance of respective folders.
- template
    - There are four template Jupyter notebooks for general usage in this folder there, all set up with
Jupytext library:  
        - template_notebook_empty.py. This is only empty notebook with Jupytext library settings, so it is stored in *.py pair file format.
        - template_notebook_repo.py. This is a raw notebook template which works within the repository and contains the basic functionality.
        - template_notebook_final.py. Notebook template ready for usage as a final notebook, with all the repository functionality and formatting.
        - template_parameterized_execution_notebook.py. Please see [*Parameterized Notebooks*](#parameterized-notebooks) 
    chapter for more information about running parameterized notebooks.
- documentation
     - This folder serves as a storage for Jupyter notebooks with the documentation of some specific 
functionality. Their exported *.html versions created when the functionality is finished can be found in the 
folder /docs, so that the code and usage can be used directly without necessity of running the notebook. 
These *.html files are updated every time the documentation Jupyter notebook is changed/updated.
- raw
    - Only preparation code is stored in this folder. 
- final
    - The most important code. The main delivery and documentation notebooks are kept in this folder.
- temporal
    - All *.py and *.ipynb are excluded for synchronisation in this folder. The folder is used for simulation and running notebooks without affecting the changes in repository itself.

<a name="jupyter-notebook-template"></a>
## Jupyter Notebook Template
[ToC](#table-of-content)  

As mentioned before, there is a notebook /notebooks/template/tempalate_notebook_final.py there in the repository. This notebook is used for all the final notebooks. The notebook uses the functionality mentioned in previous chapters, namely Envs, Config, and Logger.
The notebook is divided into four main parts.
- Notebook Description. The general description of the notebook can be found here.
- GENERAL SETTINGS. This section is meant to be used for the overall notebook setting. All the imports should be here, all the notebook constants, so that the code in the next section of the notebook can be run without any changes.
- ANALYSIS. The main code is supposed to be placed here. The code should be imported from files; no critical code should be placed in the notebook.
- Final Timestamp. The notebook is marked with timestamps at the beginning and at the end. A summary is placed in this section.

Final notebook versions:
- v1.1
  - Remove bugs while creating notebook for automatic run from the final notebook.

<a name="parameterized-notebooks"></a>
## Parameterized Notebooks
[ToC](#table-of-content)  

During the simulation stage, a need of running one notebook with different parameter values can arrise. 
In this case, a [papermill](https://github.com/nteract/papermill) library is very useful. 
There are two files created for this case:
- A script src/utils/param_notebook_executioner.py, which contains the class for running the parameterized notebook. The parameters can be specified in the config file or can be set up directly in the file. Running the scripts runs the following example notebook.
- An example notebook notebooks/template/template_parameterized_execution_notebook.py. This notebook is executed when the former script is run. The notebook also contains a guide on how to set up parameters in the notebook to be specified in the scripts.

<a name="ready-to-use-functionality"></a>
# Ready-to-Use Functionality
[ToC](#table-of-content)  

After having the primary codebase ready, the tools for the data science and machine learning tasks were developed on top. 

<a name="visualisations"></a>
## Visualisations
[ToC](#table-of-content)  

Visualisations are a necessary part of the data science analytical process. A list of basic plot types was created to 
serve for the analysis. A plotly library was chosen as a base. The reason is that the library is HTML based, 
and all the plots are interactive. That works nicely in the Jupyter notebook. What is more, these plots can be 
incorporated into Dash dashboarding tool, which can then serve as an initial tool for designing the UI and tuning the 
product's functionality. For more information, see the Dash section.  

To see the set of possible plots and their setting, see the notebook 
*/notebooks/documentation/visualisation_documentation.py* or its HTML version in */docs.* 
The code is placed in the folder */src/visualisations*. The usage is extensively described in the notebook, 
together with the proper input and variants of the plots.

<a name="data-frame-explorer"></a>
## Data Frame Explorer
[ToC](#table-of-content)  

Pandas data frame is a perfect structure for storing data and exploring data. For that, a class for fast initial data 
frame exploration for created. It is located in the file */src/data/df_explorer*. The class returns:
- Basic information about the data frame as shape, head of the data frame, and data frame description.
- Attribute types.
- NaN statistics.
- Attribute statistics in numbers and charts if the chart has a meaning.
- Data frames comparison for difference and identity testing.  

The functionality is documented in the notebook */notebooks/documentation/data_frame_explorer_documentation.py* or its 
HTML variant in */docs/data_frame_explorer_documentation.html*.

<a name="sample-data-set"></a>
## Sample Data Set
[ToC](#table-of-content)  

Income Weather data set is an artificially created data set for testing purposes. The code is in the file 
*/src/data/income_weather_data_generator.py*. The data set consists of the following attributes:  
- DATE. Date of the observation.
- WEATHER. Categorical variable describing the weather during that day.
- TEMPERATURE. That day's temperature.
- RANDOM. The random part of the regression model is created on the top of the data.
- OUTPUT. Regression output for the data.

The data simulates the regression task. The main method generate() returns the following:  
- df_data. Data frame with the basic data.
- df_data_transformed. Data frame with transformed categorical variables (DATE and WEATHER).
- X_multi, Y_multi. More complex, multidimensional regression data.

The functionality is documented in the notebook */notebooks/documentation/ income_weather_data_generator_documentation.py* 
or its HTML variant in */docs/ income_weather_data_generator_documentation.html*.

<a name="time-one-hot-encoding"></a>
## Time One Hot Encoding
[ToC](#table-of-content)  

Seasonality in time series is a phenomenon that has to be properly studied. Depending on the task, it can be in range 
from minutes, over days to months. Using regression, categorical variables reflecting these seasons have to be 
introduced. For that, the time one hot encoding class was created. The code can be found in 
*/src/transformation/datetime_one_hot_transformer.py*. The class allows adding the following time intervals:
- Any amount of minutes.
- Hours.
- Days of week.
- Weekends.
- Months.
- Years.
The functionality is documented in the notebook */notebooks/documentation/ datetime_one_hot_encoder_documentation.py*
or its HTML variant in */docs/ datetime_one_hot_encoder_documentation.html*.

<a name="dash"></a>
# Dash
[ToC](#table-of-content)  

[Dash](https://dash.plotly.com/) is a framework for building applications in Python and other languages such as R and 
Julia. It is written on the top of [Plotly](https://plotly.com/dash/). As was written in the Visualisation chapter, 
Plotly charts are the essential visualisation tools for data analysis. The benefit is that Dash allows fast and 
effective prototyping of applications on the analysis done. 

The repository contains code for a template HTML dashboarding tool. The code is in the folder /src/apps. A short video 
documenting the functionality is in /docs/dash.mp4.
