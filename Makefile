########################################################################################################################
# BEWARE OF TABS IN FRONT - PYCHARM CHANGE IT TO FOUR SPACES ###########################################################
# NEED TO REPLACE IT - FOR EXAMPLE NOTEPAD #############################################################################
########################################################################################################################

.PHONY: help

include make_config.mk

FILE_CODE = src/$(FILE_FOLDER)/$(FILE_NAME).py
FILE_TEST = tests/tests_$(FILE_FOLDER)/test_$(FILE_NAME).py
FILE_DOCTEST = tests/tests_$(FILE_FOLDER)/test_$(FILE_NAME).txt

# UTILS ----------------------------------------------------------------------------------------------------------------

.DEFAULT: help
help: clear-console
	@python ./src/utils/make_print_documentation.py help

hello: clear-console
	@python ./src/utils/make_print_documentation.py hello

clear-console:
	@echo ""
	@clear

# ENVIRONMENT ----------------------------------------------------------------------------------------------------------

create-venv: clear-console
	@python ./src/utils/make_print_documentation.py create-venv
	pip install virtualenv==20.19.0
	test -d .venv || virtualenv .venv
	( \
       source .venv/Scripts/activate;\
       pip install -r requirements.txt;\
	)

freeze-out: clear-console
	@python ./src/utils/make_print_documentation.py freeze-out
	( \
       source .venv/Scripts/activate;\
       pip freeze > requirements.txt\
	)

freeze-in: clear-console
	@python ./src/utils/make_print_documentation.py freeze-in
	@pip freeze > requirements.txt;\

# SOURCE CODE QUALITY --------------------------------------------------------------------------------------------------

mypy-no-clear:
	@python ./src/utils/make_print_documentation.py mypy-no-clear
	@mypy --strict src tests --config-file mypy.ini

mypy: clear-console mypy-no-clear

lint-no-clear:
	@python ./src/utils/make_print_documentation.py lint-no-clear
	@pylint src tests --rcfile .pylintrc

lint: clear-console lint-no-clear

lint-dup-no-clear:
	@python ./src/utils/make_print_documentation.py lint-dup-no-clear
	@pylint src pipelines tests --rcfile .pylintrc_dup

lint-dup: clear-console lint-dup-no-clear

test-no-clear:
	@python ./src/utils/make_print_documentation.py test-no-clear
	@python -m pytest

test: clear-console test-no-clear

all: clear-console mypy-no-clear lint-no-clear test-no-clear

# ONE FILE QUALITY -----------------------------------------------------------------------------------------------------

# error can be supressed by removing @ and put ;\ at the end of the line
mypy-f: clear-console
	@python ./src/utils/make_print_documentation.py mypy-f
	@echo "  - File Name: $(FILE_CODE)"
	@echo ""
	@mypy --strict $(FILE_CODE) --config-file mypy.ini
	@python ./src/utils/make_print_documentation.py mypy-f
	@echo "  - File Name: $(FILE_TEST)"
	@echo ""
	@mypy --strict $(FILE_TEST) --config-file mypy.ini

# error can be supressed by removing @ and put ;\ at the end of the line
lint-f: clear-console
	@python ./src/utils/make_print_documentation.py mypy-f;\
	echo "  - File Name: $(FILE_CODE)";\
	echo "";\
	pylint $(FILE_CODE) --rcfile .pylintrc;\
	python ./src/utils/make_print_documentation.py mypy-f;\
	echo "  - File Name: $(FILE_TEST)";\
	echo "";\
	pylint $(FILE_TEST) --rcfile .pylintrc;\
	echo "";\
	echo "";\

test-f: clear-console
	@python ./src/utils/make_print_documentation.py test-f
	@echo "  - File Folder: $(FILE_FOLDER)"
	@echo "  - File Name: $(FILE_NAME)"
ifeq ($(shell test -e $(FILE_TEST) && echo -n yes),yes)
	@python -m pytest $(FILE_TEST)
else
	@echo "ERROR: FOLLOWING FILE DOES NOT EXIST: $(FILE_TEST)"
	@echo ""
endif
ifeq ($(shell test -e $(FILE_DOCTEST) && echo -n yes),yes)
	@python -m pytest $(FILE_DOCTEST)
else
	@echo ""
	@echo "ERROR: FOLLOWING FILE DOES NOT EXIST: $(FILE_DOCTEST)"
endif

all-f: mypy-f lint-f test-f

# NOTEBOOKS CODE QUALITY -----------------------------------------------------------------------------------------------

mypy-ntb-no-clear:
	@python ./src/utils/make_print_documentation.py mypy-ntb-no-clear
	@mypy --strict notebooks/documentation notebooks/final --config-file mypy.ini

mypy-ntb: clear-console mypy-ntb-no-clear

lint-ntb-no-clear:
	@python ./src/utils/make_print_documentation.py lint-ntb-no-clear
	@pylint notebooks/documentation notebooks/final --rcfile .pylintrc_ntb

lint-ntb: clear-console lint-ntb-no-clear

all-ntb: clear-console mypy-ntb-no-clear lint-ntb-no-clear

# COVERAGE -------------------------------------------------------------------------------------------------------------

# if you need to ignore a file, add to @pytes line to the end --ignore=tests/tests_xxx/test_yyy.py
cover-base:
	@python ./src/utils/make_print_documentation.py cover-base
	@pytest --cov-report html:coverage --cov=src tests/

cover: clear-console cover-base

cover-save:
	@python ./src/utils/make_print_documentation.py cover-save
	@python ./src/utils/cover_logger.py

cover-log: clear-console cover-base cover-save


# DASH -----------------------------------------------------------------------------------------------------------------

run-dash: clear-console
	@python ./src/utils/make_print_documentation.py run-dash
	@python index.py

build-run-dash-docker: clear-console
	@python ./src/utils/make_print_documentation.py build-run-dash-docker
	@docker build -t dash . && docker run --rm -p 8050:8050 dash

build-dash-docker: clear-console
	@python ./src/utils/make_print_documentation.py build-dash-docker
	@docker build -t dash .

run-dash-docker: clear-console
	@python ./src/utils/make_print_documentation.py run-dash-docker
	@docker run --rm -p 8050:8050 dash

#-----------------------------------------------------------------------------------------------------------------------
# SAMPLE FUNCTIONALITY -------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Following make serves as an example of make command taking some input/parameter from the command line and using
# its value within the python script.
# usage - make print_text [header=text_without_blank_space]
print_section:
ifdef header
	@python ./src/utils/make_print_documentation.py $(header)
else
	@python ./src/utils/make_print_documentation.py
endif

hello_echo:
	@echo ""
	@echo "==================================================================================="
	@echo "=============================== Hail to you, hero!! ==============================="
	@echo "============= Congratulations to you running your first make command! ============="
	@echo "==================================================================================="

# Prints the respective hello section from README.md together with clear.
hello_from_readme:
	clear;\
	@python ./src/utils/make_print_documentation.py hello

# Runs simple python script with one parameter.
# Usage - make run_python_script [param=text_without_blank_space ]
run_python_script:
	@python ./src/utils/make_example_script.py
	@python ./src/utils/make_example_script.py print_this
	@python ./src/utils/make_example_script.py $(param)
ifdef param
	@python ./src/utils/make_example_script.py $(param)
else
	@python ./src/utils/make_example_script.py
endif

# Runs simple python script with two parameters.
# usage - make print_text [first=text_without_blank_space second=text_without_blank_space]
run_python_script_multi:
	@python ./src/utils/make_example_script_multi.py $(first) $(second)

dash:
	@python index.py

# https://stackoverflow.com/questions/5553352/how-do-i-check-if-file-exists-in-makefile-so-i-can-delete-it
# does not work
# file-exists:
# 	FILE make_config.mkd
# ifneq ("$(wildcard @(FILE))","")
# 	@echo "file exists"
# else
# 	@echo "file does not exist"
# endif


# working basic example - please change the file name to check

FILE=index.py
file-exists-basic:
ifeq ($(shell test -e $(FILE) && echo -n yes),yes)
	@echo "file exists"
else
	@echo "file does not exist"
endif