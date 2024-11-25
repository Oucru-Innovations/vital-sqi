.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build: ## install package with editable option
	pip install -e .

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test: ## remove test and coverage artifacts
	rm -rf .tox/
	rm -rf .pytest_cache
	rm -rf .coverage

lint: ## check style with flake8 using .flake8 config
	flake8 --config=.flake8 vital_sqi tests

lint-file: ## lint a specific file with flake8 using .flake8 config
	flake8 --config=.flake8 $(filename)

doc-style: ## convert documentation style to numpy style
	pyment -o numpydoc -w $(filename)

test: ## run tests with coverage report
	pytest --cov=vital_sqi --cov-config=.coveragerc --cov-report term tests/

BROWSER ?= firefox

cov: ## Run tests and show coverage report by file in the terminal
	pytest --cov=vital_sqi --cov-config=.coveragerc --browser=$(BROWSER) tests/ 
	coverage report -m

test-all: ## run tests on every Python version with tox
	tox

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python3 setup.py install
