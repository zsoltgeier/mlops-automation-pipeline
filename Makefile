PYTHON ?= python

.PHONY: quality test prepare

quality:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml
	$(PYTHON) -m coverage report --include='src/*' --fail-under=95
	$(PYTHON) -m coverage report --include='tests/*' --fail-under=95
	$(PYTHON) -m flake8 src tests
	$(PYTHON) -m pyright
	$(PYTHON) -m pyright -p pyright-standard.json
	$(PYTHON) -m pip check

test:
	$(PYTHON) -m pytest --cov --cov-report=term-missing

prepare:
	$(PYTHON) -m dvc repro
