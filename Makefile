PYTHON ?= $(if $(wildcard .conda/envs/mlops/bin/python),.conda/envs/mlops/bin/python,python)
PYRIGHT_PYTHON := $(shell $(PYTHON) -c 'import sys; print(sys.executable)')

.PHONY: quality test coverage lint typecheck prepare hooks docker-build system-smoke

quality:
	$(PYTHON) -m flake8 src tests dags
	$(PYTHON) -m pyright --pythonpath "$(PYRIGHT_PYTHON)" --warnings
	$(PYTHON) -m pyright -p pyright-standard.json --pythonpath "$(PYRIGHT_PYTHON)" --warnings
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml --junitxml=runtime/test-results.xml
	$(PYTHON) -m coverage report --include='src/*' --fail-under=95
	$(PYTHON) -m coverage report --include='tests/*' --fail-under=95
	$(PYTHON) -m pip check

test:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml --junitxml=runtime/test-results.xml

coverage:
	$(PYTHON) -m coverage report --include='src/*' --fail-under=95
	$(PYTHON) -m coverage report --include='tests/*' --fail-under=95

lint:
	$(PYTHON) -m flake8 src tests dags

typecheck:
	$(PYTHON) -m pyright --pythonpath "$(PYRIGHT_PYTHON)" --warnings
	$(PYTHON) -m pyright -p pyright-standard.json --pythonpath "$(PYRIGHT_PYTHON)" --warnings

prepare:
	PATH="$(dir $(PYRIGHT_PYTHON)):$$PATH" $(PYTHON) -m dvc repro prepare

hooks:
	PRE_COMMIT_HOME="$(CURDIR)/.cache/pre-commit" $(PYTHON) -m pre_commit install

docker-build:
	timeout --kill-after=5s 15s docker info > /dev/null
	docker compose config --quiet
	timeout --kill-after=5s 600s docker compose build

system-smoke:
	bash scripts/compose-smoke.sh
