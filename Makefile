SHELL := /bin/bash

PYTEST_ARG ?= onai/ml/tests

ifneq (,$(wildcard ./garden.env))
    include garden.env
endif

ecr-login:
	$$(aws ecr get-login --no-include-email --region eu-west-1)
	$$(aws ecr get-login --no-include-email --region eu-west-1 --registry-ids 715524042254)

init-garden:
	setup_env init-garden

run-dev: init-garden
	garden deploy ml-dev,ml

dev-test:
	garden exec ml-dev 'poetry run pytest $(PYTEST_ARG)'

dev-lint:
	garden exec ml-dev 'poetry run ./scripts/linter.sh'

run-bash:
	garden exec ml-dev bash

kill-dev:
	kubectl delete namespace ${ns} --force --grace-period=0 || true
	kubectl delete namespace ${ns}--metadata --force --grace-period=0 || true
	kubectl delete pv ${ns}-user-efs --force --grace-period=0 || true

init-sm-image:
	garden run task initialise-efs-venv -l=4

setup:
	pip install poetry==1.1.0 awscli
	git submodule update --init --recursive

pypi-login:
	source pypi_utils/assume_codeartifact.sh && pypi_utils/pypi-login.sh

publish:
	source pypi_utils/assume_codeartifact.sh && pypi_utils/upload.sh --allow-unpublished-changes
