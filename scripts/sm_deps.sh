#!/usr/bin/env bash
set -euo pipefail

make pypi-login
poetry config virtualenvs.in-project true
poetry install --no-root

pushd .venv
find . -type f | parallel -j80 -X rsync -R -Ha ./{} ${POETRY_VENV_PATH}/
