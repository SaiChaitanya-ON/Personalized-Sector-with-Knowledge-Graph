#!/bin/sh

isort --check -df -rc onai
ISORT_RESULT="$?"

black --diff --check onai
BLACK_RESULT="$?"

flake8 onai
FLAKE8_RESULT="$?"

[ "${FLAKE8_RESULT}${ISORT_RESULT}${BLACK_RESULT}" = "000" ] && \
    echo "\nAll checks completed successfully."
