#!/bin/sh

set -e

isort -rc onai
black onai
flake8 onai
