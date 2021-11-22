#!/bin/sh

jupyter notebook --port=8888 --ip='0.0.0.0' --notebook-dir='notebooks' --NotebookApp.token='' --allow-root
