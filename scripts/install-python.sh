#!/usr/bin/env bash
set -exo pipefail

wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xf Python-$PYTHON_VERSION.tgz

pushd Python-$PYTHON_VERSION
./configure --disable-test-suit
make -j $(nproc) && make install

popd
rm -r Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz

# Register the version in alternatives
update-alternatives --install /usr/bin/python python /usr/local/bin/python3 1

# Set python 3 as the default python
update-alternatives --set python /usr/local/bin/python3

curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --force-reinstall
rm get-pip.py
