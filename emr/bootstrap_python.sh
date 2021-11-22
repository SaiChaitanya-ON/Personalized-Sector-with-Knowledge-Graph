#!/bin/bash

CONFIG_FOLDER="onai-ml-dev-eu-west-1/configs/THIS_IS_A_LONG_MAKER_NAME"

sudo yum install -y  gcc gcc-c++ make git patch openssl-devel zlib-devel readline-devel sqlite-devel bzip2-devel zlib  libffi-devel

wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O /home/hadoop/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /mnt/conda


echo "export SPARK_HOME=/usr/lib/spark" >> ~/.bashrc
echo "export PYSPARK_PYTHON=/usr/bin/python" >> ~/.bashrc
echo "export AWS_DEFAULT_PROFILE=default" >> ~/.bashrc
echo "export PATH=/mnt/conda/bin:$PATH" >> ~/.bashrc

echo "export HADOOP_CONF_DIR=/etc/hadoop/conf" >> ~/.bashrc
echo "export SPARK_MASTER=yarn" >> ~/.bashrc
echo "export YARN_CONF_DIR=/etc/hadoop/conf" >> ~/.bashrc
echo "export PATH=/mnt/conda/bin:$PATH" >> ~/.bashrc
echo "export NLTK_DATA=/mnt/conda/ntlk_data" >> ~/.bashrc
source ~/.bashrc

conda config --set always_yes yes --set changeps1 no

conda install conda=4.5.11

conda config -f --add channels conda-forge
conda config -f --add channels defaults

conda install -y python=3.7.4

sudo alternatives --install /usr/bin/python python /mnt/conda/bin/python 0
sudo alternatives --set python /mnt/conda/bin/python

conda install -y numpy=1.16.4 scipy=1.3.0 scikit-learn=0.21.2 faiss-cpu pytorch=1.2.0 pymc3 -c pytorch -c conda-forge

pip install "poetry>=1.0.0"
poetry config virtualenvs.create false

aws s3 cp s3://${CONFIG_FOLDER}/pyproject.toml .
aws s3 cp s3://${CONFIG_FOLDER}/poetry.lock .

poetry install --no-root
python -m spacy download en_core_web_lg

sudo chmod 777 /home/
