ARG SPARK_BASE_IMG="715524042254.dkr.ecr.eu-west-1.amazonaws.com/data-infra-dev/spark-py:2.4.5-hadoop-2.9.2-scala-2.12-13eeddb"
ARG BASE_IMG="debian:buster-slim"
ARG PYTHON_VERSION=3.7.4

FROM 411980747634.dkr.ecr.eu-west-1.amazonaws.com/keyring:0.0.3 as keyring
FROM ${SPARK_BASE_IMG} AS ml_spark
FROM ${BASE_IMG} as ml_dep_base

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get -y install --reinstall build-essential && \
    apt-get -y install jq curl libbz2-dev libssl-dev libffi-dev zlib1g-dev sed wget gcc git gpg nano liblzma-dev \
    libkrb5-dev poppler-utils software-properties-common libsqlite3-dev libdb-dev \
    tesseract-ocr libtesseract-dev libleptonica-dev pkg-config parallel rsync && \
    wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && \
    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ && \
    apt-get update && apt-get -y install adoptopenjdk-8-hotspot && \
    apt-get clean -y

ARG PYTHON_VERSION

# we don't want to write bytecode in order to avoid syncing different versions of bytecode to the container with devspace sync, during development
ENV PYTHONDONTWRITEBYTECODE=1
COPY scripts/install-python.sh .
RUN ./install-python.sh ${PYTHON_VERSION}

ENV PIP_NO_CACHE_DIR=1

RUN pip install "poetry==1.1.0"

COPY --from=keyring /root/.ssh/id_rsa /root/.ssh/id_rsa
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

RUN pip install awscli

WORKDIR /acornML/

ADD Makefile .
ADD pypi_utils/ pypi_utils
ADD pyproject.toml ./pyproject.toml
ADD poetry.lock ./poetry.lock

FROM ml_dep_base as ml_base

RUN make pypi-login && \
    poetry install --no-root && \
# change this once https://github.com/python-poetry/poetry/pull/1822 is deployed
# or our own pypi private repo https://acornlab.atlassian.net/browse/INFRA-486 is released:
#    poetry run pip install "git+ssh://git@github.com/OakNorthAI/data-lake.git@4e19654#egg=onai-datalake&subdirectory=datalake" && \
    rm -rf /root/.cache/pypoetry/artifacts && \
    rm -rf /root/.cache/pypoetry/cache

FROM ml_dep_base as ml_sm_base

ENV SPARK_HOME /opt/spark
COPY --from=ml_spark ${SPARK_HOME} ${SPARK_HOME}

ENV PYTHONPATH $PYTHONPATH:$SPARK_HOME/python/lib/pyspark.zip:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip

RUN pip install sagemaker-training
ADD scripts/sm_deps.sh ./sm_deps.sh

CMD "bash"
