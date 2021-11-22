ARG SPARK_BASE_IMG="715524042254.dkr.ecr.eu-west-1.amazonaws.com/data-infra-dev/spark-py:2.4.5-hadoop-2.9.2-scala-2.12-13eeddb"
ARG ML_BASE_IMG
FROM ${SPARK_BASE_IMG} AS ml_spark
FROM ${ML_BASE_IMG}


ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH
COPY --from=ml_spark ${SPARK_HOME} ${SPARK_HOME}

ADD emr/ ./emr/
ADD scripts/ ./scripts/
ADD setup.cfg ./setup.cfg
ADD onai ./onai
ADD notebooks ./notebooks

EXPOSE 8888
EXPOSE 4040-4060

ENV PYTHONPATH $PYTHONPATH:/acornML/:$SPARK_HOME/python/lib/pyspark.zip:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip

CMD poetry run ./scripts/start-dev.sh
