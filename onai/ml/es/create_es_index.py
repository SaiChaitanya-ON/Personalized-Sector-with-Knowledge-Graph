import pyspark.sql.functions as F
from onai_datalake.utils.spark_helper import get_spark

spark = get_spark("ES")

pred_ind_path = "s3a://one-lake-prod/business/company_data_denormalized"
predicted_industries = spark.read.load(pred_ind_path).repartition(1024)
predicted_industries = predicted_industries.select(
    [F.col(x).alias(x.lower()) for x in predicted_industries.columns]
)
(
    predicted_industries.write.format("org.elasticsearch.spark.sql")
    .option("es.nodes.wan.only", "true")
    .option("es.net.ssl", False)
    .option("es.nodes", "host.docker.internal")
    .option("es.port", "9200")
    .option("es.mapping.id", "entity_id")
    .mode("Append")
    .save("company/_doc")
)
