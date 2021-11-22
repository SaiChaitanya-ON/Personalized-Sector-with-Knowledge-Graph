import argparse
import logging
import math
import os
import pickle

import pyspark.sql.functions as F
import pyspark.sql.types as T
import smart_open

from onai.ml.peers.document_analyzer import Preprocess
from onai.ml.spark import get_spark
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


def make_models(words_df: dict, num_docs: int):
    words_idf = {k: math.log(num_docs / (v + 1)) for k, v in words_df.items()}
    return words_idf


def unique_token(token_list: list) -> list:
    return list(set(token_list))


def main():
    setup_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=False,
        default="s3://one-lake-prod/business/company_data_denormalized",
        help="parquet input to file",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="s3://oaknorth-ml-dev-eu-west-1/delan/data/idf/",
        help="output to file",
    )
    parser.add_argument(
        "--save",
        default=False,
        help="do we want to save the model ?",
        dest="save",
        action="store_true",
    )
    parser.add_argument(
        "--load",
        default=False,
        help="should older data be loaded for testing ?",
        dest="load",
        action="store_true",
    )
    parser.add_argument("--min_df", default=0, help="min document frequency", type=int)
    args = parser.parse_args()

    bow_col = "bow_desc"
    spark = get_spark()
    preprocessor = Preprocess()
    preprocess_udf = F.udf(
        preprocessor.document_processor, T.MapType(T.StringType(), T.IntegerType())
    )
    token_udf = F.udf(unique_token, T.ArrayType(T.StringType()))
    logger.info(args)

    if not args.load:
        spark_df = spark.read.load(args.input)
        raw_df = (
            spark_df.select("business_description")
            .filter(
                F.col("business_description").isNotNull()
                & (F.length("business_description") > 10)
            )
            .withColumn(bow_col, token_udf(preprocess_udf("business_description")))
            .repartition(200)
        ).cache()
        (raw_df.write.parquet(f"{args.output}/df_all_companies", mode="overwrite"))

    else:
        logger.info("loading precalculated data..")
        raw_df = spark.read.load(f"{args.output}/df_all_companies")

    logger.info("calculating idf....")
    num_doc = raw_df.count()
    doc_feq_df = (
        raw_df.withColumn("token", F.explode(bow_col))
        .groupBy("token")
        .agg(F.count("*").alias("doc_freq"))
    )
    words_df = {
        row.token: row.doc_freq
        for row in doc_feq_df.collect()
        if row.doc_freq > args.min_df
    }

    idf_model = make_models(words_df, num_doc)

    if args.save:
        logger.info("Saving the IDF weights..")
        with smart_open.open(os.path.join(args.output, "idf_model.pkl"), "wb") as f:
            logger.info(os.path.join(args.output, "idf_model.pkl"))
            pickle.dump(idf_model, f)


if __name__ == "__main__":
    main()
