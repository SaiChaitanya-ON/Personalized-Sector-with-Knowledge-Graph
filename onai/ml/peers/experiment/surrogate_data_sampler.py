import argparse
import functools
import json
import logging
import multiprocessing as mp
import os
import random
import time
from datetime import date, datetime
from typing import Dict

import pyspark.sql.functions as F
import pyspark.sql.types as T
import tabulate
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col as c
from smart_open import open
from urllib3.exceptions import ReadTimeoutError

from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.types import CompanyDetail
from onai.ml.spark import get_spark
from onai.ml.spark_cache import get_es_cs
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def get_candidate(entity_id, args, start_year=2008, end_year=2020):
    cs: ESCandidateSuggestion = get_es_cs(args)
    # we need to get full details from ml es, as the original one_lake_prod dump does not have
    # predicted_industries field
    try:
        es_res = cs.search_by_entity_ids([entity_id])
    except ReadTimeoutError:
        logger.warning("Time out on %s. Giving up.", entity_id)
        return []
    if len(es_res) < 1:
        logger.warning("Cannot find %s in ML ES, skipping", entity_id)
        return []
    es_res = es_res[0]
    cd = CompanyDetail(
        name=es_res["name"],
        description=es_res["business_description"],
        region=es_res["region"],
        predicted_industries=es_res.get("predicted_industries", []),
    )

    pss = cs.suggest_candidates(
        cd,
        start_year=start_year,
        end_year=end_year,
        size=args.suggestion_size,
        get_peer_financial=False,
    )

    # choose sample_size out of suggestion_size
    sampled_pss = random.sample(pss, min(len(pss), args.sample_size))

    return [ps.detail for ps in sampled_pss]


COMPANY_DETAIL_SPARK_TYPE = T.ArrayType(
    T.StructType(
        [
            T.StructField("region", T.StringType()),
            T.StructField("description", T.StringType()),
            T.StructField("entity_id", T.StringType(), False),
            T.StructField("name", T.StringType(), False),
            T.StructField("country", T.StringType()),
        ]
    )
)


def dfZipWithIndex(spark, df, offset=1, colName="rowId"):
    """
        Enumerates dataframe rows is native order, like rdd.ZipWithIndex(), but on a dataframe
        and preserves a schema

        :param df: source dataframe
        :param offset: adjustment to zipWithIndex()'s index
        :param colName: name of the index column
    """

    new_schema = T.StructType(
        [T.StructField(colName, T.LongType(), True)]  # new added field in front
        + df.schema.fields  # previous schema
    )

    zipped_rdd = df.rdd.zipWithIndex()

    new_rdd = zipped_rdd.map(lambda args: ([args[1] + offset] + list(args[0])))

    return spark.createDataFrame(new_rdd, new_schema)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", help="How many base queries we sample", type=int, default=100
    )
    parser.add_argument(
        "--company_data",
        default="s3a://one-lake-prod/business/company_data_denormalized",
    )
    parser.add_argument(
        "--financial_data", default="s3a://one-lake-prod/business/financial_data"
    )
    parser.add_argument("--dime_map", default="s3a://one-lake-prod/business/dime_map")
    parser.add_argument(
        "--sample_size",
        help="How many companies per query to be suggested by the current Candidate Suggester"
        "should we include",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--negative_sample_size",
        help="How many random companies per query to be suggested by the candidate suggester",
        type=int,
        default=32,
    )
    parser.add_argument("--suggestion_size", type=int, default=100)
    parser.add_argument("-p", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=int(time.time()))

    parser.add_argument("--es_host", default="berry-es-test.ml.onai.cloud")
    parser.add_argument("--es_port", default=80)
    parser.add_argument(
        "--output",
        required=True,
        default="s3a://oaknorth-ml-dev-eu-west-1/iat/data/peers/surrogate_sample",
    )
    parser.add_argument(
        "--region_model",
        default="s3://oaknorth-staging-non-confidential-ml-artefacts/region/v0.0.1/",
    )
    parser.add_argument("--last_filing_year_start", default=2015, type=int)
    parser.add_argument("--last_filing_year_end", default=2020, type=int)
    parser.add_argument("--entity_id")

    args = parser.parse_args()

    spark = get_spark(memory="4g", n_threads=args.p)
    seed = args.seed
    random.seed(seed)

    seed += 1
    # 0) get all iso3 code that are mapped to Europe or UnitedStatesAndCanada

    # 1) sample a list of base companies that are _good_
    country_filter_q = get_country_filter(args.region_model)

    company_data_denormalized = spark.read.load(args.company_data)
    company_data_denormalized = company_data_denormalized.select(
        [c(x).alias(x.lower()) for x in company_data_denormalized.columns]
    )

    if args.entity_id:
        sampled_company_data = company_data_denormalized.filter(
            company_data_denormalized.entity_id == args.entity_id
        )
    else:
        company_data_denormalized = (
            company_data_denormalized.filter(country_filter_q)
            .withColumn("len_business_description", F.length("business_description"))
            .filter(
                "len_business_description is not null and len_business_description > 10"
            )
            .filter(
                F.year("last_filing_date").between(
                    args.last_filing_year_start, args.last_filing_year_end - 1
                )
            )
        )

        total_sz = company_data_denormalized.count()
        logger.info("Total size: %d", total_sz)
        bins = [0, 150, float("inf")]
        sample_ratios = {0: 0.2, 1: 0.8}
        company_data_denormalized: DataFrame = Bucketizer(
            splits=bins, inputCol="len_business_description", outputCol="bucket"
        ).transform(company_data_denormalized)

        company_data_denormalized = (
            company_data_denormalized.withColumnRenamed("bucket", "bucket_double")
            .withColumn("bucket", c("bucket_double").cast(T.IntegerType()))
            .drop("bucket_double")
        )

        hists = company_data_denormalized.groupBy("bucket").count().collect()
        counts = [0.0] * (len(bins) - 1)

        for h in hists:
            counts[h["bucket"]] = h["count"]

        tab = []
        for (start_bin, end_bin), count in zip(zip(bins, bins[1:]), counts):
            tab.append(
                [f"{start_bin} - {end_bin}", count, f"{count / total_sz * 100:.2f}%"]
            )
        logger.info(
            "Histogram: \n%s",
            tabulate.tabulate(tab, headers=["bins", "counts", "counts(%)"]),
        )

        # 2) work out the correct sampling ratios per bucket
        readjusted_sampling_ratios = {
            bucket: min(1, args.n * ratio / counts[bucket])
            for bucket, ratio in sample_ratios.items()
        }
        logger.info("Sampling Ratio: %s", sample_ratios)
        logger.info("Readjusted Sampling Ratio: %s", readjusted_sampling_ratios)

        sampled_company_data = company_data_denormalized.sampleBy(
            "bucket", readjusted_sampling_ratios, seed=int(seed)
        ).drop("bucket", "len_business_description")
        seed += 1

    get_candidate_udf = F.udf(
        functools.partial(get_candidate, args=args), COMPANY_DETAIL_SPARK_TYPE
    )
    company_data_denormalized_idxed = dfZipWithIndex(
        spark, company_data_denormalized, offset=0
    )
    overall_size = company_data_denormalized_idxed.count()
    negative_samples = (
        sampled_company_data.select(
            F.explode(F.array_repeat(c("entity_id"), args.negative_sample_size)).alias(
                "entity_id"
            )
        )
        .withColumn("neg_sample_id", F.floor(F.rand(seed) * overall_size))
        .alias("sc")
        .join(
            company_data_denormalized_idxed.alias("cd"),
            c("cd.rowId") == c("sc.neg_sample_id"),
        )
        .groupBy("sc.entity_id")
        .agg(
            F.collect_list(
                F.struct(
                    c("cd.region").alias("region"),
                    c("cd.business_description").alias("description"),
                    c("cd.entity_id").alias("entity_id"),
                    c("cd.name").alias("name"),
                    c("cd.country_of_incorporation_iso").alias("country"),
                )
            ).alias("neg_samples")
        )
        .alias("sc")
    )

    outcome = (
        sampled_company_data.repartition(1000)
        .withColumn("pos_samples", get_candidate_udf("entity_id"))
        .join(negative_samples, c("sc.entity_id") == sampled_company_data.entity_id)
        .withColumn("samples", F.concat("pos_samples", "neg_samples"))
        .drop("pos_samples", "neg_samples")
        .drop(c("sc.entity_id"))
    )

    outcome.write.parquet(args.output)


def get_country_filter(
    region_model,
    region_filter=("Europe", "United States and Canada"),
    country_iso3_filter=("AUS",),
):
    country_iso3_filter = set(f"'{k}'" for k in country_iso3_filter)
    with open(os.path.join(region_model, "model.json"), "r") as fin:
        region_by_iso3: Dict = json.load(fin)
    country_iso3_filter |= set(
        f"'{k}'" for k, v in region_by_iso3.items() if v in region_filter
    )
    logger.info("Countries to be considered %s", country_iso3_filter)
    country_filter_q = (
        f"country_of_incorporation_iso in ({','.join(country_iso3_filter)})"
    )
    return country_filter_q


if __name__ == "__main__":
    main()
