import argparse
import logging
import multiprocessing as mp

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pandas import DataFrame
from pyspark.sql import Window
from pyspark.sql.functions import col as c

from onai.ml.spark import get_spark
from onai.ml.spark_cache import get_coa_tree, setup_logger


def ebitda_availability(mnemonics):
    setup_logger()
    mnemonics_set = set(mnemonics)
    tree = get_coa_tree()
    tree.reset_concrete_vals()
    for m in mnemonics_set:
        if m in tree:
            tree[m].concrete_val = True

    return tree.root_populatable()


COUNTRIES = ["DEU", "AUS", "USA", "GBR", "NLD"]

logger = logging.getLogger(__name__)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company_denorm",
        default="s3://one-lake-prod/business/company_data_denormalized",
    )
    parser.add_argument("--dime_map", default="s3://one-lake-prod/business/dime_map")
    parser.add_argument(
        "--financial_data", default="s3://one-lake-prod/business/financial_data"
    )
    parser.add_argument("--start_yr", type=int, default=2015)
    parser.add_argument("--end_yr", type=int, default=2020)
    parser.add_argument("-p", type=int, default=mp.cpu_count())

    args = parser.parse_args()
    spark = get_spark(memory="8g", n_threads=args.p)

    company_df = spark.read.load(args.company_denorm)
    company_df = company_df.select([c(x).alias(x.lower()) for x in company_df.columns])

    financial_df = spark.read.load(args.financial_data)
    dime_df = spark.read.csv(args.dime_map, header=True)

    financial_df = financial_df.filter(
        (financial_df.period_type == "ANNUAL")
        & (F.year("event_date").between(args.start_yr, args.end_yr))
    ).join(dime_df, dime_df.oneid == financial_df.data_item_id)
    # there might be duplicates, we need to make sure that there is only one-row after all
    # https://stackoverflow.com/questions/33878370/how-to-select-the-first-row-of-each-group
    # we cannot use drop_duplicates after order as the behaviour of first is not guaranteed.

    w = Window.partitionBy(
        [financial_df.entity_id, F.year(financial_df.event_date), dime_df.mnemonic]
    ).orderBy(
        F.struct(
            financial_df.prioritization, F.to_date(financial_df.reported_date)
        ).desc()
    )
    financial_df = (
        financial_df.withColumn("_rn", F.row_number().over(w))
        .filter("_rn == 1")
        .drop("_rn")
    )
    row_to_company_detail_udf = F.pandas_udf(
        ebitda_availability, T.BooleanType(), F.PandasUDFType.GROUPED_AGG
    )
    country_iso3_filter = set(f"'{k}'" for k in COUNTRIES)
    country_filter_q = (
        f"country_of_incorporation_iso in ({','.join(country_iso3_filter)})"
    )
    company_df = company_df.filter(country_filter_q)

    filtered_financial_df = (
        financial_df.alias("f")
        .join(company_df, company_df.entity_id == financial_df.entity_id)
        .select(
            [f"f.{col}" for col in financial_df.columns]
            + ["country_of_incorporation_iso"]
        )
    )
    #
    counts_df: DataFrame = filtered_financial_df.groupBy(
        company_df.country_of_incorporation_iso,
        filtered_financial_df.entity_id,
        F.year(filtered_financial_df.event_date),
    ).agg(row_to_company_detail_udf(dime_df.mnemonic).alias("has_ebitda")).withColumn(
        "has_ebitda_float", c("has_ebitda").cast(T.FloatType())
    ).groupBy(
        company_df.country_of_incorporation_iso, filtered_financial_df.entity_id
    ).agg(
        F.max("has_ebitda_float").alias("has_ebitda_float")
    ).groupBy(
        company_df.country_of_incorporation_iso
    ).sum(
        "has_ebitda_float"
    )

    counts_df.show()


if __name__ == "__main__":
    main()
