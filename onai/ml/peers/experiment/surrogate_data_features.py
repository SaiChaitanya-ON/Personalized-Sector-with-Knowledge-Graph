import argparse
import logging
import multiprocessing as mp

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from pyspark.sql.functions import col as c

from onai.ml.spark import get_spark
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)

_MNEMONICS = ("TOTAL_REVENUE", "EBIT", "EBITDA")


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate_sample_path", required=True)
    parser.add_argument(
        "--financial_data", default="s3a://one-lake-prod/business/financial_data"
    )
    parser.add_argument("--dime_map", default="s3a://one-lake-prod/business/dime_map")
    parser.add_argument(
        "--exchange_rate", default="s3a://one-lake-prod/business/exchange_rate"
    )
    parser.add_argument("--start_yr", type=int, default=2008)
    parser.add_argument("--end_yr", type=int, default=2020)
    parser.add_argument("-p", type=int, default=mp.cpu_count())
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spark = get_spark(memory="8g", n_threads=args.p)

    sample_df = spark.read.load(args.surrogate_sample_path)
    grouped_financial_df = cleaned_financial_from_dp(
        args.financial_data,
        args.dime_map,
        args.exchange_rate,
        args.start_yr,
        args.end_yr,
        spark,
    )
    # Normalisation done

    grouped_financial_df_1 = grouped_financial_df.alias("fd_1")
    grouped_financial_df_2 = grouped_financial_df.alias("fd_2")

    exploded_df = (
        sample_df.withColumn("sample", F.explode("samples"))
        .drop("samples")
        .withColumn("sample_entity_id", c("sample.entity_id"))
    )

    exploded_df_w_fs = exploded_df.join(
        grouped_financial_df_1, c("fd_1.entity_id") == exploded_df.entity_id
    ).join(
        grouped_financial_df_2,
        c("fd_2.entity_id") == exploded_df.sample_entity_id,
        how="left",
    )

    base_name_mapper = {
        "entity_id": "base_entity_id",
        "name": "base_name",
        "business_description": "base_business_description",
        "country_of_incorporation_iso": "base_country",
        "sample.entity_id": "sample_entity_id",
        "sample.country": "sample_country",
        "sample.description": "sample_business_description",
        "sample.name": "sample_name",
    }

    outcome = exploded_df_w_fs.select(
        [exploded_df[k].alias(v) for k, v in base_name_mapper.items()]
        + [
            c("fd_1.financials").alias("base_financials"),
            c("fd_2.financials").alias("sample_financials"),
        ]
    )

    outcome.write.parquet(args.output)

    spark.stop()


def cleaned_financial_from_dp(
    financial_data, dime_map, exchange_rate, start_yr, end_yr, spark
):
    if type(financial_data) is str:
        financial_df = spark.read.load(financial_data)
    else:
        financial_df = financial_data
    dime_df = spark.read.csv(dime_map, header=True)
    rate_df = spark.read.load(exchange_rate).select("isocode", "price_date", "rate")
    # Deduplicating Financials
    financial_df = (
        financial_df.filter(
            (financial_df.period_type == "ANNUAL")
            & (F.year("event_date").between(start_yr, end_yr))
        )
        .join(dime_df, dime_df.oneid == financial_df.data_item_id)
        .filter(dime_df.mnemonic.isin(list(_MNEMONICS)))
    )
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
    financial_df = (
        financial_df.withColumn(
            "parsed_value",
            F.from_json(
                "data_value",
                schema=T.StructType(
                    [
                        T.StructField("value", T.DoubleType()),
                        T.StructField("currency", T.StringType()),
                    ]
                ),
            ),
        )
        .drop("data_value")
        .withColumnRenamed("parsed_value", "data_value")
    )
    # Deduplicating financials done
    # Normalise all financials to USD.
    financial_df = (
        financial_df.join(
            rate_df,
            (rate_df.isocode == c("data_value.currency"))
            & (rate_df.price_date == financial_df.event_date),
        )
        .withColumn("normalised_value", c("data_value.value") * rate_df.rate)
        .drop("rate", "price_date", "isocode", "data_value")
    )
    grouped_financial_df = financial_df.groupBy("entity_id").agg(
        F.collect_list(
            F.struct(
                financial_df.normalised_value, dime_df.mnemonic, financial_df.event_date
            )
        ).alias("financials")
    )
    return grouped_financial_df


if __name__ == "__main__":
    main()
