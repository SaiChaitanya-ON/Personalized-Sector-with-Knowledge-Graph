import argparse
import copy
import datetime
import functools
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from typing import List, Optional

import pandas
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import col as c
from pyspark.sql.functions import pandas_udf

from onai.ml.peers.types import CompanyDetail, Financial
from onai.ml.spark import get_spark
from onai.ml.spark_cache import (
    get_bert_cs,
    get_region_model,
    setup_logger,
    speed_counter,
)
from onai.ml.tools.argparse import add_bool_argument

logger = logging.getLogger(__name__)


def row_to_company_detail(row: T.Row, region_model, prefix="") -> CompanyDetail:
    financials = defaultdict(list)
    base_fye = None
    for financial in row[f"{prefix}financials"] or []:
        base_fye = datetime.date.fromisoformat(financial["event_date"])
        financials[financial["mnemonic"]].append(
            Financial(financial["normalised_value"], base_fye.year, currency="USD")
        )

    return CompanyDetail(
        row[f"{prefix}name"],
        row[f"{prefix}business_description"],
        region_model.get(row[f"{prefix}country"], None),
        base_fye,
        financials,
        country=row[f"{prefix}country"],
        entity_id=row[f"{prefix}entity_id"],
    )


def group_scorer(ls: List[T.Row], args) -> Optional[List[float]]:
    setup_logger()
    first_row = ls[0]
    try:
        region_model = get_region_model()
        bert_scorer = get_bert_cs(args)
        base_company = row_to_company_detail(first_row, region_model, prefix="base_")

        # just check for sanity
        for r in ls:
            assert r["base_entity_id"] == base_company.entity_id, "impossible!!!!!!"

        peer_companies = [row_to_company_detail(r, region_model, "sample_") for r in ls]
        scores = bert_scorer.score_candidates(
            base_company, peer_companies, args.end_year
        )
        logger.info("Task Speed @ %s: %f", os.getpid(), speed_counter())
        return scores
    except:
        logger.exception("Base Company %s.", first_row["base_entity_id"])
        return None


def populate_argparser(parser):
    parser.add_argument(
        "-i", help="the path to the ouput from surrogate_data_features", required=True
    )
    parser.add_argument(
        "--output", required=True, help="the path to the data w/ scores"
    )
    # TODO: should support more different types of models
    parser.add_argument(
        "--bert_model_path", required=True, help="The path to the bert model"
    )
    parser.add_argument("-p", type=int, default=mp.cpu_count())
    parser.add_argument("--bert_partitions", type=int, default=2)
    parser.add_argument("--es_host", default="berry-es-test.ml.onai.cloud")
    parser.add_argument("--es_port", default=80, type=int)
    parser.add_argument("--end_year", default=2020, type=int)
    parser.add_argument("--pred_batch_sz", default=64, type=int)
    parser.add_argument("--limit", type=int)
    add_bool_argument(parser, "cuda", default=False)


def main(args=None):

    setup_logger()
    if args is None:
        parser = argparse.ArgumentParser()
        populate_argparser(parser)
        args = parser.parse_args()

    spark = get_spark(n_threads=args.p, memory="20g")
    df = spark.read.load(args.i)

    grouped_df = (
        df.groupBy("base_entity_id")
        .agg(F.collect_list(F.struct(*(df[col] for col in df.columns))).alias("ls"))
        .coalesce(args.bert_partitions)
    )

    udf = F.udf(functools.partial(group_scorer, args=args), T.ArrayType(T.DoubleType()))

    if args.limit:
        grouped_df = grouped_df.limit(args.limit)

    outcome = grouped_df.withColumn("score", udf(grouped_df.ls)).filter(
        c("score").isNotNull()
    )

    outcome.write.parquet(args.output)


if __name__ == "__main__":
    main()
