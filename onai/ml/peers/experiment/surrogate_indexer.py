import argparse
import functools
import json
import logging
import multiprocessing as mp
import os
import tempfile
from collections import defaultdict
from datetime import date

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import torch
from pyspark.sql.functions import col as c

from onai.ml.peers.experiment.surrogate_data_features import cleaned_financial_from_dp
from onai.ml.peers.experiment.surrogate_data_sampler import get_country_filter
from onai.ml.peers.types import CompanyDetail, Financial
from onai.ml.spark import get_spark
from onai.ml.spark_cache import (
    get_region_model,
    get_surrogate_cs,
    setup_logger,
    speed_counter,
)
from onai.ml.tools.argparse import add_bool_argument

logger = logging.getLogger(__name__)


def inference(names, descs, countries, financials: pd.Series, eids: pd.Series, args):
    setup_logger()
    region_model = get_region_model()
    cs = get_surrogate_cs(
        args.model_path, args.cuda, mp.cpu_count() // args.bert_threads
    )
    companies = []
    for name, desc, country, financial, eid in zip(
        names, descs, countries, financials, eids
    ):
        financial = json.loads(financial) if financial else []
        base_fye = None
        out_financial = defaultdict(list)
        for f in financial:
            base_fye = date.fromisoformat(f["event_date"])
            out_financial[f["mnemonic"]].append(
                Financial(f["normalised_value"], base_fye.year, currency="USD")
            )

        companies.append(
            CompanyDetail(
                name,
                desc,
                region_model.get(country, None),
                base_fye,
                out_financial,
                country=country,
                entity_id=eid,
            )
        )
    reprs: torch.Tensor = cs.get_repr(companies)

    logger.info("Task Speed @ %s: %f", os.getpid(), speed_counter(len(names)))
    return pd.Series(data=reprs.tolist())


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("-p", type=int, default=mp.cpu_count())

    parser.add_argument(
        "--financial_data", default="s3://one-lake-prod/business/financial_data"
    )
    parser.add_argument("--dime_map", default="s3a://one-lake-prod/business/dime_map")
    parser.add_argument(
        "--exchange_rate", default="s3a://one-lake-prod/business/exchange_rate"
    )
    parser.add_argument(
        "--region_model",
        default="s3://oaknorth-staging-non-confidential-ml-artefacts/region/v0.0.1/",
    )
    parser.add_argument("--limit", type=int)
    add_bool_argument(parser, "cuda", default=False)
    parser.add_argument("--start_yr", type=int, default=2008)
    parser.add_argument("--end_yr", type=int, default=2020)
    parser.add_argument(
        "--company_data",
        default="s3a://one-lake-prod/business/company_data_denormalized",
    )
    # this has been tuned for our k8s set-up
    parser.add_argument("--bert_threads", type=int, default=4)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temp_dir", default="/tmp")
    args = parser.parse_args()
    spark = get_spark(n_threads=args.p, memory="10g")
    financial_data = spark.read.load(args.financial_data)

    country_filter_q = get_country_filter(args.region_model)
    company_data_denormalized = spark.read.load(args.company_data)
    company_data_denormalized = company_data_denormalized.select(
        [c(x).alias(x.lower()) for x in company_data_denormalized.columns]
    )
    company_data_denormalized = (
        company_data_denormalized.filter(country_filter_q)
        .withColumn("len_business_description", F.length("business_description"))
        .filter(
            "len_business_description is not null and len_business_description > 10"
        )
        .filter(F.year("last_filing_date").between(args.start_yr, args.end_yr))
    )
    if args.limit:
        company_data_denormalized = company_data_denormalized.limit(args.limit)

    financial_data = (
        financial_data.alias("fs")
        .join(
            company_data_denormalized.select("entity_id").alias("company"),
            c("company.entity_id") == c("fs.entity_id"),
        )
        .drop(c("company.entity_id"))
    )
    grouped_financial_df = cleaned_financial_from_dp(
        financial_data,
        args.dime_map,
        args.exchange_rate,
        args.start_yr,
        args.end_yr,
        spark,
    )

    company_df_w_fs = company_data_denormalized.alias("company").join(
        grouped_financial_df.alias("fs"),
        company_data_denormalized.entity_id == grouped_financial_df.entity_id,
        how="left",
    )

    with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp_dir:
        tmp_dir = os.path.join(tmp_dir, "data")
        company_df_w_fs.select(
            c("company.entity_id"),
            company_df_w_fs.name,
            company_df_w_fs.business_description,
            company_df_w_fs.country_of_incorporation_iso,
            # this is needed as structtype is not supported until pyspark 3.0
            F.to_json(company_df_w_fs.financials).alias("financials"),
        ).write.parquet(tmp_dir)

        spark.stop()

        spark = get_spark(
            memory="4g", n_threads=args.bert_threads, arrow_batch_size=args.batch_size
        )
        company_df_w_fs = spark.read.load(tmp_dir)
        # TODO: once update to pyspark 3.0.1, this should be changed to SCALAR_ITER
        inference_udf = F.pandas_udf(
            functools.partial(inference, args=args),
            T.ArrayType(T.FloatType()),
            F.PandasUDFType.SCALAR,
        )
        company_df_w_fs.repartition(1000).withColumn(
            "embedding",
            inference_udf(
                company_df_w_fs.name,
                company_df_w_fs.business_description,
                company_df_w_fs.country_of_incorporation_iso,
                company_df_w_fs.financials,
                company_df_w_fs.entity_id,
            ),
        ).select(c("entity_id"), c("embedding")).write.parquet(args.output)
        spark.stop()


if __name__ == "__main__":
    main()
