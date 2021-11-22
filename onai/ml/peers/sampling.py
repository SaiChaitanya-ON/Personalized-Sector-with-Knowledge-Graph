import argparse
import json
import logging
import math

import pandas as pd
import smart_open
from pyspark.sql import SparkSession

from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)

REVENUE_MARGINS = {  # [wtg, min, max]
    "r1": [0.77, 0, 100],
    "r2": [0.1, 100, 200],
    "r3": [0.02, 200, 300],
    "r4": [0.04, 300, 400],
    "r5": [0.02, 400, 500],
    "r6": [0.02, 500, 600],
    "r7": [0.02, 2900, 3000],
}


def create_pandas_df(columns):
    return pd.DataFrame([], columns=columns)


def get_config(args) -> dict():
    with smart_open.open(args.config, "r") as fp:
        conf = json.loads(fp.read())
    conf["TOTAL_SAMPLES"] = int(args.s)
    return conf


def get_random(df, n):
    count = len(df)
    if count == 0:
        return None

    required = n if (count > n) else count
    result = df.sample(n=required)
    return result


def sampling(raw_data, conf):
    clean_data = raw_data.where(
        (raw_data.region.isin(conf["REGION_LIST"]))
        & (raw_data.primary_naics_node_desc.isin(conf["SECTOR_LIST"]))
    )

    region_list_length = conf["REGION_LIST"].__len__()
    final_sample_df = create_pandas_df(clean_data.columns)

    for i in conf["SECTOR_LIST"]:
        logger.info(i)
        valid_sector = True
        sector_sample = create_pandas_df(clean_data.columns)

        sector_data = clean_data.filter(clean_data.primary_naics_node_desc == i)
        if sector_data.count() == 0:
            continue

        clubed_margins = create_pandas_df(clean_data.columns)

        for _, margin in REVENUE_MARGINS.items():
            logger.info(margin)

            margin_data = sector_data.where(
                (sector_data.TOTAL_REVENUE > (margin[1] * 10 ** 6))
                & (sector_data.TOTAL_REVENUE <= (margin[2] * 10 ** 6))
            )

            margin_data_count = margin_data.count()
            margin_samples_req = round(margin[0] * conf["TOTAL_SAMPLES"])

            if (margin_data_count == 0) | (margin_data_count < margin_samples_req):
                logger.info("NOT ENOUGH DATA TO SAMPLE ")
                valid_sector = False
                break

            # when number of samples required are less than 0.5
            elif margin_samples_req == 0:
                temp_sample = margin_data.toPandas().sample(n=1)
                clubed_margins = clubed_margins.append(temp_sample)
                continue

            temp_sample = create_pandas_df(clean_data.columns)
            num_samples = math.ceil(margin_samples_req / region_list_length)

            for region in conf["REGION_LIST"]:
                temp_sample = temp_sample.append(
                    get_random(
                        margin_data.filter(margin_data.region == region).toPandas(),
                        num_samples,
                    )
                )

            # ENSURING SAMPLE SIZE IS AS REQUIRED
            if len(temp_sample) > margin_samples_req:
                temp_sample = temp_sample.sample(n=margin_samples_req)

            elif len(temp_sample) < margin_samples_req:
                valid_sector = False
                break

            sector_sample = sector_sample.append(temp_sample)

        if valid_sector:
            sample_deficit = conf["TOTAL_SAMPLES"] - len(sector_sample)
            if sample_deficit > 0:
                sector_sample = sector_sample.append(
                    clubed_margins.sample(n=sample_deficit)
                )

            final_sample_df = final_sample_df.append(sector_sample)

    return final_sample_df


def save_results(args, result):
    pth = args.out
    writer = pd.ExcelWriter(pth, engine="xlsxwriter")

    df_info = pd.DataFrame(
        columns=[
            "Case",
            "Base Borrower Name",
            "Base Borrower Description",
            "Base Borrower Region",
            "Base Borrower Sector",
            "Base Borrower Revenue Range",
            "INTERNAL",
            "FYE",
            "currency",
            "magnitude",
        ]
    )
    df_info[
        [
            "Base Borrower Name",
            "Base Borrower Description",
            "Base Borrower Region",
            "Base Borrower Sector",
        ]
    ] = result[["NAME", "BUSINESS_DESCRIPTION", "region", "primary_naics_node_desc"]]

    df_info["INTERNAL"] = [0] * args.s
    df_info["Case"] = result.index.to_list()
    df_info.to_excel(writer, sheet_name="Info", index=False)
    # finantial sheet
    df_fin = pd.DataFrame()
    df_fin.to_excel(writer, sheet_name="Financials", index=False)

    writer.save()
    logger.info("done !!")


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="The path to the parquet data", required=True)
    parser.add_argument("--config", help="locaiton of config file", required=True)
    parser.add_argument("--out", help="results will be saved here", required=True)
    parser.add_argument("--s", help="samples required.", default=20, type=int)
    args = parser.parse_args()

    conf = get_config(args)

    spark = (
        SparkSession.builder.master("local[*]").appName("SparkSampling").getOrCreate()
    )
    spark_df = spark.read.parquet(args.path).repartition(1024)

    logger.info("starting to sample data..")
    results = sampling(spark_df, conf)
    logger.info("done with sampling")

    if len(results) > 0:
        logger.info("Storing results..")
        results.reset_index(inplace=True, drop=True)
        save_results(args, results)


if __name__ == "__main__":
    main()
