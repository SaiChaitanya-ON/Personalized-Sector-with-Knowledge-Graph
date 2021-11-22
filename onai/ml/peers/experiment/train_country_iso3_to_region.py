import argparse
import json
import logging
import multiprocessing as mp
from collections import Counter, defaultdict

import pycountry
import pyspark.sql.functions as F
import pyspark.sql.types as T
import tabulate
from pyspark.sql.functions import col as c
from smart_open import open

from onai.ml.spark import get_spark
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company_data",
        default="s3a://one-lake-prod/business/company_data_denormalized",
    )
    parser.add_argument("--output", required=True)

    parser.add_argument("-p", type=int, default=mp.cpu_count())

    args = parser.parse_args()

    spark = get_spark(memory="4g", n_threads=args.p)

    company_data_denormalized = spark.read.load(args.company_data)
    iso_region_df = (
        company_data_denormalized.select(
            [c(x).alias(x.lower()) for x in company_data_denormalized.columns]
        )
        .select("country_of_incorporation_iso", "region")
        .filter("country_of_incorporation_iso is not null and region is not null")
    )

    total_sz = iso_region_df.count()
    logger.info("Companies with country and region info: %d", total_sz)
    logger.info("Total companies: %d", company_data_denormalized.count())
    # region, country, count
    res = (
        iso_region_df.groupBy("region", "country_of_incorporation_iso")
        .count()
        .collect()
    )

    cond_prob = defaultdict(Counter)

    for r in res:
        cond_prob[r["country_of_incorporation_iso"]][r["region"]] += r["count"]

    tab = []

    out_dict = {}
    for country_iso_3, counters in sorted(cond_prob.items(), key=lambda x: x[0]):
        counters: Counter = counters
        total_sz = sum(counters.values())
        country = pycountry.countries.get(alpha_3=country_iso_3)
        for region, count in counters.most_common(2):
            tab.append((country.name, region, count, f"{count/total_sz*100:.2f}%"))
        out_dict[country_iso_3] = counters.most_common(1)[0][0]
    logger.info(
        "Stats: \n%s",
        tabulate.tabulate(
            tab, headers=("Country", "Region", "Count", "Conditional Prob")
        ),
    )
    with open(args.output, "w") as fout:
        json.dump(out_dict, fout)

    def predict_region(country_iso_3):
        if country_iso_3 in out_dict:
            return out_dict[country_iso_3]

    predict_region_udf = F.udf(predict_region, T.StringType())

    # get how many companies can obtain the region field back
    company_data_denormalized = (
        company_data_denormalized.filter("country_of_incorporation_iso is not null")
        .withColumn(
            "predicted_region", predict_region_udf("country_of_incorporation_iso")
        )
        .filter("predicted_region is not null")
    )

    logger.info(
        "Companies with predicted region, %d", company_data_denormalized.count()
    )


if __name__ == "__main__":
    main()
