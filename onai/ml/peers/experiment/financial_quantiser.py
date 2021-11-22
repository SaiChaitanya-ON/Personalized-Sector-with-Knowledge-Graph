import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from pyspark.ml.feature import Bucketizer

from onai.ml.peers.feature_extractor import convert_monetary_financial_to_value
from onai.ml.peers.types import Financial, Financials

logger = logging.getLogger(__name__)


@dataclass
class Config:
    quantisation_points: Tuple[float, ...] = MISSING


class FinancialQuantiser:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        for prev, curr in zip(
            self.cfg.quantisation_points, self.cfg.quantisation_points[1:]
        ):
            assert 0 < prev < curr

        _full_q_points = np.array(self.cfg.quantisation_points)
        self._full_q_points = np.concatenate(
            [[np.NINF], -_full_q_points[::-1], [0], _full_q_points, [np.PINF]]
        )
        self.pad_token_id = 0
        self.unk_token_id = len(self._full_q_points)

    def dequantise(self, ts):
        ret = []
        for t in ts:
            t = t.item()
            if t == self.pad_token_id:
                ret.append("PAD")
            elif t == self.unk_token_id:
                ret.append("UNK")
            else:
                ret.append(
                    f"{float(self._full_q_points[t-1]):.4f} - "
                    f"{float(self._full_q_points[t]):.4f}"
                )
        return ", ".join(ret)

    def batch_quantise(self, tss: List[List[Optional[float]]]):
        for ts in tss:
            assert len(ts) == len(tss[0])

        tss_arr = np.full((len(tss), len(tss[0])), 0.0, dtype=np.float)

        for idx, ts in enumerate(tss):
            tss_arr[idx, :] = ts

        unk_mask = np.isnan(tss_arr)
        # placeholder to prevent searchsorted go nuts
        tss_arr[unk_mask] = 0.0

        quantised_results = np.searchsorted(self._full_q_points, tss_arr, side="right")

        quantised_results[unk_mask] = self.unk_token_id
        return torch.tensor(quantised_results)

    @property
    def n_quantisation_points(self):
        # positive values and negative values occupies 1, 2, ... len(self._full_q_points)
        # padded token uses 0
        # unknown token uses the last token
        # in total there are n_quantisation_points
        return len(self._full_q_points) + 1

    def align_pad_quantise_financials(
        self,
        fss: List[Financials],
        output_mnemonics: List[str],
        max_financial_yrs: int = 16,
    ):
        end_year_by_mnemonic = {}
        currency = None
        for fs in fss:
            for mnemonic, ts in fs.items():
                for v in ts:
                    if mnemonic not in end_year_by_mnemonic:
                        end_year_by_mnemonic[mnemonic] = v.year
                    end_year_by_mnemonic[mnemonic] = max(
                        v.year, end_year_by_mnemonic[mnemonic]
                    )
                    if currency is None and v.currency:
                        currency = v.currency
                    assert (
                        v.currency is None or currency == v.currency
                    ), f"Mismatch currency {currency} v.s. {v.currency}:\n{fss}"

        # two outcomes
        # 1) end_year_by_mnemonic[k] exists
        # 2) end_year_by_mnemonic[k] does not exist at all across the whole batch
        # The default deals w/ the 2nd case
        tss_by_mnemonic = {
            k: [[None] * max_financial_yrs for _ in range(len(fss))]
            for k in output_mnemonics
        }
        for row_idx, fs in enumerate(fss):
            for mnemonic in fs.keys() & output_mnemonics:
                ts = fs[mnemonic]
                end_yr = end_year_by_mnemonic[mnemonic]
                for f in ts:
                    if f.val is None:
                        continue
                    offset = end_yr - f.year
                    if offset < max_financial_yrs:
                        tss_by_mnemonic[mnemonic][row_idx][
                            offset
                        ] = convert_monetary_financial_to_value(f)
        return {k: self.batch_quantise(tss_by_mnemonic[k]) for k in output_mnemonics}


def main():
    from onai.ml.spark_cache import setup_logger
    import argparse
    import multiprocessing as mp
    from onai.ml.spark import get_spark
    from pyspark.sql.functions import col as c
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    from smart_open import open
    import tabulate

    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-p", type=int, default=mp.cpu_count())
    parser.add_argument("-s", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spark = get_spark(memory="4g", n_threads=args.p)
    df = spark.read.load(args.i)

    agg_df = (
        df.select(
            c("base_entity_id").alias("entity_id"),
            c("base_financials").alias("financials"),
        )
        .union(
            df.select(
                c("sample_entity_id").alias("entity_id"),
                c("sample_financials").alias("financials"),
            )
        )
        .drop_duplicates(["entity_id"])
        .withColumn("financial", F.explode("financials"))
        .drop("financials")
        .select(
            c("financial.normalised_value").alias("value"),
            c("financial.mnemonic").alias("mnemonic"),
        )
    )

    agg_arr = agg_df.filter(c("value") > 0).orderBy("value").collect()

    split_idx = [
        min(math.ceil((i + 1) * (len(agg_arr) / args.s)), len(agg_arr) - 1)
        for i in range(args.s)
    ]

    # we need to append an eps to every values so that the last bucket guarantees to be empty, and
    # act just as an sentry.
    # if we do not add eps, the last element in the ordered agg_df, i.e. the one with the largest
    # number, will be put to the last bucket which is
    # meaningless to any parameterised model that does not model the relationship among the buckets.
    # note that surrogate model that we are building does not model the relationship among buckets
    # explicitly.
    eps = 1.0
    splits = [agg_arr[idx].value + eps for idx in split_idx]

    quantiser = FinancialQuantiser(Config(quantisation_points=splits))
    bins = quantiser._full_q_points
    logger.info("Bins: %s", bins)
    bucketed_df = (
        Bucketizer(splits=bins, inputCol="value", outputCol="bucket")
        .transform(agg_df)
        .withColumnRenamed("bucket", "bucket_double")
        .withColumn("bucket", c("bucket_double").cast(T.IntegerType()))
        .drop("bucket_double")
        .cache()
    )

    hists = bucketed_df.groupBy("bucket").count().collect()
    counts = [0.0] * (len(bins) - 1)

    for h in hists:
        counts[h["bucket"]] = h["count"]

    def print_hist(bins, counts):
        tab = []
        total_sz = sum(counts)
        for idx, (start_bin, end_bin, count) in enumerate(
            zip(bins, bins[1:], counts), 1
        ):
            tab.append(
                [
                    idx,
                    f"{start_bin:.4f}",
                    f"{end_bin:.4f}",
                    count,
                    f"{count / total_sz * 100:.2f}%",
                ]
            )
        logger.info(
            "Histogram: \n%s",
            tabulate.tabulate(
                tab, headers=["", "left_edge", "right_edge", "counts", "counts(%)"]
            ),
        )

    print_hist(bins, counts)

    logger.info("Per financials stats: \n")

    per_group_hists = bucketed_df.groupBy(["mnemonic", "bucket"]).count().collect()

    per_group_counts = defaultdict(lambda: [0.0] * (len(bins) - 1))
    for h in per_group_hists:
        per_group_counts[h["mnemonic"]][h["bucket"]] = h["count"]

    for k, v in per_group_counts.items():
        logger.info("Histogram for %s", k)
        print_hist(bins, v)

    with open(args.output, "w") as fout:
        fout.write(OmegaConf.structured(quantiser.cfg).pretty())


if __name__ == "__main__":
    main()
