import argparse
import json
import logging
import multiprocessing as mp
import os
import textwrap

import numpy as np
import pyarrow.parquet as pq
import tabulate
import torch

from onai.ml.peers.candidate_suggestion.bert import BertRankNetCandidateSuggestion
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.candidate_suggestion.surrogate import SurrogateCandidateSuggestion
from onai.ml.peers.feature_extractor import last_reported_financial
from onai.ml.peers.types import CompanyDetail
from onai.ml.spark import get_spark
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


def main():
    setup_logger(blacklisted_loggers=["elasticsearch"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--company_index", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bert_model_path", required=True)
    parser.add_argument("--top_k", default=10, type=int)
    # spark = get_spark(memory="8g", n_threads=mp.cpu_count())
    args = parser.parse_args()
    ds = pq.ParquetDataset(args.company_index)
    tbl = ds.read().to_pandas()
    eids = tbl["entity_id"].to_list()
    eid_to_idx = {eid: idx for idx, eid in enumerate(eids)}
    # h x C
    embs = np.array(tbl["embedding"].to_list()).transpose([1, 0])
    del tbl

    logger.info("Index loaded: size %s", embs.shape)
    ES_HOST = "berry-es-test.ml.onai.cloud"
    ES_PORT = 80
    cs = SurrogateCandidateSuggestion(
        os.path.join(args.model_path, "scorer_cfg.yaml"),
        os.path.join(args.model_path, "best_model.pth"),
        es_host=ES_HOST,
        es_port=ES_PORT,
    )

    es_cs = ESCandidateSuggestion(es_host=ES_HOST, es_port=ES_PORT)

    bert_cs = BertRankNetCandidateSuggestion(
        os.path.join(args.bert_model_path, "scorer_cfg.json"),
        os.path.join(args.bert_model_path, "best_model.pth"),
        es_host=ES_HOST,
        es_port=ES_PORT,
    )

    financial_key = ("TOTAL_REVENUE", "EBIT", "EBITDA")

    while True:
        try:
            query_eid = input("eid for the company query: ")

            company_detail = cs.get_companies_detail_by_id([query_eid])
            # 1 x h
            company_repr: np.ndarray = cs.get_repr(company_detail).numpy()
            dist_type = cs.cfg.dist_type

            if dist_type == "inner":
                # C
                surrogate_scores = np.matmul(company_repr, embs).squeeze(0)
            elif dist_type == "l2":
                # C
                surrogate_scores = -np.linalg.norm(
                    company_repr.transpose([1, 0]) - embs, ord=2, axis=0
                )
            else:
                assert False, "Unsupported distance type: %s" % dist_type
            sorted_idxes = np.argsort(surrogate_scores)[::-1][1 : args.top_k + 1]

            reqs = [eids[s] for s in sorted_idxes]
            surrogate_cands = cs.get_companies_detail_by_id(reqs)

            tab = tabulate_suggestions(
                company_detail[0],
                financial_key,
                surrogate_cands,
                surrogate_scores[sorted_idxes],
            )

            logger.info("Results from Surrogate Model: \n %s", tab)

            es_cands = es_cs.suggest_candidates(company_detail[0])

            tab_vanilla = tabulate_suggestions(
                company_detail[0],
                financial_key,
                [p.detail for p in es_cands],
                [p.score for p in es_cands],
            )

            logger.info("Results from ES vanilla model: \n %s", tab_vanilla)

            agg_reqs = set(c.entity_id for c in surrogate_cands) | set(
                c.detail.entity_id for c in es_cands
            )

            sorted_eids = sorted(
                agg_reqs,
                key=lambda eid: surrogate_scores[eid_to_idx[eid]],
                reverse=True,
            )
            agg_cands = cs.get_companies_detail_by_id(sorted_eids)
            tab_agg_surrogate = tabulate_suggestions(
                company_detail[0],
                financial_key,
                agg_cands,
                [surrogate_scores[eid_to_idx[eid]] for eid in sorted_eids],
            )

            logger.info("Results from Aggregated Surrogate: \n %s", tab_agg_surrogate)

            scores = bert_cs.score_candidates(company_detail[0], agg_cands, 2020)
            assert len(scores) == len(agg_cands)
            sorted_score, sorted_bert_cands = zip(
                *sorted(zip(scores, agg_cands), key=lambda x: x[0], reverse=True)
            )

            tag_agg_bert = tabulate_suggestions(
                company_detail[0],
                financial_key,
                list(sorted_bert_cands),
                list(sorted_score),
            )

            logger.info("Results from BERT scorer: \n %s", tag_agg_bert)
        except EOFError:
            break


def tabulate_suggestions(company_detail, financial_key, peer_company_details, scores):
    tab = []
    for idx, c in enumerate([company_detail] + peer_company_details):
        financial_val = [
            last_reported_financial(c.financials, k) for k in financial_key
        ]
        financial_val = ["N/A" if f is None else f.val for f in financial_val]

        if c is None:
            tab.append([])
            continue
        score = "N/A" if idx == 0 else scores[idx - 1]
        tab.append(
            [
                idx,
                textwrap.fill(c.name, width=35),
                c.country,
                textwrap.fill(c.description, width=70),
                score,
            ]
            + financial_val
        )
    return tabulate.tabulate(
        tab,
        headers=("Order", "Name", "Country", "Description", "Scores") + financial_key,
    )


if __name__ == "__main__":
    main()
