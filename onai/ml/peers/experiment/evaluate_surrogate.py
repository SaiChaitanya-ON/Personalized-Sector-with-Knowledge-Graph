import argparse
import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from onai.ml.peers.candidate_suggestion.surrogate import SurrogateCandidateSuggestion
from onai.ml.peers.experiment.evaluate_consensus import handle_file
from onai.ml.peers.metric import ndcg
from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


def score_df(
    df: pd.DataFrame, cs: SurrogateCandidateSuggestion, f, args
) -> Tuple[Optional[float], Optional[float]]:
    # convert all companies into CompanyDetail
    # we will do that by getting the information from elastic search
    # and data platform
    # 1) we need to understand what is the end year (and start year)
    try:
        financial_df = pd.read_excel(f, "Peer Financials", skiprows=[0])
        # we just need end year
        end_yr = financial_df.columns[-1]
        start_yr = end_yr - cs.cfg.max_financial_yrs
    except:
        logger.warning("Unsupported format, ignoring %s", f)
        return None, None

    peer_eids = list(df["peer_entity_id"].dropna())
    company_details_by_eid = {
        eid: cd
        for eid, cd in zip(
            peer_eids,
            cs.get_companies_detail_by_id(
                peer_eids, start_year=start_yr, end_year=end_yr
            ),
        )
        if cd is not None
    }
    # old entity ids do not work anymore because one API decides they want to
    # change the id
    missing_companies = set(peer_eids) - set(company_details_by_eid.keys())
    # only those do not have entity ids will fall back to search by name mode.
    company_names = df[
        df["peer_entity_id"].isna() | df["peer_entity_id"].isin(missing_companies)
    ].index

    base_name = df.index[0]
    logger.info(base_name)
    company_details_by_name = {
        cd.name: cd
        for cd in cs.get_companies_detail_by_name(
            company_names, start_year=start_yr, end_year=end_yr
        )
        if cd is not None
    }
    if base_name not in company_details_by_name:
        logger.warning("Ignoring %s as this company cannot be found.", base_name)
        return None, None
    base_company = company_details_by_name[base_name]
    peer_companies = []
    golden_scores = []
    for _, row in df.iloc[1:].iterrows():
        eid = row["peer_entity_id"]
        name = row.name
        if not pd.isna(eid) and eid in company_details_by_eid:
            peer_companies.append(company_details_by_eid[eid])
            golden_scores.append(row["relevance_score"])
        elif name in company_details_by_name:
            peer_companies.append(company_details_by_name[name])
            golden_scores.append(row["relevance_score"])
        else:
            logger.warning(
                f"Cannot find %s by either entity id or name. Ignoring", name
            )

    if len(set(golden_scores)) <= 1:
        logger.warning("%s is monolithic annotation. Ignoring.")
        return None, None

    scores = cs.score_candidates(base_company, peer_companies, None)
    assert len(scores) == len(golden_scores)
    sorted_tst = [
        i[1]
        for i in sorted(zip(scores, golden_scores), key=lambda x: x[0], reverse=True)
    ]
    return ndcg(sorted_tst), ndcg(golden_scores)


def main():
    setup_logger(blacklisted_loggers=["elasticsearch"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument(
        "--inputs",
        help="All the annotation files. One per base borrower. "
        "The file name should be named according to the following format "
        "(Peer_Annotation_B$BASE_BORROWER_IDX_$ANNOTATOR_IDX)",
        nargs="+",
    )
    args = parser.parse_args()
    cs = SurrogateCandidateSuggestion(
        os.path.join(args.model_path, "scorer_cfg.yaml"),
        os.path.join(args.model_path, "best_model.pth"),
        es_host="berry-es-test.ml.onai.cloud",
        es_port=80,
    )
    surrogate_ndcgs = []
    orig_ndcgs = []
    for f in args.inputs:
        df = handle_file(f, False)
        surrogate_ndcg, orig_ndcg = score_df(df, cs, f, args)
        if surrogate_ndcg is None or orig_ndcg is None:
            continue
        assert not math.isnan(surrogate_ndcg)
        assert not math.isnan(orig_ndcg)
        surrogate_ndcgs.append(surrogate_ndcg)
        orig_ndcgs.append(orig_ndcg)
    logger.info("Average Surrogate NDCG score: %f", np.mean(surrogate_ndcgs))
    logger.info("Average Original NDCG score: %f", np.mean(orig_ndcgs))


if __name__ == "__main__":
    main()
