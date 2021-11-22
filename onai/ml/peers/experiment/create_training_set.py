import argparse
import logging

import numpy as np
import pandas as pd
import smart_open
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, BertTokenizer

from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.embedding_retriever import EmbeddingRetriever
from onai.ml.peers.feature_extractor import PeerFeatureExtractor
from onai.ml.peers.types import PeerSuggestion
from onai.ml.tools.argparse import add_bool_argument
from onai.ml.tools.logging import _clean_hdlrs, setup_logger

logger = logging.getLogger(__name__)


def main():
    _clean_hdlrs()
    setup_logger()
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_company_path")
    parser.add_argument("--annotated_peers_csv")
    parser.add_argument("--output")
    parser.add_argument(
        "--analyser",
        default="s3://oaknorth-ml-dev-eu-west-1/delan/data/idf_model_clip12.pkl",
    )
    parser.add_argument("--es_host", default="berry-es-test.ml.onai.cloud")
    parser.add_argument("--es_port", default="80")
    parser.add_argument("--target_currency", default="EUR")
    parser.add_argument("--pretrained_albert_name", default="albert-base-v2")
    parser.add_argument("--pretrained_bert_name", default="bert-base-uncased")
    parser.add_argument("--test_size", default=0.15, type=float)
    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--bert_tokens", default="bert")
    add_bool_argument(
        parser,
        "split_data",
        default=False,
        help="Whether to split part of the data to be test data",
    )
    add_bool_argument(parser, "cuda", default=False)
    add_bool_argument(parser, "pretrained_emb", default=True)
    add_bool_argument(parser, "infer_subsidiary", default=True)

    args = parser.parse_args()

    base_companies = pd.read_excel(args.base_company_path, ["Info", "Financials"])
    annotations = pd.read_csv(args.annotated_peers_csv)

    def preprocess_df(df: pd.DataFrame):
        if len(set(df["relevance_score"])) == 1:
            s = df.iloc[0]
            logger.info(
                "%s_%s has no meaningful annotation. Discarding.",
                s["task_id"],
                s["analyst_id"],
            )
            return None
        df["es_rank"] = np.arange(len(df.index))
        return df

    annotations = annotations.groupby(["task_id", "analyst_id"]).apply(preprocess_df)

    es = ESCandidateSuggestion(
        es_host=args.es_host,
        es_port=args.es_port,
        es_index="company",
        internal_ds=args.base_company_path,
        tgt_currency=args.target_currency,
    )

    emb = EmbeddingRetriever(args.cuda) if args.pretrained_emb else None
    albert_tokeniser = AlbertTokenizer.from_pretrained(args.pretrained_albert_name)
    bert_tokeniser = BertTokenizer.from_pretrained(args.pretrained_bert_name)
    feature_extractor = PeerFeatureExtractor(
        es_client=es,
        embedding_retriever=emb,
        albert_tokenizer=albert_tokeniser,
        bert_tokenizer=bert_tokeniser,
        analyser_path=args.analyser,
    )

    base_company_details_by_name = {}

    base_company_details_by_name = {
        name: cd
        for cd, name in zip(
            es.get_companies_detail_by_name(
                base_companies["Info"]["Base Borrower Name"]
            ),
            base_companies["Info"]["Base Borrower Name"],
        )
    }

    # get the set of names in peers, get all their CompanyDetails

    peer_eids = list(annotations["peer_entity_id"].dropna())
    peer_company_details_by_eid = {
        eid: cd
        for eid, cd in zip(peer_eids, es.get_companies_detail_by_id(peer_eids))
        if cd is not None
    }

    # only those do not have entity ids will fall back to search by name mode.
    peer_names = annotations[annotations["peer_entity_id"].isna()]["peer_name"]
    peer_company_details_by_name = {
        cd.name: cd
        for cd in es.get_companies_detail_by_name(peer_names)
        if cd is not None
    }

    def featurise(df: pd.DataFrame):
        name = df["base_name"].iloc[0]
        logger.info("Extracting features for %s", name)
        assert all(df["base_name"] == name)
        # there are consensus, remove duplicate peers
        df = df[~df.duplicated("peer_name")]
        base_company = base_company_details_by_name[name]
        peer_companies = []
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            eid = row["peer_entity_id"]
            name = row["peer_name"]
            if not pd.isna(eid) and eid in peer_company_details_by_eid:
                peer_companies.append(
                    PeerSuggestion(rank, peer_company_details_by_eid[eid], 1.0)
                )
            elif name in peer_company_details_by_name:
                peer_companies.append(
                    PeerSuggestion(rank, peer_company_details_by_name[name], 1.0)
                )
            else:
                logger.warning("Cannot find f%s by either entity id or name", row)

        ret = pd.DataFrame(
            feature_extractor.extract_features_for_suggestions(
                base_company,
                peer_companies,
                pretrained_emb=args.pretrained_emb,
                bert_tokens=args.bert_tokens,
                infer_subsidiary=args.infer_subsidiary,
                end_year=max(df["end_yr"]),
            )
        )
        return ret

    features_df = annotations.groupby("base_name", as_index=False).apply(featurise)

    with smart_open.open(f"{args.output}/query_document_features.csv", "w") as f:
        features_df.to_csv(f, index=False)

    all_df = features_df.merge(annotations, on=["peer_name", "base_name"])
    task_ids = np.array(list(set(all_df["task_id"])))

    def write_selected_tasks(task_ids, prefix):
        df = all_df[all_df["task_id"].isin(task_ids)]
        df.to_csv(f"{args.output}/{prefix}.csv", index=False)
        with smart_open.open(f"{args.output}/{prefix}.pkl", "wb") as fout:
            df.to_pickle(fout, None)

    if args.split_data:
        train_ids, test_ids = train_test_split(
            task_ids, random_state=345, test_size=args.test_size
        )
        train_ids, val_ids = train_test_split(
            train_ids, random_state=333, test_size=args.val_size
        )
        train_ids = set(train_ids)
        test_ids = set(test_ids)
        val_ids = set(val_ids)

        write_selected_tasks(train_ids, "train_df")
        write_selected_tasks(test_ids, "test_df")
        write_selected_tasks(val_ids, "val_df")

    else:
        write_selected_tasks(set(all_df["task_id"]), "train_df")

    with smart_open.open(f"{args.output}/query_document_features_all_df.csv", "w") as f:
        all_df.to_csv(f, index=False)


if __name__ == "__main__":
    main()
