import argparse
import math
import os
from collections import Counter, defaultdict, namedtuple
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
from krippendorff import krippendorff

from onai.ml.peers.metric import average_precision, ndcg_df, precision_at_k

COMPANY_COL_NAME = "Company"
POS_NEG_COL_NAME = "Positive/Negative?"
REL_POS_COL_NAME = "Relative Position"
REASON_1_COL_NAME = "Reason 1 why it is negative "
REASON_2_COL_NAME = "Reason 2 why it is negative"
REASON_3_COL_NAME = "Reason 3 why it is negative"


def handle_file(f, get_end_year=False, remove_first_row=True) -> pd.DataFrame:
    print(f"Handling file {f}")
    df = pd.read_excel(f, "Output Peers Annotation", skiprows=[0])
    try:
        entity_df = pd.read_excel(
            f,
            "peer_entity_ids",
            skiprows=[0],
            header=None,
            names=["company_name", "peer_entity_id"],
        )
    except Exception:
        # no such page
        entity_df = pd.DataFrame(columns=["company_name", "peer_entity_id"])
    for idx, row in df.iterrows():
        company_name = row[COMPANY_COL_NAME]
        if not isinstance(company_name, str) and math.isnan(company_name):
            df = df.head(idx)
            break
    # remove the first row as it is the base borrower
    assert pd.isna(df[POS_NEG_COL_NAME][0])
    if remove_first_row:
        df: pd.DataFrame = df.iloc[1:]
    df["order"] = np.arange(0, len(df.index))
    df["relevance_score"] = df.apply(map_decision_to_rank, axis=1)
    df = df.merge(
        entity_df,
        how="left",
        left_on=COMPANY_COL_NAME,
        right_on="company_name",
        suffixes=("", ""),
    )
    df = df.drop("company_name", axis=1)
    if get_end_year:
        df["end_yr"] = pd.read_excel(f, header=1, sheet_name="Peer Financials").columns[
            -1
        ]
    return df.set_index(COMPANY_COL_NAME)


def fill_in_missing_annos(annos_by_task_id):
    for annos in annos_by_task_id.values():
        all_peers = set()
        for anno in annos:
            all_peers.update(anno["excel_df"].index)
        for anno in annos:
            df: pd.DataFrame = anno["excel_df"]
            missing_peers = all_peers - set(df.index)
            for missing_peer in missing_peers:
                df.loc[missing_peer] = pd.Series(
                    {POS_NEG_COL_NAME: "Negative", "relevance_score": 0},
                    index=df.columns,
                )
            anno["excel_df"] = df.sort_index()


def normalise_judgements(df: pd.DataFrame, pos_threshold: int = 0):
    return df["relevance_score"] > pos_threshold


AgreementStats = namedtuple(
    "AgreementStats",
    [
        "agreement",
        "disagreement",
        "suggestions_with_disagreement_level",
        "rel_agreement",
        "rel_disagreement",
        "relevance_counter",
        "error_case_counter",
        "n_training_pairs",
        "error_case_agreement",
        "error_case_disagreement",
    ],
)


def agreement_per_task(task_id, annos, white_list_analysts: Set[int], args):
    gold_template = annos[0]["excel_df"]
    company_names = gold_template.index
    n_peers = len(company_names)
    all_analysts = set(s["analyst_id"] for s in annos)
    if white_list_analysts:
        white_list_analysts = white_list_analysts & all_analysts
    else:
        white_list_analysts = all_analysts

    relevance_counter = Counter()
    error_case_counter = Counter()
    n_training_pairs = 0
    if not white_list_analysts:
        # there is no analyst for task
        return AgreementStats(
            agreement=0,
            disagreement=0,
            suggestions_with_disagreement_level=[],
            rel_agreement=0,
            rel_disagreement=0,
            relevance_counter=relevance_counter,
            error_case_counter=error_case_counter,
            n_training_pairs=0,
            error_case_agreement=0,
            error_case_disagreement=0,
        )

    n_annotators = len(white_list_analysts)

    for anno in annos[1:]:
        df: pd.DataFrame = anno["excel_df"]
        assert (df.index == company_names).all(), (
            f"{df.index} v.s. \n{company_names}\n"
            f"Task ID: {task_id}, Analyst ID: {anno['analyst_id']}, "
            f"Gold Analyst ID: {annos[0]['analyst_id']}"
        )

    suggestions_with_disagreement_level = []

    agreements_per_suggestion = np.zeros((n_peers,), dtype=np.int)
    disagreements_per_suggestion = np.zeros((n_peers,), dtype=np.int)

    judgement_matrix = []

    rel_agreement = 0
    rel_disagreement = 0

    error_case_agreement = 0
    error_case_disagreement = 0
    # offset the first row, which is the base borrower
    for lhs_anno_idx in range(len(annos)):
        for rhs_anno_idx in range(lhs_anno_idx + 1, len(annos)):

            lhs_analyst_id = annos[lhs_anno_idx]["analyst_id"]
            rhs_analyst_id = annos[rhs_anno_idx]["analyst_id"]

            if (
                lhs_analyst_id not in white_list_analysts
                or rhs_analyst_id not in white_list_analysts
            ):
                continue

            lhs_judgements = normalise_judgements(
                annos[lhs_anno_idx]["excel_df"], args.pos_threshold
            )
            rhs_judgements = normalise_judgements(
                annos[rhs_anno_idx]["excel_df"], args.pos_threshold
            )

            assert len(lhs_judgements) == len(rhs_judgements) == n_peers

            agreements_per_suggestion += lhs_judgements == rhs_judgements
            disagreements_per_suggestion += lhs_judgements != rhs_judgements

            lhs_rel_judgements = annos[lhs_anno_idx]["rel_ord_df"]
            rhs_rel_judgements = annos[rhs_anno_idx]["rel_ord_df"]

            rel_agreement += (lhs_rel_judgements == rhs_rel_judgements).sum()
            rel_disagreement += (lhs_rel_judgements != rhs_rel_judgements).sum()

            lhs_df: pd.DataFrame = annos[lhs_anno_idx]["excel_df"].dropna(
                subset=[REASON_1_COL_NAME]
            )
            rhs_df: pd.DataFrame = annos[rhs_anno_idx]["excel_df"].dropna(
                subset=[REASON_1_COL_NAME]
            )
            joined_df = lhs_df.join(rhs_df, how="inner", lsuffix="_lhs", rsuffix="_rhs")
            error_case_agreement += (
                joined_df[f"{REASON_1_COL_NAME}_lhs"]
                == joined_df[f"{REASON_1_COL_NAME}_rhs"]
            ).sum()
            error_case_disagreement += (
                joined_df[f"{REASON_1_COL_NAME}_lhs"]
                != joined_df[f"{REASON_1_COL_NAME}_rhs"]
            ).sum()

        if annos[lhs_anno_idx]["analyst_id"] in white_list_analysts:
            j = normalise_judgements(
                annos[lhs_anno_idx]["excel_df"], args.pos_threshold
            )
            judgement_matrix.append(j)
            relevance = annos[lhs_anno_idx]["excel_df"]["relevance_score"]
            relevance_counter.update(relevance)
            local_relevance_counter = Counter(relevance)
            sorted_relevances = sorted(local_relevance_counter.keys(), reverse=True)

            for idx, i in enumerate(sorted_relevances):
                lhs = local_relevance_counter[i]
                rhs = sum(
                    local_relevance_counter[j] for j in sorted_relevances[idx + 1 :]
                )
                n_training_pairs += lhs * rhs
            error_case_counter.update(
                annos[lhs_anno_idx]["excel_df"][REASON_1_COL_NAME].dropna()
            )

    # n_peer X n_companies
    judgement_matrix = np.vstack(judgement_matrix)
    majority_decisions = np.sum(judgement_matrix == 1, 0)
    stalemates = majority_decisions == (n_annotators / 2)
    majority_decisions = majority_decisions >= (n_annotators / 2)

    assert (
        len(majority_decisions)
        == len(disagreements_per_suggestion)
        == len(company_names)
        == len(stalemates)
    )

    for (
        company_name,
        agreement_level,
        disagreement_level,
        majority_decision,
        stalemate,
    ) in zip(
        company_names,
        agreements_per_suggestion,
        disagreements_per_suggestion,
        majority_decisions,
        stalemates,
    ):
        suggestions_with_disagreement_level.append(
            {
                "task_id": task_id,
                "suggested_company": company_name,
                "disagreement": disagreement_level,
                "agreement": agreement_level,
                "majority_decision": majority_decision,
                "stalemate": stalemate,
            }
        )

    return AgreementStats(
        agreement=agreements_per_suggestion.sum(),
        disagreement=disagreements_per_suggestion.sum(),
        suggestions_with_disagreement_level=suggestions_with_disagreement_level,
        rel_agreement=rel_agreement,
        rel_disagreement=rel_disagreement,
        relevance_counter=relevance_counter,
        error_case_counter=error_case_counter,
        n_training_pairs=n_training_pairs,
        error_case_agreement=error_case_agreement,
        error_case_disagreement=error_case_disagreement,
    )


def extract_task_analyst_id(f):
    _, tail = os.path.split(f)
    tail, _ = os.path.splitext(tail)
    tails = tail.split("_")
    return int(tails[2][1:]), "_".join(tails[3:])


def map_decision_to_rank(s: pd.Series):
    if s[POS_NEG_COL_NAME] == "Negative" or pd.isnull(s[POS_NEG_COL_NAME]):
        return 0
    relevance = s[REL_POS_COL_NAME]
    if relevance == "Least relevant":
        return 1
    if relevance == "Relevant":
        return 2
    if relevance == "Most relevant":
        return 3
    print(
        f"Warning: Suggestion {s.name} does not have relevance information. Tag it w/ Relevant"
    )
    return 2


def induce_rel_ordering(df: pd.DataFrame):
    relevance: np.ndarray = df["relevance_score"]
    assert relevance.ndim == 1
    return relevance[np.newaxis, :] >= relevance[:, np.newaxis]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        help="All the annotation files. One per base borrower. "
        "The file name should be named according to the following format "
        "(Peer_Annotation_B$BASE_BORROWER_IDX_$ANNOTATOR_IDX)",
        nargs="+",
    )
    parser.add_argument("--vp", nargs="+", type=str, default=[])
    parser.add_argument("--pos_threshold", type=int, default=0)
    parser.add_argument("--top_disagreed_k", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    annotation_by_task_id = defaultdict(list)

    all_analysts = set()

    for f in args.inputs:
        task_id, analyst_id = extract_task_analyst_id(f)
        df = handle_file(f)
        annotation_by_task_id[task_id].append(
            {"analyst_id": analyst_id, "excel_df": df}
        )
        all_analysts.add(analyst_id)

    # report precision @ k

    precision_at_ks = []
    aps = []
    ndcgs = []
    precision_at_Ks = []
    for task_id, annos in annotation_by_task_id.items():
        for anno in annos:
            df = anno["excel_df"].copy()
            ndcg = ndcg_df(df, "order", reverse=True)
            df["relevance_score"] = normalise_judgements(df, args.pos_threshold)
            ap = average_precision(df, "order", reverse=True)
            p_at_k = precision_at_k(df, "order", reverse=True)
            aps.append(ap)
            precision_at_ks.append(p_at_k)
            if not np.isnan(ndcg):
                ndcgs.append(ndcg)
            precision_at_Ks.append((p_at_k[-1], task_id, analyst_id))

    print(f"mean average precision: {np.average(aps):.4f}")
    print(f"NDCG: {np.average(ndcgs):.4f}")
    precision_at_ks_mat = np.zeros(
        (len(precision_at_ks), max(len(p) for p in precision_at_ks))
    )
    for i, p in enumerate(precision_at_ks):
        precision_at_ks_mat[i, : len(p)] = p
        precision_at_ks_mat[i, len(p) :] = p[-1]

    mean_precision_at_ks = np.average(precision_at_ks_mat, axis=0)
    var_precision_at_ks = np.std(precision_at_ks_mat, axis=0)
    for k, (mean_precision_at_k, var_precision_at_k) in enumerate(
        zip(mean_precision_at_ks, var_precision_at_ks), 1
    ):
        print(
            f"Precision @ {k}: {mean_precision_at_k:.4f} (± {var_precision_at_k:.4f})"
        )

    print("=" * 10 + "Worst annotation" + "=" * 10)
    precision_at_Ks = sorted(precision_at_Ks, key=lambda x: x[0])
    for p_at_k, task_id, analyst_id in precision_at_Ks:
        print(f"Task {task_id}, Analyst {analyst_id}: {p_at_k:.4f}")

    hist_path = os.path.join(args.output_dir, "p_at_k_hist.png")
    print("=" * 10 + f"Saving historgram graph to {hist_path}")

    plt.hist([k[0] for k in precision_at_Ks], 10, density=True)
    plt.savefig(hist_path)

    tab = []
    for cut_off in [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
        idx = int(math.ceil((1.0 - cut_off) * len(precision_at_Ks)))
        tab.append([cut_off, precision_at_Ks[idx][0]])
    print("=" * 10 + "Minimum Performance on x% of Queries" + "=" * 10)

    print(tabulate.tabulate(tab, headers=["Cut off (%)", "Min P@K"]))

    # annotators accidentally remove duplicated row. append a new one and claim it as
    fill_in_missing_annos(annotation_by_task_id)

    # populate relative ordering matrix. We can only do this once
    # 1. the missing annotations are filled
    # 2. the orderings of companies of dataframes are sorted alphabetically (so that all annotators dataframe will have
    # the exact same ordering)
    for annos in annotation_by_task_id.values():
        for anno in annos:
            anno["rel_ord_df"] = induce_rel_ordering(anno["excel_df"])

    vp_analysts = set(args.vp)

    other_analysts = all_analysts - vp_analysts

    print(f"All analysts id: {all_analysts}")
    print(f"VP analysts id: {vp_analysts}")
    print(f"Other analysts id: {other_analysts}")
    print("=" * 80)
    print("=" * 10 + "VPs:" + "=" * 10)
    vp_suggestion_details = report_stats(annotation_by_task_id, vp_analysts, args)
    vp_suggestion_details = vp_suggestion_details[~vp_suggestion_details["stalemate"]]
    print("=" * 80)
    print("=" * 10 + "Others:" + "=" * 10)
    report_stats(annotation_by_task_id, other_analysts, args)
    print("=" * 80)
    print("=" * 10 + "All:" + "=" * 10)
    report_stats(annotation_by_task_id, set(), args)
    print("=" * 80)
    print("=" * 10 + "Per-class consensus" + "=" * 10)

    pos_vp = vp_suggestion_details[vp_suggestion_details["majority_decision"]]
    neg_vp = vp_suggestion_details[~vp_suggestion_details["majority_decision"]]
    pos_agreement = pos_vp["agreement"].sum()
    pos_disagreement = pos_vp["disagreement"].sum()
    neg_agreement = neg_vp["agreement"].sum()
    neg_disagreement = neg_vp["disagreement"].sum()
    print("=" * 80)
    print(
        f"Pos agreement Level: {pos_agreement / (pos_agreement + pos_disagreement):.4f}"
    )
    print(
        f"Neg agreement Level: {neg_agreement / (neg_agreement + neg_disagreement):.4f}"
    )


def krippendorf_alpha(annotation_by_task_id, white_list_analysts: Set[int], args):
    all_analysts = set(
        s["analyst_id"] for annos in annotation_by_task_id.values() for s in annos
    )
    n_judgements = max(
        len(s["excel_df"].index)
        for annos in annotation_by_task_id.values()
        for s in annos
    )

    n_cases = len(annotation_by_task_id)
    if white_list_analysts:
        white_list_analysts = white_list_analysts & all_analysts
    else:
        white_list_analysts = all_analysts

    # we need to normalise analyst_ids so that they are zero_based (they can be mapped to a columnin np matrix)
    analyst_idx_by_id = {
        analyst_id: idx for idx, analyst_id in enumerate(white_list_analysts)
    }

    reliability_data = np.full(
        (len(white_list_analysts), n_cases, n_judgements), fill_value=np.nan
    )

    for idx, (task_id, annos) in enumerate(annotation_by_task_id.items()):
        for anno in annos:
            # filtered by whitelist
            if anno["analyst_id"] not in analyst_idx_by_id:
                continue
            analyst_idx = analyst_idx_by_id[anno["analyst_id"]]
            n_local_judgements = len(anno["excel_df"].index)
            reliability_data[analyst_idx, idx, :n_local_judgements] = anno["excel_df"][
                "relevance_score"
            ]

    return krippendorff.alpha(
        reliability_data.reshape((len(white_list_analysts), -1)),
        level_of_measurement="ordinal",
    )


def report_stats(annotation_by_task_id, analysts: Set[int], args):
    agreement_ratio_by_task_id = {}
    acc_aggreement = 0
    acc_disagreement = 0
    acc_suggestion_details = []
    acc_rel_agreement = 0
    acc_rel_disagreement = 0

    acc_total_judgements = 0
    acc_total_qs = 0
    acc_relevance_counter = Counter()
    acc_error_counter = Counter()
    acc_n_training_pairs = 0
    acc_error_case_agreement = 0
    acc_error_case_disagreement = 0
    for task_id, annotations in annotation_by_task_id.items():
        agreement, disagreement, suggestions_disagreement_level, rel_agreement, rel_disagreement, relevance_counter, error_case_counter, n_training_pairs, error_case_agreement, error_case_disagreement = agreement_per_task(
            task_id, annotations, analysts, args
        )
        acc_total_judgements += sum(len(a["excel_df"].index) for a in annotations)
        acc_total_qs += len(annotations[0]["excel_df"].index)
        acc_aggreement += agreement
        acc_disagreement += disagreement
        acc_rel_agreement += rel_agreement
        acc_rel_disagreement += rel_disagreement
        acc_suggestion_details.extend(suggestions_disagreement_level)
        acc_relevance_counter += relevance_counter
        acc_error_counter += error_case_counter
        acc_n_training_pairs += n_training_pairs
        acc_error_case_agreement += error_case_agreement
        acc_error_case_disagreement += error_case_disagreement

        if agreement + disagreement > 0:
            agreement_ratio_by_task_id[task_id] = agreement / (agreement + disagreement)
    print(f"Total Questions: {acc_total_qs}. Total Judgements: {acc_total_judgements}")
    print(f"Total agreement: {acc_aggreement}. Total disagreement: {acc_disagreement}")
    print(
        f"Inter-annotator agreement: {np.divide(acc_aggreement, (acc_aggreement + acc_disagreement)):.4f}"
    )
    print("=" * 10 + "Rel ordering agreement" + "=" * 10)
    print(
        f"Total relative agreement: {acc_rel_agreement}. Total relative disagreement: {acc_rel_disagreement}"
    )
    print(
        f"Inter-annotator relative order agreement: "
        f"{np.divide(acc_rel_agreement, acc_rel_agreement + acc_rel_disagreement)}"
    )
    print(
        f"Error case agreement: "
        f"{np.divide(acc_error_case_agreement, acc_error_case_agreement + acc_error_case_disagreement)}"
    )
    print(
        "=" * 10
        + f"Krippendorff Alpha: {krippendorf_alpha(annotation_by_task_id, analysts, args)}"
        + "=" * 10
    )
    agreement_variance = np.std(list(agreement_ratio_by_task_id.values()))
    print(
        f"Std-dev of Inter-annotator agreement across different base borrowers cases: {agreement_variance:.4f}"
    )
    print("=" * 10 + "Relevance Stats" + "=" * 10)
    for k, v in acc_relevance_counter.items():
        print(f"{k}: {v}")
    print("=" * 10 + "Error Case Stats" + "=" * 10)
    n_errors = sum(acc_error_counter.values())
    tab = []
    for k, v in sorted(acc_error_counter.items(), key=lambda x: x[0]):
        tab.append([k, v / n_errors, v])

    print(tabulate.tabulate(tab, headers=["Error", "Percentage", "Counts"]))

    print("=" * 10 + f"Number of training pairs: {acc_n_training_pairs}" + "=" * 10)
    print("=" * 10 + "Most disagreed suggested companies" + "=" * 10)
    acc_suggestion_details = pd.DataFrame(acc_suggestion_details)
    for _, s in (
        acc_suggestion_details.sort_values(by=["disagreement"], ascending=False)
        .head(args.top_disagreed_k)
        .iterrows()
    ):
        print(
            f"Task: {s['task_id']}, Suggested Company: {s['suggested_company']}, Disagreement: {s['disagreement']}"
        )

    return acc_suggestion_details


if __name__ == "__main__":
    main()
