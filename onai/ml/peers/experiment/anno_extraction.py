import argparse
from collections import defaultdict

import pandas as pd
import smart_open

from onai.ml.peers.experiment.evaluate_consensus import (
    POS_NEG_COL_NAME,
    REASON_1_COL_NAME,
    REL_POS_COL_NAME,
    extract_task_analyst_id,
    handle_file,
    map_decision_to_rank,
)


def map_is_duplicate(s):
    return (
        isinstance(s[REASON_1_COL_NAME], str)
        and "duplicate" in s[REASON_1_COL_NAME].lower()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        help="internal database about details on the base borrowers",
        required=True,
    )
    parser.add_argument(
        "-a",
        help="paths of annotation files. "
        "The file name follows the pattern of "
        "Peer_Annotation_B$TASK_ID_$ANNOTATOR_ID",
        nargs="+",
    )
    parser.add_argument("-o", help="The path to the output annotation")
    args = parser.parse_args()

    base_borrowers = pd.read_excel(args.b)
    taskid2borrower = {
        row["Case"]: row["Base Borrower Name"] for _, row in base_borrowers.iterrows()
    }

    is_base_id = True if "Base Borrower Id" in base_borrowers.columns else False
    if is_base_id:
        taskid2borrowerid = {
            row["Case"]: row["Base Borrower Id"] for _, row in base_borrowers.iterrows()
        }

    anno_paths = args.a

    annotation_by_task_id = defaultdict(lambda: {"annotations": list()})

    for f in anno_paths:
        input_p = f
        task_id, analyst_id = extract_task_analyst_id(input_p)
        df = handle_file(input_p, get_end_year=True)
        df = df.reset_index()

        df["analyst_id"] = analyst_id

        base_borrower_name = taskid2borrower[task_id]

        annotation_by_task_id[task_id]["base_borrower_name"] = base_borrower_name
        annotation_by_task_id[task_id]["annotations"].append(
            {
                "analyst_id": analyst_id,
                "base_borrower_name": base_borrower_name,
                "excel_df": df,
            }
        )

    excel_dfs = []
    for task_id, metainfo in annotation_by_task_id.items():
        base_borrower_name = metainfo["base_borrower_name"]
        all_annos = pd.concat(
            el["excel_df"][
                [
                    "Company",
                    POS_NEG_COL_NAME,
                    REL_POS_COL_NAME,
                    REASON_1_COL_NAME,
                    "analyst_id",
                    "peer_entity_id",
                    "end_yr",
                ]
            ]
            for el in metainfo["annotations"]
        )
        if is_base_id:
            all_annos["base_id"] = taskid2borrowerid[task_id]
        all_annos["is_duplicate"] = all_annos.apply(map_is_duplicate, axis=1)
        all_annos["relevance_score"] = all_annos.apply(map_decision_to_rank, axis=1)
        all_annos["base_name"] = base_borrower_name
        all_annos["task_id"] = task_id
        all_annos = all_annos[~all_annos["is_duplicate"]]
        all_annos = all_annos.rename(columns={"Company": "peer_name"})
        excel_dfs.append(all_annos)

    with smart_open.open(args.o, "w") as f:
        pd.concat(excel_dfs).to_csv(f, index=False)


if __name__ == "__main__":
    main()
