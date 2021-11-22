import argparse
import logging
import os

import pandas as pd

from onai.ml.tools.logging import setup_logger

logger = logging.getLogger(__name__)


def compare(li1, li2, args):
    count = 0
    for i in range(args.p):
        if li1[i] == li2[i]:
            count += 1
    return count


def get_res(path, i, args):
    assert i.endswith(".xlsx")
    df_temp = pd.read_excel(os.path.join(path, i), header=1)
    li = list(df_temp["Positive/Negative?"][1:])
    assert len(li) == args.p
    return li


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_golden", help="The path to the Golden records", required=True
    )
    parser.add_argument(
        "--participants", help="The path to participants records", required=True
    )
    parser.add_argument("--out", help="results will be saved here", required=False)
    parser.add_argument("--p", help="samples required.", default=20, type=int)
    args = parser.parse_args()

    golden_dict = {}

    for r, d, f in os.walk(args.path_golden):
        for i in f:
            golden_dict[i] = get_res(r, i, args)

    participants_list = os.listdir(args.participants)

    for name in participants_list:
        logger.info("Evaluating %s's annotations", name)
        max_score = 0
        pos_score = 0
        dir_path = os.path.join(args.participants, name)
        for file in [f for r, d, f in os.walk(dir_path)][0]:
            if file not in golden_dict.keys():
                logger.info("file %s not found in golden records", file)
            score = compare(golden_dict[file], get_res(dir_path, file, args), args)
            max_score += args.p
            pos_score += score
            logger.info("%d out of %d match for file\n %s", score, args.p, file)
        logger.info(
            "Overall Accuracy of %s is %.2f %% \n", name, (pos_score / max_score * 100)
        )


if __name__ == "__main__":
    main()
