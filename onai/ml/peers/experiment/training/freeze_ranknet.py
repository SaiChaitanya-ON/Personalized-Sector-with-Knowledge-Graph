import argparse
import os
import shutil
import tarfile
import tempfile

import torch
from smart_open import open
from transformers import BertConfig, BertTokenizer

from onai.ml.peers.experiment.modeling.base import Scorer, ScorerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="Directory of the model snapshot "
        "(which contains scorer_cfg.json and best_model.pth)",
        required=True,
    )
    parser.add_argument(
        "-analyser", help="Location of analyser (idf_model.pkl)", required=True
    )
    parser.add_argument("-o", help="Output directory")

    args = parser.parse_args()

    scorer_cfg = ScorerConfig.from_json_file(os.path.join(args.i, "scorer_cfg.json"))

    with open(os.path.join(args.o, "scorer_cfg.json"), "w") as fout:
        fout.write(scorer_cfg.to_json_string())

    # Validating the Scorer's integrity with scorer_cfg and uploading the artifact
    with open(os.path.join(args.i, "best_model.pth"), "rb") as f:
        model_state = torch.load(f, map_location="cpu")

    scorer = Scorer(scorer_cfg)
    scorer.load_state_dict(model_state)  # checking integrity

    with open(os.path.join(args.o, "best_model.pth"), "wb") as fout:
        torch.save(scorer.state_dict(), fout)

    """copying the Analyser to the output path"""
    with open(os.path.join(args.analyser), "rb") as fin, open(
        os.path.join(args.o, "idf_model.pkl"), "wb"
    ) as fout:
        shutil.copyfileobj(fin, fout)


if __name__ == "__main__":
    main()
