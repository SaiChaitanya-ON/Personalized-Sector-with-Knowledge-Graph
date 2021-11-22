import argparse
import os
import shutil
import tarfile
import tempfile

import torch
from smart_open import open
from transformers import BertConfig, BertTokenizer

from onai.ml.peers.experiment.modeling.bert import BertScorer, BertScorerConfig


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

    scorer_cfg = BertScorerConfig.from_json_file(
        os.path.join(args.i, "scorer_cfg.json")
    )
    scorer_cfg.pretrained_last_layer_path = None
    scorer_cfg.pretrained_bert_name = (
        scorer_cfg.pretrained_bert_name
        if scorer_cfg.pretrained_bert_name
        else "bert-base-uncased"
    )

    # we assume the pretrained bert name must be a huggingface recognisable alias of a model
    # that means we must be able to get the config by this name as well
    bert_cfg = BertConfig.from_pretrained(scorer_cfg.pretrained_bert_name)
    bert_cfg_p = os.path.join(args.o, "bert_cfg.json")
    with open(bert_cfg_p, "w") as fout:
        fout.write(bert_cfg.to_json_string())

    # freeze bert tokenizer
    bert_tokeniser_p = os.path.join(args.o, "berttokeniser.tar.gz")
    with tempfile.TemporaryDirectory() as temp_dir, open(
        bert_tokeniser_p, "wb"
    ) as fout, tarfile.open(fileobj=fout, mode="w:gz") as tar_out:
        BertTokenizer.from_pretrained(scorer_cfg.pretrained_bert_name).save_pretrained(
            temp_dir
        )
        tar_out.add(temp_dir, "")

    # enforce the loader in candidatesuggestion not to load the pretrained model
    scorer_cfg.pretrained_bert_name = None
    # pretrained_bert_cfg_path will be set during inference to enable relocation of the artifacts
    scorer_cfg.pretrained_bert_cfg_path = None

    with open(os.path.join(args.o, "scorer_cfg.json"), "w") as fout:
        fout.write(scorer_cfg.to_json_string())

    # Validating the Scorer's integrity with scorer_cfg and uploading the artifact
    with open(os.path.join(args.i, "best_model.pth"), "rb") as f:
        model_state = torch.load(f, map_location="cpu")

    scorer = BertScorer(scorer_cfg)
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
