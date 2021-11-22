import argparse
import os

import torch
from omegaconf import OmegaConf
from smart_open import open

from onai.ml.peers.experiment.training.surrogate import SurrogateDist
from onai.ml.tools.logging import setup_logger


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("-o", help="Output Directory", required=True)
    args = parser.parse_args()

    with open(args.model_cfg, "r") as fin:
        cfg = OmegaConf.load(fin)
    with open(os.path.join(args.o, "scorer_cfg.yaml"), "w") as fout:
        OmegaConf.save(cfg["m"]["repr"], fout)
    with open(args.model_ckpt, "rb") as fin:
        net = SurrogateDist(cfg["m"])
        net.load_state_dict(
            torch.load(fin, map_location=torch.device("cpu"))["state_dict"]
        )
    with open(os.path.join(args.o, "best_model.pth"), "wb") as fout:
        torch.save(net.repr.state_dict(), fout)


if __name__ == "__main__":
    main()
