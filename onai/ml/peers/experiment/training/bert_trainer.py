import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import smart_open
import tabulate
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader, Dataset

from onai.ml.peers.experiment.modeling.bert import (
    BertRankNet,
    BertRankNetConfig,
    BertScorer,
    BertScorerConfig,
)
from onai.ml.peers.experiment.training.albert_trainer import (
    create_pairwise_set,
    error_analysis,
    performance_stats,
)
from onai.ml.peers.experiment.training.default_feats import COLS
from onai.ml.tools.argparse import extract_subgroup_args
from onai.ml.tools.logging import _clean_hdlrs, _formatter, setup_logger

logger = logging.getLogger(__name__)


def populate_parser(parser):
    parser.add_argument("--train", help="train pickle", required=True)
    parser.add_argument("--val", help="val pickle", required=True)
    parser.add_argument("--test", help="test pickle", required=True)
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--epochs", help="Epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=os.cpu_count())

    parser.add_argument("--scorer_cfg", help="Config file path of scorer")
    parser.add_argument("--ranknet_cfg", help="Config file path of ranknet")

    parser.add_argument(
        "--local_ckpt",
        help="Local path to store temporary ckpt",
        default=os.path.join(os.curdir, "checkpoints"),
    )
    parser.add_argument(
        "--dev",
        default=False,
        help="Run Trainer in fast_dev_run mode ?",
        dest="dev",
        action="store_true",
    )

    g = parser.add_argument_group("scorer config", description="Override scorer config")

    BertScorerConfig.populate_argparser(g)

    g = parser.add_argument_group(
        "ranknet config", description="Override RankNet Config"
    )

    BertRankNetConfig.populate_argparser(g)

    parser.add_argument(
        "--model_path",
        help="Loading a pre-trained ranknet, "
        "instead of starting from a randomised model",
    )


def create_scorer(args, parser):
    if args.scorer_cfg:
        with smart_open.open(args.scorer_cfg, "r") as fin:
            scorer_cfg = json.loads(fin.read())
    else:
        scorer_cfg = {}

    assert parser._action_groups[2].title == "scorer config"
    scorer_cfg.update(extract_subgroup_args(args, parser._action_groups[2]))

    if "cols" not in scorer_cfg:
        scorer_cfg["cols"] = COLS

    scorer_cfg = BertScorerConfig.from_dict(scorer_cfg)

    with smart_open.open(os.path.join(args.output_dir, "scorer_cfg.json"), "w") as fout:
        fout.write(scorer_cfg.to_json_string())

    return BertScorer(scorer_cfg)


def create_rank_net(args, parser, scorer: BertScorer):
    if args.ranknet_cfg:
        with smart_open.open(args.ranknet_cfg, "r") as fin:
            ranknet_cfg = json.loads(fin.read())
    else:
        ranknet_cfg = {}
    assert parser._action_groups[3].title == "ranknet config"
    ranknet_cfg.update(extract_subgroup_args(args, parser._action_groups[3]))
    ranknet_cfg = BertRankNetConfig.from_dict(ranknet_cfg)
    with smart_open.open(
        os.path.join(args.output_dir, "ranknet_cfg.json"), "w"
    ) as fout:
        fout.write(ranknet_cfg.to_json_string())
    return RankNet(cfg=ranknet_cfg, scorer=scorer, hparams=args)


class RankNet(BertRankNet):
    def __init__(self, cfg: BertRankNetConfig, scorer: BertScorer, hparams):
        super(RankNet, self).__init__(cfg, scorer)
        self.hparams = hparams
        self.best_map_val = 0

    def prepare_data(self):
        logger.info("loading training and validation Data..")
        with smart_open.open(self.hparams.train, "rb") as fin:
            self.x_train = pd.read_pickle(fin, None)

        with smart_open.open(self.hparams.val, "rb") as fin:
            self.x_val = pd.read_pickle(fin, None)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def train_dataloader(self):
        x_train_loader = DataLoader(
            RankerDataset(self.x_train, self.scorer.cols),
            batch_size=self.hparams.batch_size,
            collate_fn=collate_batch,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.cpus,
        )
        return x_train_loader

    def val_dataloader(self):
        x_val_loader = DataLoader(
            RankerDataset(self.x_val, self.scorer.cols),
            collate_fn=collate_batch,
            pin_memory=True,
            num_workers=self.hparams.cpus,
        )
        return x_val_loader

    def training_step(self, batch, batch_idx):
        x, y, d_base, dx, dy = (
            batch["x"],
            batch["y"],
            batch["base_token_ids"],
            batch["peer_token_ids_x"],
            batch["peer_token_ids_y"],
        )

        dim = len(self.scorer.config.cols)
        res = self(x[:, :dim], x[:, dim:], y, d_base, dx, dy)
        loss = res["loss"]

        return {"loss": loss, "log": {"training_loss": loss}}

    def training_epoch_end(self, training_output):
        avg_loss = torch.stack([out["loss"] for out in training_output]).mean()

        self.x_train["predicted_f"] = self.scorer.predict_from_df(self.x_train)
        map_train, ndcg_train = performance_stats(self.x_train, "predicted_f")
        return {
            "log": {
                "avg_epoch_loss": avg_loss,
                "map_train": map_train.mean(),
                "ndcg_train": ndcg_train.mean(),
            }
        }

    def validation_step(self, batch, batch_idx):
        x, y, d_base, dx, dy = (
            batch["x"],
            batch["y"],
            batch["base_token_ids"],
            batch["peer_token_ids_x"],
            batch["peer_token_ids_y"],
        )
        dim = len(self.scorer.config.cols)
        res = self(x[:, :dim], x[:, dim:], y, d_base, dx, dy)
        val_loss = res["loss"]
        return {"val_loss": val_loss, "log": {"val_loss": val_loss}}

    def validation_epoch_end(self, val_output):
        avg_val_loss = torch.stack([out["val_loss"] for out in val_output]).mean()

        self.x_val["predicted_f"] = self.scorer.predict_from_df(self.x_val)
        map_val, ndcg_val = performance_stats(self.x_val, "predicted_f")
        logs = {
            "avg_val_loss": avg_val_loss,
            "map_val": torch.tensor(map_val.mean()),
            "ndcg_val": torch.tensor(ndcg_val.mean()),
        }
        if logs["map_val"] > self.best_map_val:
            self.best_map_val = logs["map_val"]
        return {"map_val": logs["map_val"], "val_loss": avg_val_loss, "log": logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        return optimizer


class RankerDataset(Dataset):
    def __init__(self, x_df: pd.DataFrame, cols: list):
        self.x_seq, self.y_seq, self.x_descs = create_pairwise_set(x_df, cols)
        self.base_token_ids = [d["base_token_ids"] for d in self.x_descs]
        self.peer_token_ids_x = [d["peer_token_ids_x"] for d in self.x_descs]
        self.peer_token_ids_y = [d["peer_token_ids_y"] for d in self.x_descs]

    def __len__(self):
        return self.x_seq.shape[0]

    def __getitem__(self, item):

        return {
            "x": self.x_seq[item],
            "y": self.y_seq[item],
            "base_token_ids": self.base_token_ids[item],
            "peer_token_ids_x": self.peer_token_ids_x[item],
            "peer_token_ids_y": self.peer_token_ids_y[item],
        }


def collate_batch(batch):
    size = len(batch)
    return {
        "x": torch.stack([batch[i]["x"] for i in range(size)]),
        "y": torch.stack([batch[i]["y"] for i in range(size)]),
        "base_token_ids": [batch[i]["base_token_ids"] for i in range(size)],
        "peer_token_ids_x": [batch[i]["peer_token_ids_x"] for i in range(size)],
        "peer_token_ids_y": [batch[i]["peer_token_ids_y"] for i in range(size)],
    }


def estimate_feature_scales(net: BertRankNet, x: pd.DataFrame, batch_size: int):
    original_mode = net.training
    net.eval()
    x, y, x_descs = create_pairwise_set(x, net.scorer.cols)
    # for all other features, just use the
    # stats on the first half minus second half of the vectors
    dim = len(net.scorer.config.cols)
    scalar_scales = np.std((x[:, :dim] - x[:, dim:]).detach().numpy(), axis=0)
    # sample 300 pairs and get the mean and variance estimate
    bert_emb_dot_prod_diffs = []
    with torch.no_grad():
        for i in range(0, min(x.shape[0], 300), batch_size):
            x_minibatch = x[i : i + batch_size]
            y_minibatch = y[i : i + batch_size]
            x_descs_minibatch = x_descs[i : i + batch_size]
            ret = net(
                peer1_scalar_feats=x_minibatch[:, :dim],
                peer2_scalar_feats=x_minibatch[:, dim:],
                y=y_minibatch,
                base_desc_token_ids=[d["base_token_ids"] for d in x_descs_minibatch],
                peer1_desc_token_ids=[d["peer_token_ids_x"] for d in x_descs_minibatch],
                peer2_desc_token_ids=[d["peer_token_ids_y"] for d in x_descs_minibatch],
            )
            peer1_feat, peer2_feat = ret["peer1_feat"], ret["peer2_feat"]

            bert_emb_dot_prod_diffs.append(
                (peer1_feat[:, -1] - peer2_feat[:, -1]).numpy()
            )
    bert_emb_dot_prod_diffs = np.concatenate(bert_emb_dot_prod_diffs, axis=0)
    net.train(original_mode)

    return np.concatenate(
        [scalar_scales, np.std(bert_emb_dot_prod_diffs, keepdims=True)], axis=0
    )


def main(args=None):
    _clean_hdlrs()
    setup_logger()

    parser = argparse.ArgumentParser()
    populate_parser(parser)
    if args is None:
        args = parser.parse_args()

    log_fout = smart_open.open(os.path.join(args.local_ckpt, "log"), "w")
    ch = logging.StreamHandler(log_fout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)
    logger.info(args)

    ranknet = create_rank_net(args, parser, create_scorer(args, parser))

    assert ranknet.scorer.config.pretrained_bert_name or (
        ranknet.scorer.config.pretrained_bert_cfg_path and args.model_path
    ), "bert model parametres need to be initialized"

    if args.model_path:
        with smart_open.open(args.model_path, "rb") as fin:
            ranknet.load_state_dict(
                torch.load(fin, map_location=torch.device("cpu"))["state_dict"]
            )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor="map_val",
        mode="max",
        verbose=1,
        filepath=os.path.join(args.local_ckpt, "{epoch:02d}-{map_val:.2f}-{loss:.2f}"),
    )

    trainer_param = {
        "fast_dev_run": args.dev,
        "max_epochs": args.epochs,
        "checkpoint_callback": checkpoint_callback,
        "default_root_dir": args.local_ckpt,
        "gpus": args.gpus,
    }

    if os.path.exists(os.path.join(args.local_ckpt, "last.ckpt")):
        trainer_param["resume_from_checkpoint"] = os.path.join(
            args.local_ckpt, "last.ckpt"
        )
        logger.info(
            "loaded state from Checkpoint %s", trainer_param["resume_from_checkpoint"]
        )

    trainer = pl.Trainer(**trainer_param)
    logger.info("Starting Training..")
    trainer.fit(ranknet)

    # loading the best epoch
    if args.epochs > 0 and checkpoint_callback.best_model_path:
        try:
            logger.info(
                "Post Training: loading from %s", checkpoint_callback.best_model_path
            )
            ckpt = torch.load(checkpoint_callback.best_model_path)
            ranknet.load_state_dict(ckpt["state_dict"])
            if args.output_dir:
                with smart_open.open(
                    os.path.join(args.output_dir, "best_model.pth"), "wb"
                ) as fout:
                    torch.save(ranknet.scorer.state_dict(), fout)
        except Exception:
            logger.info(
                "not able to load from %s using the last ranker ",
                checkpoint_callback.best_model_path,
            )

    scales = estimate_feature_scales(ranknet, ranknet.x_train, args.batch_size * 2)

    scorer = ranknet.scorer

    weights = scorer.fc.weight.cpu().detach().numpy()[0] * scales

    tab_data = [
        (feature, weight)
        for weight, feature in sorted(
            zip(weights, scorer.cols + ["Fine Tuned Bert"]), key=lambda x: -abs(x[0])
        )
    ]
    logger.info("Most important feature: \n")
    logger.info(tabulate.tabulate(tab_data, headers=["Feature", "Weight"]))

    ranknet.x_train["predicted_f"] = scorer.predict_from_df(ranknet.x_train)

    map_train, ndcg_train = performance_stats(ranknet.x_train, "predicted_f")

    with smart_open.open(args.test, "rb") as fin:
        x_test = pd.read_pickle(fin, None)
    x_test["predicted_f"] = scorer.predict_from_df(x_test)
    map_test, ndcg_test = performance_stats(x_test, "predicted_f")
    logger.info("Performance Stats: \n")
    logger.info(
        tabulate.tabulate(
            [
                (
                    "LTR",
                    map_train.mean(),
                    ndcg_train.mean(),
                    map_test.mean(),
                    ndcg_test.mean(),
                )
            ],
            headers=["Baseline", "MAP Train", "NDCG Train", "MAP Test", "NDCG Test"],
        )
    )

    logger.info("Out of order pairs @ Train: \n")
    logger.info(error_analysis(ranknet.x_train, "predicted_f", False))

    logger.info("Original Out of order pairs @ Train: \n ")
    logger.info(error_analysis(ranknet.x_train))

    logger.info("Out of order pairs @ Test: \n")
    logger.info(error_analysis(ranknet.x_val, "predicted_f", False))

    logger.info("Original Out of order pairs @ Test: \n")

    logger.info(error_analysis(ranknet.x_val))

    if os.path.exists(os.path.join(args.local_ckpt, "last.ckpt")):
        os.rename(
            os.path.join(args.local_ckpt, "last.ckpt"),
            os.path.join(args.local_ckpt, f"last_epoch_{ranknet.current_epoch}.ckpt"),
        )

    return map_test.mean(), ndcg_test.mean()


if __name__ == "__main__":
    main()
