"""
Burges, Chris, et al. "Learning to rank using gradient descent."
Proceedings of the 22nd international conference on Machine learning. 2005.
"""


import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import smart_open
import tabulate
import torch
from torch import nn, optim

from onai.ml.peers.experiment.modeling.base import RankNet, ScorerConfig
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
    parser.add_argument("--test", help="test pickle", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Epoch", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)

    parser.add_argument("--scorer_cfg", help="Config file path of scorer")

    g = parser.add_argument_group("scorer config", description="Override scorer config")

    ScorerConfig.populate_argparser(g)

    parser.add_argument(
        "--model_path",
        help="Loading a pre-trained ranknet, "
        "instead of starting from a randomised model",
    )


def create_rank_net(args, parser):
    if args.scorer_cfg:
        with smart_open.open(args.scorer_cfg, "r") as fin:
            scorer_cfg = json.loads(fin.read())
    else:
        scorer_cfg = {}
    assert parser._action_groups[2].title == "scorer config"
    scorer_cfg.update(extract_subgroup_args(args, parser._action_groups[2]))
    if "cols" not in scorer_cfg:
        scorer_cfg["cols"] = COLS + ["bert_embedding_sim"]
    scorer_cfg = ScorerConfig.from_dict(scorer_cfg)
    with smart_open.open(os.path.join(args.output_dir, "scorer_cfg.json"), "w") as fout:
        fout.write(scorer_cfg.to_json_string())
    return RankNet(scorer_cfg)


def train_net(X, Y, net, args):
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = args.batch_size
    l1_reg = args.l1
    l2_reg = args.l2
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        for i in range(0, X.shape[0], batch_size):
            optimizer.zero_grad()
            X_minibatch = X[i : i + batch_size]
            Y_minibatch = Y[i : i + batch_size]

            output = net(X_minibatch)
            loss = criterion(output, Y_minibatch)

            reg_loss_l1 = 0
            reg_loss_l2 = 0
            for param in net.parameters():
                reg_loss_l1 += param.norm(1)
                reg_loss_l2 += param.norm(2) ** 2

            loss += l1_reg * reg_loss_l1 + l2_reg * reg_loss_l2
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            output = net(X)
            loss = criterion(output, Y)
            logger.info(loss)

    return net


def main(args=None):
    _clean_hdlrs()
    setup_logger()
    parser = argparse.ArgumentParser()

    populate_parser(parser)
    if args is None:
        args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    log_fout = smart_open.open(os.path.join(args.output_dir, "log"), "w")
    ch = logging.StreamHandler(log_fout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)
    with smart_open.open(args.train, "rb") as f:
        train_df = pd.read_pickle(f, None)

    ranknet = create_rank_net(args, parser)

    if args.model_path:
        with smart_open.open(args.model_path, "rb") as fin:
            ranknet.scorer.load_state_dict(torch.load(fin))

    cols = ranknet.scorer.cols

    X_train, Y_train, _ = create_pairwise_set(train_df, cols)
    train_net(X_train, Y_train, ranknet, args)

    scorer = ranknet.scorer

    # saving the ranknet
    with smart_open.open(os.path.join(args.output_dir, "best_model.pth"), "wb") as fout:
        logger.info("Model is the best baseline. Saving it.")
        torch.save(scorer.state_dict(), fout)
    # save the last layer parameter in the form of json
    weights = scorer.fc.weight.detach().numpy()[0]
    weight_by_col_name = {
        feature: float(weight) for weight, feature in zip(weights, cols)
    }

    with smart_open.open(
        os.path.join(args.output_dir, "last_layer_parm.json"), "w"
    ) as fout:
        json.dump(weight_by_col_name, fout)

    dim = X_train.shape[1] // 2

    scales = np.std((X_train[:, :dim] - X_train[:, dim:]).detach().numpy(), axis=0)

    weights = ranknet.scorer.fc.weight.detach().numpy()[0] * scales

    logger.info(
        "Most important feature: \n {}".format(
            tabulate.tabulate(
                [
                    (feature, weight)
                    for weight, feature in sorted(
                        zip(weights, cols), key=lambda x: -abs(x[0])
                    )
                ],
                headers=["Feature", "Weight"],
            )
        )
    )

    train_df["predicted_f"] = scorer.predict_from_df(train_df)

    map_train, ndcg_train = performance_stats(train_df, "predicted_f")

    with smart_open.open(args.test, "rb") as f:
        test_df = pd.read_pickle(f, None)

    test_df["predicted_f"] = scorer.predict_from_df(test_df)
    map_test, ndcg_test = performance_stats(test_df, "predicted_f")

    logger.info(
        "Performance Stats: \n {}".format(
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
                headers=[
                    "Baseline",
                    "MAP Train",
                    "NDCG Train",
                    "MAP Test",
                    "NDCG Test",
                ],
            )
        )
    )
    logger.info(
        "Out of order pairs @ Train: \n {}".format(
            error_analysis(train_df, "predicted_f", False)
        )
    )
    logger.info(
        "Original Out of order pairs @ Train: \n {}".format(error_analysis(train_df))
    )

    logger.info(
        "Out of order pairs @ Test: \n {}".format(
            error_analysis(test_df, "predicted_f", False)
        )
    )
    logger.info(
        "Original Out of order pairs @ Test: \n {}".format(error_analysis(test_df))
    )
    return map_test.mean(), ndcg_test.mean()


if __name__ == "__main__":
    main()
