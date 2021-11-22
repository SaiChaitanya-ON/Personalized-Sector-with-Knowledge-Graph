import argparse
import json
import logging
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import smart_open
import tabulate
import torch
from torch import nn, optim

from onai.ml.peers.experiment.evaluate_consensus import REASON_1_COL_NAME
from onai.ml.peers.experiment.modeling.albert import (
    AlbertRankNet,
    AlbertRankNetConfig,
    AlbertScorer,
    AlbertScorerConfig,
)
from onai.ml.peers.feature_extractor import _LAST_REVENUE_WINDOWS
from onai.ml.peers.metric import average_precision, ndcg_df
from onai.ml.tools.argparse import add_bool_argument, extract_subgroup_args
from onai.ml.tools.logging import _clean_hdlrs, _formatter, setup_logger

logger = logging.getLogger(__name__)

_COLS = [
    "predicted_industries_overlap",
    "country_overlap",
    "weighted_symmetric_diff",
    "weighted_intersection",
    "weighted_intersection_negative_sample",
    "weighted_intersection_negative_sample_tail_end",
    "peer_diff",
    "is_subsidiary",
    "no_last_revenue_diff",
    "no_last_ebitda_diff",
    "no_last_ebit_diff",
]

for window_size in _LAST_REVENUE_WINDOWS:
    _COLS.append(f"last_revenue_diff_{window_size:.2f}")
    _COLS.append(f"last_ebitda_diff_{window_size:.2f}")
    _COLS.append(f"last_ebit_diff_{window_size:.2f}")


def map_get_label(s):
    return (
        0.5
        if s["relevance_score_x"] == s["relevance_score_y"]
        else float(s["relevance_score_x"] > s["relevance_score_y"])
    )


def create_pairwise_set(X, cols):
    joined_cols = [el + "_x" for el in cols] + [el + "_y" for el in cols]

    joined_df = X.merge(X, on=["task_id", "analyst_id"])
    joined_df = joined_df[
        (joined_df["relevance_score_x"] > joined_df["relevance_score_y"])
    ]
    joined_df["Y"] = joined_df.apply(map_get_label, axis=1)
    joined_df = joined_df.reset_index(drop=True)

    X = joined_df[joined_cols].to_numpy()
    Y = joined_df["Y"].to_numpy()
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y[:, np.newaxis]).float()
    X_descs = [
        {
            "base_token_ids": row[
                "base_token_ids_x"
            ],  # does not matter which one to choose
            "peer_token_ids_x": row["peer_token_ids_x"],
            "peer_token_ids_y": row["peer_token_ids_y"],
        }
        for _, row in joined_df.iterrows()
    ]
    return X, Y, X_descs


def create_scorer(args, parser):
    if args.scorer_cfg:
        with smart_open.open(args.scorer_cfg, "r") as fin:
            scorer_cfg = json.loads(fin.read())
    else:
        scorer_cfg = {}

    assert parser._action_groups[2].title == "scorer config"
    scorer_cfg.update(extract_subgroup_args(args, parser._action_groups[2]))

    if "cols" not in scorer_cfg:
        scorer_cfg["cols"] = _COLS

    scorer_cfg = AlbertScorerConfig.from_dict(scorer_cfg)

    with smart_open.open(os.path.join(args.output_dir, "scorer_cfg.json"), "w") as fout:
        fout.write(scorer_cfg.to_json_string())

    return AlbertScorer(scorer_cfg)


def create_rank_net(args, parser, scorer: AlbertScorer):
    if args.ranknet_cfg:
        with smart_open.open(args.ranknet_cfg, "r") as fin:
            ranknet_cfg = json.loads(fin.read())
    else:
        ranknet_cfg = {}
    assert parser._action_groups[3].title == "ranknet config"
    ranknet_cfg.update(extract_subgroup_args(args, parser._action_groups[3]))
    ranknet_cfg = AlbertRankNetConfig.from_dict(ranknet_cfg)
    with smart_open.open(
        os.path.join(args.output_dir, "ranknet_cfg.json"), "w"
    ) as fout:
        fout.write(ranknet_cfg.to_json_string())
    return AlbertRankNet(ranknet_cfg, scorer)


def train_net(
    x_train_df: pd.DataFrame, x_val_df: pd.DataFrame, net: AlbertRankNet, args
):
    x, y, x_descs = create_pairwise_set(x_train_df, net.scorer.cols)
    x_val, y_val, x_descs_val = create_pairwise_set(x_val_df, net.scorer.cols)
    torch.manual_seed(43)
    np.random.seed(43)

    if args.cuda:
        net = net.cuda()
    else:
        net = net.cpu()
    net.train()

    best_map_val = 0.0
    # keep track of the best run
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    dim = len(net.scorer.config.cols)
    for epoch in range(args.epochs):
        shuffled_idxes = np.arange(x.shape[0])
        np.random.shuffle(shuffled_idxes)
        x = x[shuffled_idxes]
        y = y[shuffled_idxes]
        x_descs = [x_descs[idx] for idx in shuffled_idxes]

        final_loss = inner_training_loop(
            criterion, dim, net, optimizer, x, x_descs, y, epoch, args
        )
        x_train_df["predicted_f"] = net.scorer.predict_from_df(x_train_df)

        map_train, ndcg_train = performance_stats(x_train_df, "predicted_f")
        logger.info(
            f"EoE Average training loss: {final_loss:.4f}. "
            f"MAP Train: {map_train.mean()}. NDCG Train: {ndcg_train.mean()}"
        )

        if x_val is not None and x_descs_val is not None and y_val is not None:
            net.eval()
            with torch.no_grad():
                final_val_loss = inner_training_loop(
                    criterion,
                    dim,
                    net,
                    optimizer,
                    x_val,
                    x_descs_val,
                    y_val,
                    epoch,
                    args,
                )
                x_val_df["predicted_f"] = net.scorer.predict_from_df(x_val_df)

                map_val, ndcg_val = performance_stats(x_val_df, "predicted_f")
                logger.info(
                    f"EoE Average Val Loss: {final_val_loss:.4f}. "
                    f"Val MAP: {map_val.mean()}. "
                    f"Val NDCG: {ndcg_val.mean()}."
                )

                if map_val.mean() > best_map_val:
                    best_map_val = map_val.mean()
                    with smart_open.open(
                        os.path.join(args.output_dir, "best_model.pth"), "wb"
                    ) as fout:
                        logger.info("Model is the best baseline. Saving it.")
                        torch.save(net.scorer.state_dict(), fout)
            net.train()

        # cannot compute the normalised feature importance here since estimating the mean of
        # dot product albert vectors is expensive.
        logger.info(
            "Final layers features activation: \n{}".format(
                tabulate.tabulate(
                    [
                        (feat, w)
                        for feat, w in sorted(
                            zip(
                                net.scorer.cols + ["Albert"],
                                net.scorer.fc.weight.detach().cpu().numpy()[0],
                            ),
                            key=lambda x: -abs(x[1]),
                        )
                    ],
                    headers=["Feature", "Weight"],
                )
            )
        )

    return net


def inner_training_loop(criterion, dim, net, optimizer, x, x_descs, y, epoch, args):
    total_loss = 0.0
    start_t = time.time()
    iteration = 0
    for i in range(0, x.shape[0], args.batch_size):
        iteration += 1
        if net.training:
            optimizer.zero_grad()

        x_minibatch = x[i : i + args.batch_size].to(device=net.scorer.fc.weight.device)
        y_minibatch = y[i : i + args.batch_size].to(device=net.scorer.fc.weight.device)
        x_descs_minibatch = x_descs[i : i + args.batch_size]

        ret = net(
            x_minibatch[:, :dim],
            x_minibatch[:, dim:],
            [d["base_token_ids"] for d in x_descs_minibatch],
            [d["peer_token_ids_x"] for d in x_descs_minibatch],
            [d["peer_token_ids_y"] for d in x_descs_minibatch],
        )
        loss, base_emb, peer1_emb, peer2_emb = [
            ret[k] for k in ["loss", "base_emb", "peer1_emb", "peer2_emb"]
        ]

        loss = criterion(loss, y_minibatch)

        reg_loss_l1 = 0
        reg_loss_l2 = 0
        for param in net.scorer.fc.parameters():
            reg_loss_l1 += param.norm(1)
            reg_loss_l2 += param.norm(2)

        repr_reg_loss = (
            base_emb.pow(2).sum(dim=1).mean()
            + peer1_emb.pow(2).sum(dim=1).mean()
            + peer2_emb.pow(2).sum(dim=1).mean()
        ) / 3
        # TODO: the loss should be in within ranknet. Rather than in here.
        loss += (
            args.l1_last_layer * reg_loss_l1
            + args.l2_last_layer * reg_loss_l2
            + args.repr_reg * repr_reg_loss
        )
        if net.training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss)
        mode_str = "train" if net.training else "dev"
        if (iteration + 1) % 10 == 0:
            logger.info(
                f"Average {mode_str} loss: {total_loss / (i + 1) * args.batch_size:.4f}. "
                f"Progress {i / x.shape[0] * 100:.2f}%. "
                f"Epoch {epoch}th elapsed time {time.time() - start_t:.2f} sec. "
                f"EoD: {(x.shape[0] - i - 1) / ((i + 1) / (time.time() - start_t)):.2f} sec"
            )
    return total_loss / (x.shape[0]) * args.batch_size


def performance_stats(
    x: pd.DataFrame, score_column: str, reverse=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mAP = x.groupby(["task_id", "analyst_id"]).apply(
        average_precision, score_column=score_column, reverse=reverse
    )
    ndcg = x.groupby(["task_id", "analyst_id"]).apply(
        ndcg_df, score_column=score_column, reverse=reverse
    )
    return mAP, ndcg


def error_analysis(X, ranking_column="es_rank", reverse=True):
    joined_df = X.merge(X, on=["task_id", "analyst_id"])

    ranking_column_mask = (
        (joined_df[ranking_column + "_x"] < joined_df[ranking_column + "_y"])
        if reverse
        else (joined_df[ranking_column + "_x"] > joined_df[ranking_column + "_y"])
    )

    joined_df = joined_df[
        (joined_df["relevance_score_x"] < joined_df["relevance_score_y"])
        & ranking_column_mask
    ]

    joined_df = joined_df[[REASON_1_COL_NAME + "_x", "peer_name_x"]]

    return joined_df.groupby(REASON_1_COL_NAME + "_x").count()


def populate_parser(parser):
    parser.add_argument("--train", help="train pickle", required=True)
    parser.add_argument("--test", help="test pickle", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--epochs", help="Epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--scorer_cfg", help="Config file path of scorer")
    parser.add_argument("--ranknet_cfg", help="Config file path of ranknet")

    g = parser.add_argument_group("scorer config", description="Override scorer config")

    AlbertScorerConfig.populate_argparser(g)

    g = parser.add_argument_group(
        "ranknet config", description="Override RankNet Config"
    )

    AlbertRankNetConfig.populate_argparser(g)

    parser.add_argument(
        "--model_path",
        help="Loading a pre-trained ranknet, "
        "instead of starting from a randomised model",
    )
    add_bool_argument(parser, "cuda", default=False)


def estimate_feature_scales(net: AlbertRankNet, x: pd.DataFrame, batch_size: int):
    original_mode = net.training
    net.eval()
    x, y, x_descs = create_pairwise_set(x, net.scorer.cols)
    # for all other features, just use the
    # stats on the first half minus second half of the vectors
    dim = len(net.scorer.config.cols)
    scalar_scales = np.std((x[:, :dim] - x[:, dim:]).detach().numpy(), axis=0)
    # sample 300 pairs and get the mean and variance estimate
    albert_emb_dot_prod_diffs = []
    with torch.no_grad():
        for i in range(0, min(x.shape[0], 300), batch_size):
            x_minibatch = x[i : i + batch_size].to(device=net.scorer.fc.weight.device)
            x_descs_minibatch = x_descs[i : i + batch_size]
            ret = net(
                x_minibatch[:, :dim],
                x_minibatch[:, dim:],
                [d["base_token_ids"] for d in x_descs_minibatch],
                [d["peer_token_ids_x"] for d in x_descs_minibatch],
                [d["peer_token_ids_y"] for d in x_descs_minibatch],
            )
            peer1_feat, peer2_feat = ret["peer1_feat"], ret["peer2_feat"]

            albert_emb_dot_prod_diffs.append(
                (peer1_feat[:, -1] - peer2_feat[:, -1]).cpu().numpy()
            )
    albert_emb_dot_prod_diffs = np.concatenate(albert_emb_dot_prod_diffs, axis=0)
    net.train(original_mode)

    return np.concatenate(
        [scalar_scales, np.std(albert_emb_dot_prod_diffs, keepdims=True)], axis=0
    )


def main(args=None):
    _clean_hdlrs()
    setup_logger()

    parser = argparse.ArgumentParser()
    populate_parser(parser)
    if args is None:
        args = parser.parse_args()

    log_fout = smart_open.open(os.path.join(args.output_dir, "log"), "w")
    ch = logging.StreamHandler(log_fout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)

    logger.info(args)

    with smart_open.open(args.train, "rb") as fin:
        x_train = pd.read_pickle(fin, None)
    with smart_open.open(args.test, "rb") as fin:
        x_val = pd.read_pickle(fin, None)

    ranknet = create_rank_net(args, parser, create_scorer(args, parser))

    if args.model_path:
        with smart_open.open(args.model_path, "rb") as fin:
            ranknet.scorer.load_state_dict(torch.load(fin))

    train_net(x_train.copy(), x_val.copy(), ranknet, args)

    # loading the best epoch
    if args.epochs > 0:
        with smart_open.open(
            os.path.join(args.output_dir, "best_model.pth"), "rb"
        ) as fout:
            ranknet.scorer.load_state_dict(torch.load(fout))

    scales = estimate_feature_scales(ranknet, x_train, args.batch_size * 2)

    scorer = ranknet.scorer

    weights = scorer.fc.weight.cpu().detach().numpy()[0] * scales

    logger.info(
        "Most important feature: \n {}".format(
            tabulate.tabulate(
                [
                    (feature, weight)
                    for weight, feature in sorted(
                        zip(weights, scorer.cols + ["Fine Tuned Albert"]),
                        key=lambda x: -abs(x[0]),
                    )
                ],
                headers=["Feature", "Weight"],
            )
        )
    )

    x_train["predicted_f"] = scorer.predict_from_df(x_train)

    map_train, ndcg_train = performance_stats(x_train, "predicted_f")

    x_val["predicted_f"] = scorer.predict_from_df(x_val)
    map_test, ndcg_test = performance_stats(x_val, "predicted_f")

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
            error_analysis(x_train, "predicted_f", False)
        )
    )
    logger.info(
        "Original Out of order pairs @ Train: \n {}".format(error_analysis(x_train))
    )

    logger.info(
        "Out of order pairs @ Test: \n {}".format(
            error_analysis(x_val, "predicted_f", False)
        )
    )
    logger.info(
        "Original Out of order pairs @ Test: \n {}".format(error_analysis(x_val))
    )

    return map_test.mean(), ndcg_test.mean()


if __name__ == "__main__":
    main()
