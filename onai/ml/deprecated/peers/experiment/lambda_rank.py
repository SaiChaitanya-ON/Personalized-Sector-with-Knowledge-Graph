import argparse
import functools
import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import smart_open
import tabulate
import torch
from GPyOpt.methods import BayesianOptimization
from sklearn.model_selection import KFold
from torch import optim

from onai.ml.tools.logging import setup_logger
from onai.ml.peers.experiment.evaluate_consensus import REASON_1_COL_NAME
from onai.ml.peers.experiment.modeling import Scorer
from onai.ml.peers.feature_extractor import _LAST_REVENUE_WINDOWS

logger = logging.getLogger(__name__)

# TODO: this should be refactored and use a file to specify what features to use
_COLS = [
    "es_score",
    "es_score_normed",
    "predicted_industries_overlap",
    "country_overlap",
    "weighted_symmetric_diff",
    "weighted_intersection",
    "weighted_intersection_negative_sample",
    "bert_embedding_sim_hidden_state",
    "bert_embedding_sim",
    "bert_embedding_1st_sim",
    "bert_embedding_pretrained",
    "weighted_intersection_negative_sample_tail_end",
    "peer_diff",
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


def create_pairwise_set(X):
    joined_cols = [el + "_x" for el in _COLS] + [el + "_y" for el in _COLS]

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

    return X, Y


def ndcg_df(df, score_column, reverse=False):
    tst = [(el["relevance_score"], el[score_column]) for _, el in df.iterrows()]
    l = [el for el in sorted(tst, key=lambda el: el[1] if reverse else -el[1])]
    return ndcg([relevance for relevance, _ in l])


def average_precision(df, score_column, reverse=False):
    tst = [(el["relevance_score"], el[score_column]) for _, el in df.iterrows()]
    l = [el for el in sorted(tst, key=lambda el: el[1] if reverse else -el[1])]

    rel = 0
    total = 0
    m = 0
    for k, (relevance, _) in enumerate(l):
        if relevance > 0:
            rel += 1
            precision_at_k = rel / (k + 1)
            total += precision_at_k
            m += 1

    if m == 0:
        return 0

    return total / m


def dcg(labels):
    labels = np.array(labels, dtype=np.float)
    labels = 2 ** labels - 1
    denoms = np.log2(np.arange(len(labels)) + 2)
    return (labels / denoms).sum()


def perfect_dcg(labels):
    return dcg(sorted(labels, reverse=True))
    # labels = (2 ** np.array(sorted(labels, reverse=False)) - 1)
    # 1-indexed instead of 0-indexed
    # denoms = np.log2(np.arange(len(labels)) + 1 + 1)
    # return (labels  / denoms).sum()


def ndcg(labels):
    return dcg(labels) / perfect_dcg(labels)


def train_net(x, epochs=20000, lbd=1e-3, alpha=1e-3, lr=0.01, lambda_typ="ndcg"):
    torch.manual_seed(42)
    np.random.seed(42)

    scorer = Scorer(len(_COLS))

    optimizer = optim.Adam(scorer.parameters(), lr=lr, weight_decay=0.0)

    grouped_x = [df for _, df in x.groupby(["task_id", "analyst_id"])]
    for epoch in range(epochs):
        random.shuffle(grouped_x)
        # online per query.
        # TOOD: implement the mini batch version of this
        total_eval_metric = 0.0
        for i in range(0, len(grouped_x)):
            x_batch = grouped_x[i]
            y_batch = torch.tensor(x_batch["relevance_score"].values, dtype=torch.int)
            if len(set(y.item() for y in y_batch)) < 2:
                # there is no learning signal
                continue

            optimizer.zero_grad()

            n_docs = len(x_batch.index)
            # B x 1
            scores = scorer(torch.from_numpy(x_batch[_COLS].to_numpy()).float())

            with torch.no_grad():
                if lambda_typ == "ndcg":
                    lambda_i, local_eval_metric = lambda_ndcg(n_docs, scores, y_batch)
                elif lambda_typ == "map":
                    lambda_i, local_eval_metric = lambda_map(n_docs, scores, y_batch)
                elif lambda_typ == "vanilla":
                    lambda_i, local_eval_metric = lambda_vanilla(
                        n_docs, scores, y_batch
                    )
                else:
                    assert False, "Unknown lambda type"

            total_eval_metric += local_eval_metric
            scores.backward(lambda_i)

            reg_loss_l1 = 0
            reg_loss_l2 = 0
            for param in scorer.parameters():
                reg_loss_l1 += param.norm(1)
                reg_loss_l2 += param.norm(2) ** 2

            reg_loss = lbd * reg_loss_l1 + alpha * reg_loss_l2
            reg_loss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Average {lambda_typ}: {total_eval_metric / len(grouped_x)}")

    return scorer


def lambda_map(n_docs, scores, y_batch):

    raise NotImplementedError


def lambda_vanilla(n_docs, scores, y_batch):
    y_batch_unsqze = y_batch.unsqueeze(1)
    s_ij = (y_batch_unsqze > y_batch_unsqze.t()).float() - (
        y_batch_unsqze < y_batch_unsqze.t()
    ).float()
    score_diff = 1.0 + torch.exp(scores - scores.t())
    lambda_ij = 0.5 * (1 - s_ij) - 1.0 / score_diff
    lambda_i = lambda_ij.sum(axis=1, keepdim=True)

    loss_ij = 0.5 * (1 - s_ij) * (scores - scores.t()) + torch.log(
        1 + torch.exp(-(scores - scores.t()))
    )
    return lambda_i, loss_ij.sum() / (np.prod(loss_ij.shape))


def lambda_ndcg(n_docs, scores, y_batch):
    y_batch_unsqze = y_batch.unsqueeze(1)
    ndcg_n = 1.0 / perfect_dcg(y_batch)
    (sorted_scores, sorted_idxes) = scores.squeeze(dim=1).sort(descending=True)
    doc_ranks = torch.zeros(n_docs)
    doc_ranks[sorted_idxes] = 1 + torch.arange(n_docs).float()
    local_eval_metric = ndcg(y_batch[sorted_idxes])
    # B x B
    score_diff = 1.0 + torch.exp(scores - scores.t())
    # matrix whether peer i is more relevant than peer j
    s_ij = (y_batch_unsqze > y_batch_unsqze.t()).float() - (
        y_batch_unsqze < y_batch_unsqze.t()
    ).float()
    delta_ndcg_nom = (2 ** y_batch_unsqze).float()
    delta_ndcg_denom = 1.0 / torch.log2(1 + doc_ranks.unsqueeze(1))
    delta_ndcg = (
        ndcg_n
        * (delta_ndcg_nom - delta_ndcg_nom.t())
        * (delta_ndcg_denom - delta_ndcg_denom.t())
    )
    lambda_ij = (0.5 * (1 - s_ij) - 1.0 / score_diff) * delta_ndcg.abs()
    # note that lambda_j and lambda_i are the same, as lambda_ij is (and must be) symmetric
    lambda_i = lambda_ij.sum(axis=1, keepdim=True)
    return lambda_i, local_eval_metric


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


def cv_score(x, args, n_splits=5, lbd=0.0, alpha=0.0, lr=1e-3):

    grouped = x.groupby("task_id").count()
    multi_annotator_ids = set(grouped[grouped["base_name"] > 20].index)

    task_ids = np.array(list(set(x["task_id"]) - multi_annotator_ids))
    kf = KFold(n_splits=n_splits, random_state=42)
    ndcg_sum = 0

    for i, (train_index, val_index) in enumerate(kf.split(task_ids)):
        train_ids = np.append(
            task_ids[train_index], np.array(list(multi_annotator_ids))
        )
        val_ids = task_ids[val_index]
        logger.info(f"----- Fold {i} -----")

        x_train = x[x["task_id"].isin(train_ids)]
        logger.info(x_train.shape)
        net = train_net(
            x_train, epochs=30, lbd=lbd, alpha=alpha, lr=lr, lambda_typ=args.lambda_type
        )

        train_input = torch.from_numpy(x_train[_COLS].to_numpy()).float()
        x_train["predicted_f"] = net(train_input).detach().numpy()
        map_train, ndcg_train = performance_stats(x_train, "predicted_f")

        x_val = x[x["task_id"].isin(val_ids)]
        val_input = torch.from_numpy(x_val[_COLS].to_numpy()).float()
        x_val["predicted_f"] = net(val_input).detach().numpy()
        map_val, ndcg_val = performance_stats(x_val, "predicted_f")

        logger.info(
            "Fold {} Report: \n {}".format(
                i,
                tabulate.tabulate(
                    [
                        ("Training set", map_train.mean(), ndcg_train.mean()),
                        ("Validation set", map_val.mean(), ndcg_val.mean()),
                    ],
                    headers=("Dataset", "MAP", "NDCG"),
                ),
            )
        )

        ndcg_sum += ndcg_val.mean()

    return ndcg_sum / n_splits


def func(param, x_train, args):
    param = np.atleast_2d(np.exp(param))
    lbd = param[0, 0]
    alpha = param[0, 1]
    logger.info(f"Training with lambda={lbd} and alpha={alpha}")
    return cv_score(x_train, lbd=lbd, alpha=alpha, args=args)


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
        & (joined_df["relevance_score_y"] >= 1)
        & (joined_df["relevance_score_x"] == 0)
    ]

    joined_df = joined_df[[REASON_1_COL_NAME + "_x", "peer_name_x"]]

    return joined_df.groupby(REASON_1_COL_NAME + "_x").count()


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train csv", required=True)
    parser.add_argument("--test", help="test csv", required=True)
    parser.add_argument("--output_folder", help="Output directory", required=True)
    parser.add_argument(
        "--lambda_type",
        help="What is the type of loss to use in lambdarank?",
        default="vanilla",
    )

    args = parser.parse_args()

    with smart_open.open(args.train, "r") as fin:
        x_train = pd.read_csv(fin)

    domain = [
        {"name": "lbd", "type": "continuous", "domain": (-12.0, 4.0)},
        {"name": "alpha", "type": "continuous", "domain": (-12.0, 0)},
    ]

    opt = BayesianOptimization(
        f=functools.partial(func, x_train=x_train, args=args),
        domain=domain,
        maximize=True,
        normalize_Y=True,
    )

    opt.run_optimization(max_iter=15)
    with smart_open.open(
        os.path.join(args.output_folder, "bo_convergence.png"), "wb"
    ) as fout:
        opt.plot_convergence(fout)
    with smart_open.open(
        os.path.join(args.output_folder, "bo_acquisition.png"), "wb"
    ) as fout:
        opt.plot_acquisition(fout)

    lbd_best = opt.X[np.argmin(opt.Y)]
    lbd_best = np.exp(lbd_best)
    logger.info(f"Best hyper parameter is: {lbd_best}")

    net = train_net(x_train, epochs=100, lbd=lbd_best[0], alpha=lbd_best[1], lr=1e-3)

    # net = train_net(x_train, epochs=100, lbd=1e-6, alpha=1e-6, lr=1e-3, lambda_typ=args.lambda_type)

    weights = net.fc.weight.detach().numpy()[0]

    logger.info(
        "Most important feature: \n {}".format(
            tabulate.tabulate(
                [
                    (feature, weight)
                    for weight, feature in sorted(
                        zip(weights, _COLS), key=lambda x: -abs(x[0])
                    )
                ],
                headers=["Feature", "Weight"],
            )
        )
    )

    x_train["predicted_f"] = (
        net(torch.from_numpy(x_train[_COLS].to_numpy()).float()).detach().numpy()
    )
    map_train, ndcg_train = performance_stats(x_train, "predicted_f")

    with smart_open.open(args.test) as fin:
        x_test = pd.read_csv(fin)

    x_test["predicted_f"] = (
        net(torch.from_numpy(x_test[_COLS].to_numpy()).float()).detach().numpy()
    )
    map_test, ndcg_test = performance_stats(x_test, "predicted_f")

    map_train_orig, ndcg_train_orig = performance_stats(
        x_train, "es_rank", reverse=True
    )
    map_test_orig, ndcg_test_orig = performance_stats(x_test, "es_rank", reverse=True)

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
                    ),
                    (
                        "Orig",
                        map_train_orig.mean(),
                        ndcg_train_orig.mean(),
                        map_test_orig.mean(),
                        ndcg_test_orig.mean(),
                    ),
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
        "Out of order pairs: \n {}".format(
            error_analysis(x_train, "predicted_f", False)
        )
    )
    logger.info("Original Out of order pairs: \n {}".format(error_analysis(x_train)))


if __name__ == "__main__":
    main()
