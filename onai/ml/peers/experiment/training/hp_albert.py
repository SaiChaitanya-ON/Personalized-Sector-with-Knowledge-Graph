import argparse
import copy
import logging
import math
import os
import shutil
import tempfile

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.track import get_session
from ray.tune.trial import Trial

from onai.ml.peers.experiment.training.albert_trainer import main as train_main
from onai.ml.peers.experiment.training.albert_trainer import populate_parser
from onai.ml.tools.logging import _clean_hdlrs, setup_logger

logger = logging.getLogger(__name__)


def train(config):
    sess = get_session()

    args = copy.copy(config["args"])
    logdir, trial_name = os.path.split(sess._logdir)
    if not trial_name:
        _, trial_name = os.path.split(logdir)

    args.output_dir = os.path.join(args.output_dir, "train_dump", trial_name)

    os.makedirs(args.output_dir, exist_ok=True)

    args.l1_last_layer = math.exp(config["l1_last_layer"])
    args.l2_last_layer = math.exp(config["l2_last_layer"])
    args.repr_reg = math.exp(config["repr_reg"])
    args.lr = config["lr"]
    _, best_ndcg = train_main(args)
    tune.track.log(best_ndcg=best_ndcg)


def trial_name_creator(trial: Trial):
    return trial.trial_id


def bayes_opt_search(args):
    if args.output_dir.startswith("s3://") or args.output_dir.startswith("gs://"):
        local_dir = tempfile.mkdtemp()
        upload_dir = os.path.join(args.output_dir, "ray_results")
    else:
        local_dir = os.path.join(args.output_dir, "ray_results")
        upload_dir = None
    space = {
        "l1_last_layer": (math.log(1.0e-6), math.log(1.0e-1)),
        "l2_last_layer": (math.log(1.0e-6), math.log(1.0e-1)),
        "repr_reg": (math.log(1.0e-6), math.log(1.0e-1)),
        "lr": (1.0e-5, 5.0e-5),
    }
    algo = BayesOptSearch(
        space,
        max_concurrent=args.hp_max_concurrent,
        metric="best_ndcg",
        mode="max",
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    )
    analysis = tune.run(
        train,
        name="bert",
        search_alg=algo,
        config={"args": args},
        num_samples=args.hp_budget,
        resources_per_trial={"gpu": 1.0, "cpu": 1},
        local_dir=local_dir,
        upload_dir=upload_dir,
        trial_name_creator=trial_name_creator,
    )
    logger.info(
        "Best Config: \n {}".format(
            analysis.get_best_config(metric="best_ndcg", mode="max")
        )
    )
    if upload_dir is not None:
        shutil.rmtree(local_dir, True)


def main():
    _clean_hdlrs()
    setup_logger()
    parser = argparse.ArgumentParser()
    populate_parser(parser)
    parser.add_argument("--hp_budget", type=int, default=10)
    parser.add_argument("--hp_max_concurrent", type=int, default=1)
    args = parser.parse_args()
    bayes_opt_search(args)


if __name__ == "__main__":
    main()
