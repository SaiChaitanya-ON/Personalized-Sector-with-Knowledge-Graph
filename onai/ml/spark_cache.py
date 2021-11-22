import functools
import json
import logging
import os
import tarfile
import tempfile
import time
from random import random

import torch
from bsddb3 import db, dbshelve
from bsddb3.dbshelve import DBShelf
from transformers import BertTokenizer

from onai.ml.coa.computation import ComputationalTree
from onai.ml.peers.candidate_suggestion.bert import BertRankNetCandidateSuggestion
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.candidate_suggestion.surrogate import SurrogateCandidateSuggestion
from onai.ml.tools.file import cached_fs_open
from onai.ml.tools.logging import setup_logger as _setup_logger
from onai.ml.tools.torch.ds import open_db_env

logger = logging.getLogger(__name__)

_cache = {}

# berkeley db handles cache
_dc_cache = {}


def setup_logger():
    if "_is_logger_init" not in _cache:
        _setup_logger(blacklisted_loggers=["elasticsearch", "filelock"])
        _cache["_is_logger_init"] = True
        # logger.info(id(_cache))


def get_es_cs(args) -> ESCandidateSuggestion:
    if not hasattr(get_es_cs, "_cs"):
        logger.info("Initialising es cs PID: %s", os.getpid())
        setattr(
            get_es_cs,
            "_cs",
            ESCandidateSuggestion(
                es_host=args.es_host,
                es_port=args.es_port,
                dp_financials=["TOTAL_REVENUE", "EBIT", "EBITDA"],
                retry_on_timeout=True,
                max_request_timeout=60,
            ),
        )
    return getattr(get_es_cs, "_cs")


REGION_MODEL_PATH = (
    "s3://oaknorth-staging-non-confidential-ml-artefacts/region/v0.0.1/model.json"
)


def get_bert_cs(args) -> BertRankNetCandidateSuggestion:
    if "bert_cs" not in _cache:
        logger.info("initialising bert cs PID: %s", os.getpid())
        _cache["bert_cs"] = cs = BertRankNetCandidateSuggestion(
            os.path.join(args.bert_model_path, "scorer_cfg.json"),
            os.path.join(args.bert_model_path, "best_model.pth"),
            es_host=args.es_host,
            es_port=args.es_port,
            cuda=args.cuda,
        )
        cs.bert_scorer.config.pred_batch_size = args.pred_batch_sz

    return _cache["bert_cs"]


def get_bert_tokeniser(args) -> BertTokenizer:
    if "bert_tokeniser" not in _cache:
        with tempfile.TemporaryDirectory() as tmpdir, open(
            os.path.join(args.model_path, "berttokeniser.tar.gz"), "rb"
        ) as fin, tarfile.open(fileobj=fin, mode="r:gz") as tar_fout:
            tar_fout.extractall(tmpdir)
            _cache["bert_tokeniser"] = BertTokenizer.from_pretrained(tmpdir)
    return _cache["bert_tokeniser"]


def speed_counter(count=1) -> float:
    if "_task_completion_cnt" not in _cache:
        _cache["_task_completion_cnt"] = 0
        _cache["_task_start_time"] = time.time()
    _cache["_task_completion_cnt"] += count
    return _cache["_task_completion_cnt"] / (time.time() - _cache["_task_start_time"])


def get_coa_tree() -> ComputationalTree:
    if "coa_tree" not in _cache:
        logger.info("initialising coa tree: %s", os.getpid())
        _cache["coa_tree"] = ComputationalTree()
    return _cache["coa_tree"]


def get_dbenv(args):
    if "_dbenv" not in _cache:
        _cache["_dbenv"] = open_db_env(args.dbenv)
    return _cache["_dbenv"]


def get_opened_index(p: str, args) -> DBShelf:
    if p not in _dc_cache:
        _dc_cache[p] = dbshelve.open(
            p, dbenv=get_dbenv(args), flags=db.DB_CREATE, filetype=db.DB_RECNO
        )

    return _dc_cache[p]


def get_and_add_written_rows():
    if "_written_row" not in _cache:
        _cache["_written_row"] = 0
    _cache["_written_row"] += 1
    return _cache["_written_row"]


def broadcast_variable(variable_ctor):
    @functools.wraps(variable_ctor)
    def inner(*args, **kwargs):
        key = variable_ctor.__qualname__
        if key not in _cache:
            logger.info("PID %d: Initialising %s", os.getpid(), key)
            _cache[key] = variable_ctor(*args, **kwargs)
        return _cache[key]

    return inner


@broadcast_variable
def get_region_model():
    t = random() * 5.0
    time.sleep(t)
    with cached_fs_open(REGION_MODEL_PATH, "rb") as fin:
        return json.load(fin)
    # if "region_model" not in _cache:
    #     with cached_smart_open(REGION_MODEL_PATH, "rb") as fin:
    #         _cache["region_model"] = json.load(fin)
    # return _cache["region_model"]


@broadcast_variable
def get_surrogate_cs(p, cuda, num_threads=4):
    t = random() * 5.0
    time.sleep(t)
    torch.set_num_threads(num_threads)
    return SurrogateCandidateSuggestion(
        os.path.join(p, "scorer_cfg.yaml"),
        os.path.join(p, "best_model.pth"),
        es_host="berry-es-test.ml.onai.cloud",
        es_port=80,
        cuda=cuda,
    )
