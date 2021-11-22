# TODO: withdraw support for s3 until
# https://github.com/PyTorchLightning/pytorch-lightning/pull/2175 is resolved
import atexit
import functools
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Union

import hydra
import pytorch_lightning as pl
import torch
from bsddb3 import dbshelve
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import kendalltau
from smart_open import open
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, DistilBertTokenizer, get_linear_schedule_with_warmup

from onai.ml.peers.experiment.financial_quantiser import FinancialQuantiser
from onai.ml.peers.experiment.modeling.surrogate import (
    SurrogateRepr,
    SurrogateReprConfig,
)
from onai.ml.peers.types import Financial, Financials
from onai.ml.tools.logging import setup_logger
from onai.ml.tools.torch.callback import IntervalLearningRateLogger
from onai.ml.tools.torch.ds import DCDataset, DCLazyItem, open_db_env
from onai.ml.tools.torch.functional import pad_and_stack_tensor

logger = logging.getLogger(__name__)


@dataclass
class Config:
    repr: SurrogateReprConfig
    training_ds: str = MISSING
    val_ds: str = MISSING
    margin: float = 1.0
    lr: float = 5e-5
    cuda: bool = False
    batch_size: int = 2
    resume_from_last: bool = True
    load_path: Optional[str] = None
    weight_decay: float = 0.0
    warmup_steps: int = 4000
    max_steps: int = 1000000
    cache_path: str = tempfile.mkdtemp()
    lr_schedule: bool = True
    debug: bool = False
    val_check_interval: float = 1.0


def _handle_singe_batch(batch: DCLazyItem, cfg):

    if not hasattr(_handle_singe_batch, "_tokeniser"):
        setup_logger(
            blacklisted_loggers=(
                "botocore.credentials",
                "transformers.tokenization_utils",
            )
        )
        logger.debug("PID %s: Initialising Collate FN", os.getpid())
        setattr(
            _handle_singe_batch,
            "_tokeniser",
            DistilBertTokenizer.from_pretrained(cfg.repr.text_distill_bert_tokeniser),
        )
        with open(cfg.repr.quantiser_cfg_path, "r") as fin:
            setattr(
                _handle_singe_batch,
                "_quantiser",
                FinancialQuantiser(OmegaConf.load(fin)),
            )
        setattr(
            _handle_singe_batch,
            "_id_by_country",
            {v: idx for idx, v in enumerate(cfg.repr.countries)},
        )
        setattr(
            _handle_singe_batch,
            "_cache",
            dbshelve.open(
                os.path.join(cfg.cache_path, "cache.bdb"),
                dbenv=open_db_env(cfg.cache_path),
            ),
        )

        def rm():
            logger.debug("PID %s: Finalising Collate FN", os.getpid())
            feat_cache = getattr(_handle_singe_batch, "_cache")
            feat_cache.close()

        atexit.register(rm)

    feat_cache = getattr(_handle_singe_batch, "_cache")
    batch_idx = batch.idx
    base_eid = (batch.dbs.get_dbname()[0] + str(batch.idx)).encode(encoding="utf8")
    if base_eid in feat_cache:
        return feat_cache[base_eid]

    batch = batch.get()
    batch = [
        i[0]
        for i in sorted(
            zip(batch["ls"], batch["score"]), key=lambda x: x[1], reverse=True
        )
    ]

    first_row = batch[0]
    tokeniser: DistilBertTokenizer = getattr(_handle_singe_batch, "_tokeniser")

    texts = [first_row["base_business_description"]] + [
        t["sample_business_description"] for t in batch
    ]

    text_outputs = tokeniser.batch_encode_plus(
        texts,
        max_length=cfg.repr.max_seq_length,
        pad_to_max_length=True,
        return_tensors="pt",
        return_attention_masks=True,
        add_special_tokens=True,
    )

    quantiser: FinancialQuantiser = getattr(_handle_singe_batch, "_quantiser")

    fss = []
    for r in [first_row["base_financials"]] + [r["sample_financials"] for r in batch]:
        fs: Financials = defaultdict(list)
        fss.append(fs)
        if r is None:
            continue
        for f in r:
            fs[f["mnemonic"]].append(
                Financial(
                    f["normalised_value"],
                    date.fromisoformat(f["event_date"]).year,
                    "USD",
                )
            )
    financial_outputs = quantiser.align_pad_quantise_financials(
        fss, cfg.repr.financials, cfg.repr.max_financial_yrs
    )

    id_by_country = getattr(_handle_singe_batch, "_id_by_country")

    country_outputs = torch.tensor(
        [
            id_by_country[c]
            for c in (
                [first_row["base_country"]] + [i["sample_country"] for i in batch]
            )
        ]
    )

    ret = {
        "stacked_text_inputs": text_outputs["input_ids"],
        "stacked_text_masks": text_outputs["attention_mask"],
        "stacked_financial_inputs": financial_outputs,
        "stacked_countries": country_outputs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.int),
    }

    feat_cache[base_eid] = ret

    return ret


def _ds_collate_fn(batch, cfg: Config):
    split_ret = []
    for b in batch:
        split_ret.append(_handle_singe_batch(b, cfg))

    ret = {
        # B' x B x T
        "stacked_text_inputs": pad_and_stack_tensor(
            [t["stacked_text_inputs"] for t in split_ret]
        ),
        # B' x B x T
        "stacked_text_masks": pad_and_stack_tensor(
            [t["stacked_text_masks"] for t in split_ret]
        ),
        # (F) x B' x B x Tf
        "stacked_financial_inputs": {
            k: pad_and_stack_tensor(
                [t["stacked_financial_inputs"][k] for t in split_ret]
            )
            for k in cfg.repr.financials
        },
        # B' x B
        "stacked_countries": pad_and_stack_tensor(
            [t["stacked_countries"] for t in split_ret]
        ),
        "stacked_idxes": torch.stack([t["batch_idx"] for t in split_ret], dim=0),
    }

    return ret


class SurrogateDist(pl.LightningModule):
    def val_dataloader(self):
        if self.cfg.val_ds:
            return DataLoader(
                DCDataset(self.cfg.val_ds),
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0 if self.cfg.debug else 8,
                collate_fn=functools.partial(_ds_collate_fn, cfg=self.cfg),
                pin_memory=True,
                multiprocessing_context=self.mp_ctx,
            )
        return None

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            DCDataset(self.cfg.training_ds),
            batch_size=self.cfg.batch_size,
            shuffle=not self.cfg.debug,
            num_workers=0 if self.cfg.debug else 8,
            collate_fn=functools.partial(_ds_collate_fn, cfg=self.cfg),
            pin_memory=True,
            multiprocessing_context=self.mp_ctx,
        )

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        r = self(**batch)
        return {
            "val_loss": r["norm_loss"],
            "val_tau": r["tau"],
            "val_n_ambiguous": r["n_ambiguous"],
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        tau_mean = torch.stack([x["val_tau"] for x in outputs]).mean()
        ambiguous_sum = torch.stack([x["val_n_ambiguous"] for x in outputs]).sum()
        return {
            "val_loss": val_loss_mean,
            "tau": tau_mean,
            "progress_bar": {
                "val_loss": val_loss_mean,
                "tau": tau_mean,
                "ambiguous_sum": ambiguous_sum,
            },
            "log": {
                "val_loss_mean": val_loss_mean,
                "tau": tau_mean,
                "ambiguous_sum": ambiguous_sum,
            },
        }

    def training_step(self, batch, batch_idx):
        r = self(**batch)
        loss = r["norm_loss"]

        logs = {"training_loss": loss, "training_tau": r["tau"]}

        return {"loss": loss, "progress_bar": logs, "log": logs}

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.lr, eps=1e-9)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.max_steps,
        )
        if self.cfg.lr_schedule:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def __init__(self, cfg: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repr = SurrogateRepr(cfg.repr)
        self.cfg = cfg
        self.mp_ctx = mp.get_context("spawn")

    def forward(
        self,
        stacked_text_inputs,
        stacked_text_masks,
        stacked_financial_inputs,
        stacked_countries,
        **kwargs
    ):
        # assuming that the first element of each batch is the base entity
        # the rest of the elements are peer entities that are sorted based on relevance
        # from the most relevant to the least relevant
        B_pi, B, _ = stacked_text_inputs.shape

        # (B' x B) x h
        hidden_repr: torch.Tensor = self.repr(
            stacked_text_inputs.reshape((B_pi * B, -1)),
            stacked_text_masks.reshape((B_pi * B, -1)),
            {k: v.reshape((B_pi * B, -1)) for k, v in stacked_financial_inputs.items()},
            stacked_countries.reshape((-1,)),
        )["concat"].reshape((B_pi, B, -1))

        # B' x 1 x h, take the first element out, which is the base entity
        base_repr = hidden_repr[:, 0:1, :]
        # B' x (B - 1) x h
        peer_reprs: torch.Tensor = hidden_repr[:, 1:, :]
        if self.repr.cfg.dist_type == "inner":
            # B' x 1 X (B - 1)
            dists = -torch.matmul(base_repr, peer_reprs.transpose(1, 2))
        elif self.repr.cfg.dist_type == "l2":
            dists = torch.norm(
                base_repr - peer_reprs, p=2, dim=2, keepdim=True  # B' x (B - 1) x h
            ).transpose(
                1, 2
            )  # B' x 1 x (B - 1)
        # elif self.repr.cfg.dist_type == 'cosine':
        #
        #     dists = -torch.cosine_similarity(
        #         base_repr.expand_as(peer_reprs), peer_reprs, dim=1
        #     ).reshape((1, -1))
        else:
            assert False, "unrecognised distance type %s" % self.repr.cfg.dist_type

        # B' x (B - 1) x (B - 1)
        loss = torch.max(
            torch.zeros_like(dists),
            self.cfg.margin + torch.tril(dists - dists.transpose(1, 2), -1),
        ).sum()

        # there are in total (B - 1) * (B - 2) / 2 pairs of tuplets for each B'
        norm_loss = loss / ((B - 1) * (B - 2) / 2 * B_pi)

        perfect_r = range(B - 1)
        taus = []
        n_ambiguous = 0
        for d in dists:
            # TODO: check if the shape is problmeatic
            tau = kendalltau(perfect_r, d.detach().cpu())[0]
            if math.isnan(tau):
                logger.warning("Tau is NaN! Skipping...")
                n_ambiguous += 1
            else:
                taus.append(tau)

        if len(taus) == 0:
            taus = [0.0]

        if torch.isinf(norm_loss).any().item():
            logger.warning("Detect infinite loss. Discarding the batch")
            norm_loss = 0.0

        return {
            "norm_loss": norm_loss,
            "dists": dists.squeeze(dim=1),
            "tau": torch.tensor(taus, dtype=torch.float).mean(),
            "n_ambiguous": torch.tensor(n_ambiguous, dtype=torch.int),
        }


cs = ConfigStore.instance()
cs.store(name="surrogate", node={"m": Config})


@hydra.main(config_path="conf", config_name="surrogate")
def main(cfg):

    cfg: Config = cfg["m"]

    logger.info("Config : \n%s", cfg.pretty())
    net = SurrogateDist(cfg)

    output_path = os.getcwd()

    last_ckpt_path = os.path.join(output_path, "last.ckpt")
    model_ckpt_cb = ModelCheckpoint(
        os.path.join(output_path, "{best:d}"), save_last=True
    )
    trainer_kwargs = dict(
        gpus=1 if cfg.cuda else 0,
        max_epochs=sys.maxsize,
        checkpoint_callback=model_ckpt_cb,
        default_root_dir=output_path,
        log_gpu_memory=True,
        max_steps=cfg.max_steps,
        callbacks=[IntervalLearningRateLogger(50)],
        val_check_interval=cfg.val_check_interval,
    )
    if cfg.load_path:
        trainer_kwargs["resume_from_checkpoint"] = cfg.load_path
    elif cfg.resume_from_last and os.path.exists(last_ckpt_path):
        trainer_kwargs["resume_from_checkpoint"] = last_ckpt_path

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(net)


if __name__ == "__main__":
    main()
