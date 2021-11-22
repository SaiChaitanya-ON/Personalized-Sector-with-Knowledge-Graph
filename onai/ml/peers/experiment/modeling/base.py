import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn as nn

from onai.ml.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig(BaseConfig):
    @staticmethod
    def populate_argparser(parser):
        pass

    cols: List[str]


class Scorer(nn.Module):
    def __init__(self, cfg: ScorerConfig):
        super(Scorer, self).__init__()
        self.cols = cfg.cols
        self.x_dim = len(self.cols)
        logger.info(f"{self.x_dim} features in total. " f"They are {self.cols}")
        self.fc = nn.Linear(self.x_dim, 1, bias=False)

    def forward(self, x):
        return self.fc(x)

    def predict(self, x: np.array):
        original_mode = self.training
        self.eval()
        with torch.no_grad():
            scores = self(x)
        self.train(original_mode)
        return scores

    def predict_from_df(self, df: pd.DataFrame):
        net_input = torch.tensor(df[self.cols].values).float()
        return self.predict(net_input).detach().numpy()


class RankNet(nn.Module):
    def __init__(self, scorer_cfg: ScorerConfig):
        super(RankNet, self).__init__()
        self.scorer = Scorer(scorer_cfg)

    def forward(self, x: np.array):
        batch_size, d = x.shape
        assert d // 2 == self.scorer.x_dim
        instance_dim = d // 2
        peer1 = x[:, :instance_dim]
        peer2 = x[:, instance_dim:]
        o = self.scorer(peer1) - self.scorer(peer2)

        return o
