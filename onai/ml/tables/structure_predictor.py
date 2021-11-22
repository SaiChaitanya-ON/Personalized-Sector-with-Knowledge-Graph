import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .structure_model import MU, SIZE, STD, CenterNet, CenterNetOnnx
from .structure_postprocess import table_grid
from .table import Table


def _post_process(
    row_hm: np.ndarray, col_hm: np.ndarray, h: int, w: int, x: int, y: int
) -> Table:
    table = table_grid(row_hm, col_hm)
    table.scale((h / SIZE[1], w / SIZE[0]))
    table.translate(x, y)
    return table


class StructurePredictor:
    def __init__(
        self,
        device="cpu",
        max_batch_size=None,
        model_impl=os.environ.get("ONAIML_TABLES_S_MODEL", "torch"),
    ):
        self.device = torch.device(device)
        self.input_transforms = Compose(
            [Resize(SIZE), ToTensor(), Normalize(MU, STD, True)]
        )
        self.max_batch_size = max_batch_size
        model_impl = model_impl.lower()
        if model_impl == "torch":
            self.model = CenterNet(self.device)
        elif model_impl == "onnx":
            self.model = CenterNetOnnx()
        else:
            raise ValueError(f"Unknown model implementation {model_impl}")

    @staticmethod
    def _crop(
        i: Image.Image,
        x0: int = 0,
        y0: int = 0,
        x1: Optional[int] = None,
        y1: Optional[int] = None,
    ) -> Image.Image:
        if x1 is None:
            x1 = i.width
        if y1 is None:
            y1 = i.height
        if not (0 <= x0 <= i.width):
            raise ValueError("x0 is out of bounds of image")
        if x1 <= x0 or not (0 <= x1 <= i.width):
            raise ValueError("x1 is not within the image and larger than x0")
        if not (0 <= y0 <= i.height):
            raise ValueError("y0 is out of bounds of image")
        if y1 <= y0 or not (0 <= y1 <= i.height):
            raise ValueError("y1 is not within the image and larger than y0")
        if x0 != 0 or y0 != 0 or y1 != i.height or x1 != i.width:
            return i.crop((x0, y0, x1, y1))
        return i

    def predict(
        self,
        input: Image.Image,
        x0: int = 0,
        y0: int = 0,
        x1: Optional[int] = None,
        y1: Optional[int] = None,
    ) -> Table:
        cropped_input = self._crop(input, x0, y0, x1, y1)
        input = self.input_transforms(cropped_input).to(self.device).unsqueeze(0)
        output = self.model(input)
        row_hm = output["heatmap"].mean(axis=2).squeeze(0).to("cpu").detach().numpy()
        col_hm = output["heatmap"].mean(axis=1).squeeze(0).to("cpu").detach().numpy()
        return _post_process(
            row_hm, col_hm, cropped_input.height, cropped_input.width, x0, y0
        )

    def _predict_batch(
        self,
        inputs: List[Image.Image],
        x0: List[int] = None,
        y0: List[int] = None,
        x1: List[Optional[int]] = None,
        y1: List[Optional[int]] = None,
    ) -> List[Table]:
        cropped_inputs = [
            self._crop(i, _x0, _y0, _x1, _y1)
            for i, _x0, _y0, _x1, _y1 in zip(inputs, x0, y0, x1, y1)
        ]
        inputs = torch.stack(
            [self.input_transforms(i) for i in cropped_inputs], dim=0
        ).to(self.device)
        output = self.model(inputs)
        row_hms = output["heatmap"].mean(axis=2).to("cpu").detach().numpy()
        col_hms = output["heatmap"].mean(axis=1).to("cpu").detach().numpy()
        tables = [
            _post_process(rhm, chm, i.height, i.width, x, y)
            for rhm, chm, i, x, y in zip(row_hms, col_hms, cropped_inputs, x0, y0)
        ]
        return tables

    def predict_batch(
        self,
        inputs: List[Image.Image],
        x0: Optional[List[int]] = None,
        y0: Optional[List[int]] = None,
        x1: Optional[List[Optional[int]]] = None,
        y1: Optional[List[Optional[int]]] = None,
    ) -> List[Table]:
        if x0 is None:
            x0 = [0] * len(inputs)
        if y0 is None:
            y0 = [0] * len(inputs)
        if x1 is None:
            x1 = [None] * len(inputs)
        if y1 is None:
            y1 = [None] * len(inputs)

        if not (len(inputs) == len(x0) == len(y0) == len(x1) == len(y1)):
            raise ValueError("inputs are not the same length")
        if self.max_batch_size is None:
            return self._predict_batch(inputs, x0, y0, x1, y1)
        res = []
        for bidx in range(0, len(inputs), self.max_batch_size):
            res.extend(
                self._predict_batch(
                    inputs[bidx : bidx + self.max_batch_size],
                    x0[bidx : bidx + self.max_batch_size],
                    y0[bidx : bidx + self.max_batch_size],
                    x1[bidx : bidx + self.max_batch_size],
                    y1[bidx : bidx + self.max_batch_size],
                )
            )
        return res
