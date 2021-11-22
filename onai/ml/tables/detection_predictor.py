from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from .detection_model import MAX_DIM, detection_config, detection_model
from .detection_post_process import backoff_regions


def _post_process(outputs, cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    # This method is WIP, basically v0 as there should be additional
    # tricks that can be exploited to get better results,
    # e.g. relative size check and margin addition.
    # Currently, this relies to best NMS with low IoU and high score thresholds.
    ret = []
    for o in outputs:
        regions_bo, scores_bo = backoff_regions(
            o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()
        )
        ret.append((regions_bo, scores_bo))
    return ret


class DetectionPredictor:
    def __init__(self, device="cpu", max_batch_size=None):
        self.device = torch.device(device)
        self.model = detection_model(self.device)
        self.cfg = detection_config()
        self.input_transforms = ToTensor()
        self.max_batch_size = max_batch_size

    @staticmethod
    def check_img(i: Image.Image) -> bool:
        return i.width <= MAX_DIM and i.height <= MAX_DIM

    def predict(
        self, input: Union[Image.Image, Sequence[Image.Image]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(input, Sequence):
            input = [input]
        if not all(self.check_img(i) for i in input):
            raise ValueError("Invalid input image dimensions")
        input = [self.input_transforms(i).to(self.device) for i in input]
        output = self.model(input)
        return _post_process(output, self.cfg)
