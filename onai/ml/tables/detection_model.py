import json
from typing import Dict, Union

import torch
import torchvision

from onai.ml.tools.file import cached_fs_open

MODEL_WEIGHT_URL = (
    "s3://oaknorth-staging-non-confidential-ml-artefacts/tabledetection/0.0.1/model.pth"
)

MODEL_CONFIG_URL = "s3://oaknorth-staging-non-confidential-ml-artefacts/tabledetection/0.0.1/model_cfg.json"
MAX_DIM = 5000


def detection_config() -> Dict:
    with cached_fs_open(MODEL_CONFIG_URL) as fin:
        return json.load(fin)


def detection_model(device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    cfg = detection_config()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, progress=False, pretrained_backbone=False
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, cfg["n_classes"]
    )
    with cached_fs_open(MODEL_WEIGHT_URL) as fin:
        weights = torch.load(fin, map_location=device)
    model.roi_heads.nms_thresh = cfg["nms_threshold"]
    model.roi_heads.score_thresh = cfg["score_threshold"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return torch.no_grad()(model)
