import torch

from onai.ml.tables.detection_model import detection_model


def test_model_load():
    detection_model(device="cpu")


def test_model_output():
    a = detection_model(device="cpu")
    out = a([torch.randn(3, 256, 256), torch.randn(3, 512, 1024)])
    assert len(out) == 2
    assert "boxes" in out[0]
    assert "scores" in out[0]
