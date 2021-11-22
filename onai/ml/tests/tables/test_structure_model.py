import torch

import onai.ml.tables.structure_model as model


def test_model_load():
    model.CenterNet("cpu")


def test_model_output_shape():
    a = model.CenterNet("cpu")
    a.eval()
    out = a(torch.randn(1, 3, 256, 256))["heatmap"]
    assert out.size(0) == 1
    assert out.size(1) == 256
    assert out.size(2) == 256


def test_model_load_onnx():
    model.CenterNetOnnx()


def test_modelonnx_output_shape():
    a = model.CenterNetOnnx()
    out = a(torch.randn(1, 3, 512, 512))["heatmap"]
    assert out.size(0) == 1
    assert out.size(1) == 512
    assert out.size(2) == 512
