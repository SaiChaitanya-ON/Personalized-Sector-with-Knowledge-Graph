import mock
import numpy as np
import pytest
from PIL import Image

import onai.ml.tables.structure_predictor as predictor


@pytest.fixture
def mock_centernet():
    with mock.patch(
        "onai.ml.tables.structure_predictor.CenterNet", autospec=True
    ) as mcn:
        mcn.return_value.side_effect = lambda x: {"heatmap": x[:, 0]}
        yield mcn


@pytest.fixture
def mock_centernetonnx():
    with mock.patch(
        "onai.ml.tables.structure_predictor.CenterNetOnnx", autospec=True
    ) as mcn:
        mcn.return_value.side_effect = lambda x: {"heatmap": x[:, 0]}
        yield mcn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device(device, mock_centernet):
    predictor.StructurePredictor(device)
    assert str(mock_centernet.call_args.args[0]) == device


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_predict(backend, mock_centernet, mock_centernetonnx):
    m = {"torch": mock_centernet, "onnx": mock_centernetonnx}
    a = predictor.StructurePredictor(model_impl=backend)
    input = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    table = a.predict(input)
    assert (table.box == np.array([0, 0, 99, 99])).all()
    m[backend].return_value.assert_called_once()


@pytest.mark.parametrize(
    "x0, y0, x1, y1", [(0, 0, 20, 20), (20, 20, 100, 100), (30, 30, None, None)]
)
def test_predict_crop(x0, y0, x1, y1, mock_centernet):
    a = predictor.StructurePredictor()
    input = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    table = a.predict(input, x0, y0, x1, y1)
    assert (table.box == np.array([x0, y0, (x1 or 100) - 1, (y1 or 100) - 1])).all()


@pytest.mark.parametrize(
    "x0, y0, x1, y1", [(100, 0, None, None), (30, 40, 10, 10), (0, 0, 101, 101)]
)
def test_predic_raise_oob(x0, y0, x1, y1, mock_centernet):
    a = predictor.StructurePredictor()
    input = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        a.predict(input, x0, y0, x1, y1)


def test_predict_batch(mock_centernet):
    batch = [
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8)),
    ]
    a = predictor.StructurePredictor()
    output = a.predict_batch(batch)
    for i, o in zip(batch, output):
        assert (o.box == np.array([0, 0, i.height - 1, i.width - 1])).all()


def test_predict_batch_smol(mock_centernet):
    batch = [
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8)),
        Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8)),
    ]
    a = predictor.StructurePredictor(max_batch_size=2)
    output = a.predict_batch(batch)
    for i, o in zip(batch, output):
        assert (o.box == np.array([0, 0, i.height - 1, i.width - 1])).all()
