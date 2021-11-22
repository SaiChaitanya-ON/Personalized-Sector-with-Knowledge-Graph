import mock
import numpy as np
import pytest
import torch
from PIL import Image

import onai.ml.tables.detection_predictor as predictor


@pytest.fixture
def mock_fasterrcnn():
    with mock.patch(
        "onai.ml.tables.detection_predictor.detection_model", autospec=True
    ) as mdn:
        mdn.return_value.side_effect = lambda x: [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0)} for i in x
        ]
        yield mdn


def test_image_too_large(mock_fasterrcnn):
    input = Image.fromarray(
        np.random.randint(256, size=(5096, 4096, 3)).astype(np.uint8), "RGB"
    )
    pred = predictor.DetectionPredictor("cpu")
    with pytest.raises(ValueError):
        pred.predict(input)


def test_cor_output(mock_fasterrcnn):
    inputs = [
        Image.fromarray(
            np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8), "RGB"
        ),
        Image.fromarray(
            np.random.randint(256, size=(1024, 512, 3)).astype(np.uint8), "RGB"
        ),
    ]
    pred = predictor.DetectionPredictor("cpu")
    output = pred.predict(inputs)
    assert len(output) == 2
    assert isinstance(output[0][0], np.ndarray)
    assert isinstance(output[0][1], np.ndarray)
