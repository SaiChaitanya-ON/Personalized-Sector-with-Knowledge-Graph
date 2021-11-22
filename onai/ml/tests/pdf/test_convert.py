import numpy as np

import onai.ml.pdf.convert as convert


def test_image_based():
    filename = "onai/ml/tests/pdf/examples/DUT Het_Financieele_Dagblad_-_15_01_2018.pdf"
    images = convert.as_images(filename)
    assert len(images) == 3
    assert all(image.width == 2845 and image.height == 4262 for image in images)
    assert all(len(np.unique(np.array(image).flatten())) > 1 for image in images)


def test_new_format():
    filename = "onai/ml/tests/pdf/examples/annual-report-2018.pdf"
    images = convert.as_images(filename)
    assert len(images) == 3
    assert all(image.width == 1654 and image.height == 2339 for image in images)
    assert all(len(np.unique(np.array(image).flatten())) > 1 for image in images)
