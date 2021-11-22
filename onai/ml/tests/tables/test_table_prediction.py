import pytest
from pikepdf._qpdf import PdfError

from onai.ml.tables import TablePredictor, pdf_bytes_to_images


@pytest.mark.slow
def test_table_predictor():
    filename = "onai/ml/tests/pdf/examples/short_example.pdf"
    pred = TablePredictor(1, 3)

    with open(filename, "rb") as f:
        file_contents = f.read()
        contents, images = pdf_bytes_to_images(file_contents)

    tables = pred(images)

    assert len(tables) == 3
    assert len(tables[0]) == 0
    assert len(tables[1]) == 1
    assert len(tables[2]) == 1


def test_non_pdf():

    with pytest.raises(PdfError):
        pdf_bytes_to_images(b"not a pdf")


def test_pdf_page_limit():
    filename = "onai/ml/tests/pdf/examples/short_example.pdf"
    pred = TablePredictor(max_pages=1)

    with open(filename, "rb") as f:
        file_contents = f.read()
        contents, images = pdf_bytes_to_images(file_contents)

    with pytest.raises(ValueError):
        pred(images)

    pred = TablePredictor(max_pages=3)
    pred(images)
