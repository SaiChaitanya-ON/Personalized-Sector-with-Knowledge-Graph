import tempfile
from typing import List

import numpy as np
from PIL import Image

from ..pdf import convert
from ..pdf.extraction import content_lines
from .detection_predictor import DetectionPredictor
from .structure_predictor import StructurePredictor
from .table import Table, TextBox


def pdf_bytes_to_images(input_bytes: bytes) -> [List[TextBox], List[Image.Image]]:
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(input_bytes)

        # pdf pages -> images
        images = convert.as_images(temp_file.name)
        # scrape text
        try:
            contents = content_lines(temp_file.name)
        except ValueError:
            contents = [[] for _ in images]
        contents = [[TextBox.from_args(*c) for c in conts] for conts in contents]

    return contents, images


class TablePredictor:
    """Three step process:
    1) Convert each page in an uploaded PDF to an image.
    2) Detect all table bounding boxes for each image.
    3) Crop image to each bounding box, and extract table structure.
    """

    def __init__(
        self,
        detector_max_batch_size: int = 1,
        structure_max_batch_size: int = 1,
        max_pages: int = 5,
    ):
        self.table_detector = DetectionPredictor(max_batch_size=detector_max_batch_size)
        self.structure_predictor = StructurePredictor(
            max_batch_size=structure_max_batch_size
        )
        self.max_pages = max_pages

    def __call__(self, images: List[Image.Image]) -> List[List[Table]]:

        if len(images) > self.max_pages:
            raise ValueError(
                "File has %i pages, limit is %i" % (len(images), self.max_pages)
            )

        pages = []
        for img in images:
            image = img.convert("RGB")

            # image -> table regions
            # TODO: Batch this?
            # [0] because we only pass in one image, which is then wrapped in a list.
            regions, confidences = self.table_detector.predict(image)[0]

            if regions.shape[0] == 0:
                pages.append([])
                continue
            x0, y0, x1, y1 = regions.astype(np.int32).T
            tables = self.structure_predictor.predict_batch(
                [image for _ in x0], x0=x0, y0=y0, x1=x1, y1=y1
            )
            for tbl in tables:
                tbl.normalise(image.height, image.width)
            pages.append(tables)
        self.images = images
        return pages
