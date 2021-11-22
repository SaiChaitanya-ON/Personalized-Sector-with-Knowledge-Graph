import logging
from os import PathLike
from typing import BinaryIO, List, Tuple, Union

from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.layout import LAParams, LTChar, LTTextLine
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFException

logger = logging.getLogger(__name__)


def __inside(x0, y0, x1, y1, cropbox):
    cx0, cy0, cx1, cy1 = cropbox
    return x0 >= cx0 and y0 >= cy0 and x1 <= cx1 and y1 <= cy1


def __extract(
    item, width, height, cropbox, cont_type=LTTextLine, cur_depth=0, max_depth=100
):
    if cur_depth >= max_depth:
        logger.warning("Max depth reached")
        return []
    ret = []
    if isinstance(item, cont_type):
        text = item.get_text()
        x0, y0, x1, y1 = item.bbox
        x0 /= width
        x1 /= width
        y0 /= height
        y1 /= height
        # Invert Y axis to be in image (matrix) order
        y0, y1 = 1 - y1, 1 - y0
        if not __inside(x0, y0, x1, y1, cropbox):
            logger.warning("Content ouside of page cropbox; skipping")
        elif x1 <= x0 or y1 <= y0:
            logger.warning("Item with invalid size; skipping")
        else:
            ret.append((y0, x0, y1, x1, text))
    try:
        for i in item:
            try:
                ret.extend(
                    __extract(
                        i, width, height, cropbox, cont_type, cur_depth + 1, max_depth
                    )
                )
            except PDFException:
                logger.exception("Problem extracting %s", str(i))
    except TypeError:
        logger.debug("Item noniterable")
    return ret


def _contents(
    pdf_file: BinaryIO, cont_type=LTTextLine
) -> List[List[Tuple[float, float, float, float, str]]]:
    pdf_file.seek(0)
    try:
        parser = PDFParser(pdf_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = PDFLayoutAnalyzer(rsrcmgr, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        document_contents = []
        for pagenum, page in enumerate(PDFPage.create_pages(doc)):
            logger.debug("Processing page %d", pagenum)
            # Typically mediabox is (0,0, width, height)
            # Describing the coordinates for the "paper"
            # Cropbox contains region that should be cut after rendering
            # (0,0) is bottom left.
            *_, m_width, m_height = page.mediabox
            cx0, cy0, cx1, cy1 = page.cropbox
            width = max(m_width, cx1)
            height = max(m_height, cy1)
            if width <= 0 or height <= 0:
                raise ValueError("Unknown page size")
            cropbox = [cx0 / width, cy0 / height, cx1 / width, cy1 / height]
            interpreter.process_page(page)
            page_item = device.cur_item
            page_contents = __extract(page_item, width, height, cropbox, cont_type)
            document_contents.append(page_contents)
        return document_contents
    except PDFException:
        raise ValueError("Could not extract contents")


def content_lines(pdf: Union[PathLike, BinaryIO]):
    if isinstance(pdf, BinaryIO):
        return _contents(pdf, LTTextLine)
    with open(pdf, "rb") as pdf_file:
        return _contents(pdf_file, LTTextLine)


def content_chars(pdf: Union[PathLike, BinaryIO]):
    if isinstance(pdf, BinaryIO):
        return _contents(pdf, LTChar)
    with open(pdf, "rb") as pdf_file:
        return _contents(pdf_file, LTChar)
