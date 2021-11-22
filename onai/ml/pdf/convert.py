import os
import subprocess
import tempfile
from typing import List, Optional, Sequence

from pdf2image import convert_from_path
from pikepdf import Pdf
from PIL import Image

LOW_MEM = str(os.environ.get("CI", False)).lower() == "true"


class ConversionException(Exception):
    pass


def convert(
    path: str,
    dpi: Optional[int],
    size: Optional[Sequence[int]],
    password: Optional[str],
    low_mem: bool = LOW_MEM,
):
    tc = min(os.cpu_count(), 4)
    if low_mem:
        tc = 1
    try:
        return convert_from_path(
            path, dpi=dpi, userpw=password, size=size, thread_count=tc
        )
    except subprocess.CalledProcessError as err:
        raise ConversionException(str(err))


def as_images(
    path: str,
    password: Optional[str] = None,
    size: Optional[Sequence[int]] = None,
    dpi: int = 200,
    batchsize: int = 15,
    low_mem: bool = LOW_MEM,
) -> List[Image.Image]:
    if low_mem:
        batchsize = 1
    with Pdf.open(path, password=password or "") as pdf:
        if len(pdf.pages) <= batchsize:
            return convert(path, dpi, size, password)
        res = []
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Pdf.new()
            for i in range(len(pdf.pages)):
                out_pdf.pages.append(pdf.pages[i])
                if len(out_pdf.pages) >= batchsize:
                    out_path = os.path.join(tmpdir, f"out_{i}.pdf")
                    out_pdf.save(out_path)
                    res.extend(convert(out_path, dpi, size, None, low_mem=low_mem))
                    out_pdf = Pdf.new()
            if len(out_pdf.pages):
                out_path = os.path.join(tmpdir, f"out_{len(pdf.pages)}.pdf")
                out_pdf.save(out_path)
                res.extend(convert(out_path, dpi, size, None))
        return res
