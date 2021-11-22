from dataclasses import dataclass
from typing import Generator, Optional, Sequence, Union

import numpy as np


@dataclass(eq=True, frozen=True)
class TextBox:
    # Box is in (y0, x0, y1, x1) order
    box: np.ndarray
    text: str

    @classmethod
    def from_args(cls, y0, x0, y1, x1, text):
        return cls(np.array([y0, x0, y1, x1]), text)

    def __getitem__(self, idx: int) -> Union[np.ndarray, str]:
        if idx == 0:
            return self.box
        elif idx == 1:
            return self.text
        raise IndexError(f"{idx} is out of bounds")

    def translate(self, x, y):
        return TextBox(
            np.array(
                [self.box[0] + y, self.box[1] + x, self.box[2] + y, self.box[3] + x]
            ),
            self.text,
        )

    def scale(self, sx, sy):
        return TextBox(
            np.array(
                [self.box[0] * sy, self.box[1] * sx, self.box[2] * sy, self.box[3] * sx]
            ),
            self.text,
        )

    def __eq__(self, other):
        return (self.box == other.box).all() and self.text == other.text

    def __ne__(self, other):
        return not self.__eq__(other)


def iou(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - intersection_area
    if union > 0.0:
        return intersection_area / float(union)
    else:
        return 0.0


class Table:
    """
    Table is a collection of implicit cells
    Defined as grid coordinates (rows, cols)
    and a collection of span identifiers.

    rows[0] is the top coordinate of the table
    cols[0] if the left coordinate of the table
    rows[-1] is the (exclusive) bottom coordinate of the table
    cols[-1] is the (exclusive) right cooridinate of the table
    one can only access
    """

    def __init__(self, rows: np.ndarray, cols: np.ndarray):
        """WIP -- signiture will change"""
        self._normalised = False
        self._rows = rows
        self._cols = cols

        self._cell_start_row = []
        self._cell_end_row = []
        self._cell_start_col = []
        self._cell_end_col = []

        for r in range(len(rows) - 1):
            for c in range(len(cols) - 1):
                self._add_cell(r, c, r, c)

    @property
    def box(self) -> np.ndarray:
        return np.array([self._rows[0], self._cols[0], self._rows[-1], self._cols[-1]])

    @property
    def slicebox(self) -> np.ndarray:
        return (
            slice(self._rows[0], self._rows[-1]),
            slice(self._cols[0], self._cols[-1]),
        )

    def _box(self, sr, sc, er, ec) -> np.ndarray:
        y_min = self._rows[sr]
        x_min = self._cols[sc]
        if er == len(self._rows) - 2:
            y_max = self._rows[-1]
        else:
            y_max = self._rows[er + 1]
        if ec == len(self._cols) - 2:
            x_max = self._cols[-1]
        else:
            x_max = self._cols[ec + 1]
        return np.array([y_min, x_min, y_max, x_max])

    def _cell_box(self, idx) -> np.ndarray:
        sr = self._cell_start_row[idx]
        sc = self._cell_start_col[idx]
        er = self._cell_end_row[idx]
        ec = self._cell_end_col[idx]
        return self._box(sr, sc, er, ec)

    def _cell_area(self, idx) -> int:
        """return cell <idx> area"""
        y_min, x_min, y_max, x_max = self._cell_box(idx)
        return (y_max - y_min) * (x_max - x_min)

    def _remove_cell(self, idx) -> None:
        del self._cell_start_row[idx]
        del self._cell_start_col[idx]
        del self._cell_end_row[idx]
        del self._cell_end_col[idx]

    def _add_cell(self, sr, sc, er, ec) -> int:
        new_idx = len(self._cell_start_row)
        self._cell_start_row.append(sr)
        self._cell_start_col.append(sc)
        self._cell_end_row.append(er)
        self._cell_end_col.append(ec)
        return new_idx

    def __len__(self) -> int:
        """Number of cells"""
        return len(self._cell_start_row)

    def num_rows(self) -> int:
        return len(self._rows) - 1

    def num_cols(self) -> int:
        return len(self._cols) - 1

    def scale(
        self, factor: Union[float, Sequence], max_shape: Optional[Sequence] = None
    ) -> None:
        """
        factor - float or pair of floats (h, w)
        """
        if isinstance(factor, (tuple, list)) and len(factor) == 2:
            h, w = factor
            h = float(h)
            w = float(w)
        else:
            h = float(factor)
            w = float(factor)
        self._rows = self._rows.astype(float) * h
        self._cols = self._cols.astype(float) * w
        if not self._normalised:
            self._rows = self._rows.astype(int)
            self._cols = self._cols.astype(int)
        if max_shape:
            max_h, max_w = max_shape
            self._rows = self._rows.clip(max=max_h)
            self._cols = self._cols.clip(max=max_w)

    def normalise(self, height: int, width: int):
        if not self._normalised:
            self._normalised = True
            self.scale((1.0 / height, 1.0 / width))

    def translate(self, x, y) -> None:
        """
        translate by x, y
        """
        self._rows += y
        self._cols += x

    def cellboxes(self) -> Generator[np.ndarray, None, None]:
        for i in range(len(self._cell_start_row)):
            yield self._cell_box(i)

    def rowboxes(self) -> Generator[np.ndarray, None, None]:
        for r in range(len(self._rows) - 1):
            yield np.array(
                [self._rows[r], self._cols[0], self._rows[r + 1], self._cols[-1]]
            )

    def colboxes(self) -> Generator[np.ndarray, None, None]:
        for c in range(len(self._cols) - 1):
            yield np.array(
                [self._rows[0], self._cols[c], self._rows[-1], self._cols[c + 1]]
            )

    def cells(self) -> Generator[np.ndarray, None, None]:
        yield from self.cellboxes()
