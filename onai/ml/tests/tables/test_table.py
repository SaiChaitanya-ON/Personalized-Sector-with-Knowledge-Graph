import numpy as np
import pytest

from onai.ml.tables.table import Table


@pytest.mark.parametrize(
    "row, col, celllist",
    [
        (
            np.array([0, 10, 20]),
            np.array([0, 10, 20]),
            [
                np.array([0, 0, 10, 10]),
                np.array([0, 10, 10, 20]),
                np.array([10, 0, 20, 10]),
                np.array([10, 10, 20, 20]),
            ],
        ),
        (
            np.array([10, 20, 30]),
            np.array([40, 50, 60]),
            [
                np.array([10, 40, 20, 50]),
                np.array([10, 50, 20, 60]),
                np.array([20, 40, 30, 50]),
                np.array([20, 50, 30, 60]),
            ],
        ),
        (
            np.array([0, 10, 20]),
            np.array([0, 1, 2]),
            [
                np.array([0, 0, 10, 1]),
                np.array([0, 1, 10, 2]),
                np.array([10, 0, 20, 1]),
                np.array([10, 1, 20, 2]),
            ],
        ),
    ],
    ids=[
        "2rows 2cols 4cell",
        "2rows 2cols 4cell offset",
        "2rows 2cols 4cell 1px width",
    ],
)
def test_table_grid_creation_success(row, col, celllist):
    tbl = Table(row, col)

    cells = list(tbl.cellboxes())
    # Put them into reading order -- by increasing rows starts and
    # then by increasing column starts
    cells = sorted(cells, key=lambda x: x[1])
    cells = sorted(cells, key=lambda x: x[0])

    assert len(cells) == len(celllist)
    for i, (c1, c2) in enumerate(zip(cells, celllist)):
        assert (c1 == c2).all()


def test_table_rowcol_boxes(table):
    cells = list(table.cellboxes())

    rowboxes = list(table.rowboxes())
    colboxes = list(table.colboxes())

    assert len(rowboxes) * len(colboxes) == len(cells)
