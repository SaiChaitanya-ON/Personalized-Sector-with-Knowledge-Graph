import numpy as np
import pytest

from onai.ml.tables.table import Table


@pytest.fixture
def table():
    rows = (
        np.array([0, 10, 20]),
        np.array([0, 10, 20]),
        [
            np.array([0, 0, 10, 10]),
            np.array([0, 10, 10, 20]),
            np.array([10, 0, 20, 10]),
            np.array([10, 10, 20, 20]),
        ],
    )
    cols = (
        np.array([10, 20, 30]),
        np.array([40, 50, 60]),
        [
            np.array([10, 40, 20, 50]),
            np.array([10, 50, 20, 60]),
            np.array([20, 40, 30, 50]),
            np.array([20, 50, 30, 60]),
        ],
    )

    return Table(rows=rows, cols=cols)
