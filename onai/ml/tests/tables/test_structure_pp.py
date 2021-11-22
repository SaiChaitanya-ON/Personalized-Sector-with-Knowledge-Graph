import numpy as np

from onai.ml.tables.structure_postprocess import table_grid


def test_table_grid_one_peak():
    hm = np.zeros((21, 21)) + 0.5
    hm[:, 10] = 1
    hm[10, :] = 1

    celllist = [
        np.array([0, 0, 10, 10]),
        np.array([0, 10, 10, 20]),
        np.array([10, 0, 20, 10]),
        np.array([10, 10, 20, 20]),
    ]
    tbl = table_grid(hm.mean(axis=0), hm.mean(axis=1))
    cells = list(tbl.cellboxes())
    # Put them into reading order -- by increasing rows starts and
    # then by increasing column starts
    cells = sorted(cells, key=lambda x: x[1])
    cells = sorted(cells, key=lambda x: x[0])

    assert len(cells) == len(celllist)
    for i, (c1, c2) in enumerate(zip(cells, celllist)):
        assert (c1 == c2).all()


def test_table_grid_one_peak_pad_ignore():
    hm = np.zeros((21, 21)) + 0.5
    hm[:, 10] = 1
    hm[10, :] = 1
    hm[0, :] = 10
    hm[-1, :] = 10

    celllist = [
        np.array([0, 0, 10, 10]),
        np.array([0, 10, 10, 20]),
        np.array([10, 0, 20, 10]),
        np.array([10, 10, 20, 20]),
    ]
    tbl = table_grid(hm.mean(axis=0), hm.mean(axis=1))
    cells = list(tbl.cellboxes())
    # Put them into reading order -- by increasing rows starts and
    # then by increasing column starts
    cells = sorted(cells, key=lambda x: x[1])
    cells = sorted(cells, key=lambda x: x[0])

    assert len(cells) == len(celllist)
    for i, (c1, c2) in enumerate(zip(cells, celllist)):
        assert (c1 == c2).all()


def test_table_grid_zeros():
    hm = np.zeros((21, 21))

    celllist = [np.array([0, 0, 20, 20])]
    tbl = table_grid(hm.mean(axis=0), hm.mean(axis=1))
    cells = list(tbl.cellboxes())

    assert len(cells) == len(celllist)
    for i, (c1, c2) in enumerate(zip(cells, celllist)):
        assert (c1 == c2).all()
