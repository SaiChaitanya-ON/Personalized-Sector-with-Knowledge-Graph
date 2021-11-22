import numpy as np

from onai.ml.tables.detection_post_process import (
    _backoff_region,
    _find_max_overlap_boxes,
    backoff_regions,
)


def test_backoff_region_private():
    """
    tests the private _backoff_region method, given a specified
    region to backoff
    """
    region_bo = [-1.0, -1.0, 1.0, 1.0]
    region_stay = [-1.0, -1.0, 1.0, 1.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert region_bo_new is None

    # region bo entirely enclosed, it disappears
    region_bo = [-0.5, -0.5, 0.5, 0.5]
    region_stay = [-1.0, -1.0, 1.0, 1.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert region_bo_new is None

    # region_stay entirely enclosed, bigger region backs off preferentially on y
    region_bo = [-1.0, -1.0, 1.0, 1.0]
    region_stay = [-0.5, -0.5, 0.5, 0.5]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([0.5, -1.0, 1.0, 1.0]))

    region_bo = [-1.0, -0.5, 1.0, 0.5]
    region_stay = [-0.5, -1.0, 0.5, 1.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([0.5, -0.5, 1.0, 0.5]))

    region_bo = [-1.0, -0.5, 1.5, 0.5]
    region_stay = [-0.5, -1.0, 0.5, 1.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([0.5, -0.5, 1.5, 0.5]))

    region_bo = [0.0, -0.5, 1.0, 0.5]
    region_stay = [-0.5, -1.0, 0.5, 1.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([0.5, -0.5, 1.0, 0.5]))

    # a tie between x and y, but y selectively backs-off
    region_bo = [-1.0, -1.0, 1.0, 1.0]
    region_stay = [0.0, 0.0, 2.0, 2.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([-1.0, -1.0, 0.0, 1.0]))

    # y a little more overlap than x, x backs off
    region_bo = [-1.0, -1.0, 1.1, 1.0]
    region_stay = [0.0, 0.0, 2.0, 2.0]
    region_bo = np.array(region_bo)
    region_stay = np.array(region_stay)
    region_bo_new = _backoff_region(region_bo, region_stay)
    assert np.allclose(region_bo_new, np.array([-1.0, -1.0, 1.1, 0.0]))


def test_backoff_regions():
    """
    tests the full backoff region algo
    """
    regions = np.array([[-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])
    confs = np.array([0.0, 1.0])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 1
    assert np.allclose(regions_new[0], regions[1]) and confs_new[0] == 1.0

    # region [0] entirely enclosed, and it has smaller confidence, it disappears
    regions = np.array([[-0.5, -0.5, 0.5, 0.5], [-1.0, -1.0, 1.0, 1.0]])
    confs = np.array([0.0, 1.0])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 1
    assert np.allclose(regions_new[0], regions[1]) and confs_new[0] == 1.0

    # region [1] entirely enclosed, but it has higher confidence, bigger region backs off
    confs = np.array([1.0, 0.0])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 2
    assert np.allclose(confs_new, confs)
    assert np.allclose(regions_new[0], regions[0])
    assert np.allclose(regions_new[1], np.array([0.5, -1.0, 1.0, 1.0]))

    # region [0] backs off in y
    regions = np.array([[-1.0, -0.5, 1.0, 0.5], [-0.5, -1.0, 0.5, 1.0]])
    confs = np.array([0.5, 0.8])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 2
    assert np.allclose(confs_new, confs)
    assert np.allclose(regions_new[0], np.array([0.5, -0.5, 1.0, 0.5]))
    assert np.allclose(regions_new[1], regions[1])

    # region [1] backs off in x
    regions = np.array([[-1.0, -0.5, 1.0, 0.5], [-0.5, -1.0, 0.5, 1.0]])
    confs = np.array([0.8, 0.5])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 2
    assert np.allclose(confs_new, confs)
    assert np.allclose(regions_new[0], regions[0])
    assert np.allclose(regions_new[1], np.array([-0.5, 0.5, 0.5, 1.0]))

    # a tie between x and y, but y selectively backs-off
    regions = np.array([[-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    confs = np.array([0.5, 0.8])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == len(confs_new) == 2
    assert np.allclose(confs_new, confs)
    assert np.allclose(regions_new[0], np.array([-1.0, -1.0, 0.0, 1.0]))
    assert np.allclose(regions_new[1], regions[1])

    # 2 disjoint sets of 2 overlapping boxes
    regions = np.array([[-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    regions_offset = regions + 5.0
    regions = np.concatenate([regions, regions_offset], axis=0)
    confs = np.array([0.5, 0.8, 0.5, 0.8])
    regions_new, confs_new = backoff_regions(regions, confs)
    regions_new[2:] -= 5.0

    assert np.allclose(confs_new, confs)
    assert np.allclose(regions_new[0], np.array([-1.0, -1.0, 0.0, 1.0]))
    assert np.allclose(regions_new[1], regions[1])
    assert np.allclose(regions_new[[0, 1]], regions_new[[2, 3]])

    # 2 disjoint sets of 2 overlapping boxes, 2 boxes disappear
    regions = np.array([[-0.5, -0.5, 0.5, 0.5], [-1.0, -1.0, 1.0, 1.0]])
    regions_offset = regions + 5.0
    regions = np.concatenate([regions, regions_offset], axis=0)
    confs = np.array([0.5, 0.8, 0.5, 0.8])
    regions_new, confs_new = backoff_regions(regions, confs)
    assert len(regions_new) == 2
    regions_new[1] -= 5.0

    assert np.allclose(regions_new[0], regions[1])
    assert np.allclose(regions_new[1], regions[1])
    assert np.allclose(confs_new, 0.8)


def test_stress_backoff_algo():
    """
    stress tests the algorithm by generating a random bunch of different boxes,
    and checking that the maximum pairwise iou is always zero
    """
    num_reps = 50
    bound = 2.0
    num_boxes = 5

    for rep in range(num_reps):
        top = np.random.uniform(-bound, bound, size=(num_boxes,))
        left = np.random.uniform(-bound, bound, size=(num_boxes,))
        bot = np.random.uniform(top, bound)  # ymax
        right = np.random.uniform(left, bound)  # xmax
        random_boxes = np.vstack([top, left, bot, right]).transpose()
        confs = np.random.uniform(0.0, 1.0, size=(num_boxes,))

        random_boxes_new, confs_new = backoff_regions(random_boxes, confs)
        _, _, max_iou, _ = _find_max_overlap_boxes(random_boxes_new)
        assert max_iou == 0.0
