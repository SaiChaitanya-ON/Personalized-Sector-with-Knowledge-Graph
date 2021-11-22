from typing import Optional, Tuple

import numpy as np


def _iou_with_boxes(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    computes iou of multiple pairs of boxes in the cross product of boxes_a and boxes_b
    and returns the list of overlapping regions for convenience
    boxes are ymin, xmin, ymax, xmax
    boxes_a: [na x 4] array
    boxes_b: [nb x 4] array
    returns [na x nb] array, the all-pairs ious
    """
    na = len(boxes_a)
    nb = len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros(shape=(na, nb)), np.zeros(shape=(na, nb, 4))
    boxes_a = np.expand_dims(boxes_a, axis=1)  # na x 1 x 4
    boxes_b = np.expand_dims(boxes_b, axis=0)  # 1 x nb x 4
    # calculate the intersection area
    # all of following will be na x nb
    x_a = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    y_a = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    x_b = np.minimum(boxes_a[..., 3], boxes_b[..., 3])
    y_b = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    intersection_area: np.ndarray = np.maximum(0.0, x_b - x_a) * np.maximum(
        0.0, y_b - y_a
    )
    # calculate the union area
    box_a_area = (boxes_a[..., 2] - boxes_a[..., 0]) * (
        boxes_a[..., 3] - boxes_a[..., 1]
    )
    box_b_area = (boxes_b[..., 2] - boxes_b[..., 0]) * (
        boxes_b[..., 3] - boxes_b[..., 1]
    )
    union_area: np.ndarray = box_a_area + box_b_area - intersection_area
    # calculate iou, set iou of any union_area <= 0.0 to 0
    intersection_area[union_area <= 0.0] = 0.0
    union_area[union_area <= 0.0] = 1.0
    # np.divide has no signature
    return intersection_area / union_area, np.stack([y_a, x_a, y_b, x_b], axis=-1)


def _get_max_iou(ious: np.ndarray) -> Tuple[int, int, float]:
    """
    Given a 2D array of ious, find the (off-diagonal) maximum iou
    and return its indices
    """
    assert ious.ndim == 2
    for i in np.arange(ious.shape[0]):
        ious[i, i] = 0.0
    i, j = np.unravel_index(ious.argmax(), ious.shape)
    max_iou = ious[i, j]
    return i, j, max_iou


def _handle_backoff_axis(
    xmin0: float, xmax0: float, xmin1: float, xmax1: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    given two overlapping regions (xmin0, xmax0) and (xmin1, xmax1)
    (this must be guaranteed to hold, since it's only called on boxes with iou > 0)
    back off the first region (xmin0, xmax0), and keep the second region
    (xmin1, xmax1) fixed, according to the following rules:

    - if (xmin0, xmax0) is entirely contained in (xmin1, xmax1), return None (x0
    to disappear)
    - if partial overlap on either side, backoff of the overlapping side
    - if (xmin1, xmax1) is entirely contained in (xmin0, xmax0) backoff to the side
    that would result in the smallest distance backed off, in the event of a tie,
    preferentially backoff the 'max' side (the right / bottom side)
    """
    xmin_in = xmin1 <= xmin0 <= xmax1
    xmax_in = xmin1 <= xmax0 <= xmax1
    if xmin_in and xmax_in:
        return None, None, None

    xminbo, xmaxbo = xmin0, xmax0
    min_backoff = xmax1 - xmin0
    max_backoff = xmax0 - xmin1
    if xmin_in:
        xminbo = xmax1
        backoff_dist = min_backoff
    elif xmax_in:
        xmaxbo = xmin1
        backoff_dist = max_backoff
    else:  # both are outside
        assert min_backoff > 0.0 and max_backoff > 0.0
        if min_backoff > max_backoff:
            xmaxbo = xmin1
            backoff_dist = max_backoff
        else:
            xminbo = xmax1
            backoff_dist = min_backoff

    return xminbo, xmaxbo, backoff_dist


def _backoff_region(
    to_back_off: np.ndarray, to_stay: np.ndarray
) -> Optional[np.ndarray]:
    """
    Given a region to back off to_back_off, and a region that overlaps with it to_stay
    compute the backed off version of to_back_off, assuming it backs off along either
    the x or y axes, choose the axis with the smallest backed off distance to do
    the backoff

    in the event of a tie, preferentially back off y axis
    """
    ymin0, xmin0, ymax0, xmax0 = to_back_off
    ymin1, xmin1, ymax1, xmax1 = to_stay

    yminbo, xminbo, ymaxbo, xmaxbo = ymin0, xmin0, ymax0, xmax0

    _xminbo, _xmaxbo, x_backoff_dist = _handle_backoff_axis(xmin0, xmax0, xmin1, xmax1)
    _yminbo, _ymaxbo, y_backoff_dist = _handle_backoff_axis(ymin0, ymax0, ymin1, ymax1)

    if x_backoff_dist is None and y_backoff_dist is None:
        return None
    elif y_backoff_dist is None:
        xminbo = _xminbo
        xmaxbo = _xmaxbo
    elif x_backoff_dist is None:
        yminbo = _yminbo
        ymaxbo = _ymaxbo
    else:
        if x_backoff_dist < y_backoff_dist:
            xminbo = _xminbo
            xmaxbo = _xmaxbo
        else:
            yminbo = _yminbo
            ymaxbo = _ymaxbo

    return np.array([yminbo, xminbo, ymaxbo, xmaxbo])


def _find_max_overlap_boxes(regions: np.ndarray) -> Tuple[int, int, float, np.ndarray]:
    """
    Given a list of bpxes, find the pair of boxes that maximally overlap (largest iou)
    """
    ious, boxes = _iou_with_boxes(regions, regions)
    i, j, max_iou = _get_max_iou(ious)
    overlap_box = boxes[i, j]
    return i, j, max_iou, overlap_box


def backoff_regions(
    regions: np.ndarray, confidences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    for a list of regions, and a list of confidences for each region, choose a pair of boxes
    that maximally overlap (iou > 0), and back-off the box that has a lower confidence, using
    _backoff_region. Repeat until no boxes overlap each other
    """

    regions = regions.copy()
    confidences = confidences.copy()

    assert regions.ndim == 2 and confidences.ndim == 1
    assert regions.shape[0] == confidences.shape[0] and regions.shape[1] == 4
    assert np.all(regions[:, 2] >= regions[:, 0]) and np.all(
        regions[:, 3] >= regions[:, 1]
    ), "requirement not satisfied ymax > ymin and xmax > xmin"

    if regions.shape[0] == 0:
        return regions, confidences

    # find maximum overlap areas to backoff
    i, j, max_iou, _ = _find_max_overlap_boxes(regions)

    while max_iou > 0.0:
        # choose which box to stay / backoff
        to_back_off = i
        to_stay = j
        if confidences[i] > confidences[j]:
            to_back_off = j
            to_stay = i

        # perform back off and update regions
        backed_off_region = _backoff_region(regions[to_back_off], regions[to_stay])
        if backed_off_region is not None:
            regions[to_back_off] = backed_off_region
        else:
            # remove the region
            regions = np.concatenate(
                [regions[:to_back_off], regions[to_back_off + 1 :]], axis=0
            )
            confidences = np.concatenate(
                [confidences[:to_back_off], confidences[to_back_off + 1 :]], axis=0
            )

        # find new maximum overlap areas to backoff
        i, j, max_iou, _ = _find_max_overlap_boxes(regions)

    return regions, confidences
