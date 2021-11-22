from onai.ml.peers.feature_extractor import _LAST_REVENUE_WINDOWS

COLS = [
    "country_overlap",
    "weighted_symmetric_diff",
    "weighted_intersection",
    "weighted_intersection_negative_sample",
    "weighted_intersection_negative_sample_tail_end",
    "peer_diff",
    "is_subsidiary",
    "no_last_revenue_diff",
    "no_last_ebitda_diff",
    "no_last_ebit_diff",
]

for window_size in _LAST_REVENUE_WINDOWS:
    COLS.append(f"last_revenue_diff_{window_size:.2f}")
    COLS.append(f"last_ebitda_diff_{window_size:.2f}")
    COLS.append(f"last_ebit_diff_{window_size:.2f}")
