"""LaserScan -> End2Race 360-feature preprocessing, shared by inference and training.

agent_node.py (inference) and extract_bag_csv.py (training CSVs) must turn a scan into
the same 360 values; they used to pool beams differently, so this module is the single
implementation. Numpy only, so the bag extractor can import it without ROS.

Each beam goes, by its angle, into one of `num_bins` equal bins over [fov_min, fov_max]
(no overlap); a bin keeps the minimum range of its beams (closest obstacle). Beams
outside the FOV are dropped, and a bin with no beam stays at max_range.
"""

import numpy as np

NUM_FEATURES = 360
FOV_MIN = -2.35619   # simulator 270 deg LiDAR
FOV_MAX = 2.35619
MAX_RANGE = 30.0


def build_scan_mapping(n_beams, angle_min, angle_increment,
                       fov_min=FOV_MIN, fov_max=FOV_MAX, num_bins=NUM_FEATURES):
    """Return (bin index of each in-FOV beam, in-FOV mask, number of covered bins)."""
    angles = angle_min + np.arange(n_beams) * angle_increment
    edges = np.linspace(fov_min, fov_max, num_bins + 1)
    bin_idx = np.digitize(angles, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < num_bins)
    covered = np.unique(bin_idx[valid]).size
    return bin_idx[valid], valid, covered


def pool_scan(ranges, mapping, num_bins=NUM_FEATURES, max_range=MAX_RANGE):
    """Min-pool raw ranges into num_bins features with a mapping from build_scan_mapping."""
    bin_idx, valid, _ = mapping
    ranges = np.asarray(ranges, dtype=np.float64)
    ranges = np.where(np.isinf(ranges), max_range, ranges)
    ranges = np.where(np.isnan(ranges), 0.0, ranges)
    out = np.full(num_bins, max_range, dtype=np.float64)
    np.minimum.at(out, bin_idx, ranges[valid])
    return out
