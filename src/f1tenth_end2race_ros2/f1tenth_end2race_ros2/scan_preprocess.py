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


class SpatioTemporalScanFilter:
    """공간-시간 연속성 기반 라이다 2중벽(외벽 투과 및 코너 튐) 제거 필터.

    - 롤링/피칭 시 트랙 배리어(15~20cm) 위로 광선이 넘어가 3.5m~18m의 연구실 건물 외벽을 때리는
      2중벽(Double Wall) 이상치를 감지하여 인접한 실제 배리어 라인으로 연속 복원한다.
    - 정면(±10° 이내) 직선로 시야(최대 6m)는 100% 보존하여 최고속 주행을 유지하고,
      측면 및 전방 우측/좌측 코너(-10° ~ -100°)의 누출 빔만 핀포인트로 필터링한다.
    - 벽이나 장애물이 다가오는 신호(closer)는 지연 시간 0ms로 즉각 반영(Zero Lag)한다.
    """

    def __init__(self, num_bins=NUM_FEATURES, fov_min=FOV_MIN, fov_max=FOV_MAX,
                 a=6.0, b=2.2, alpha=0.85):
        self.num_bins = num_bins
        self.fov_min = fov_min
        self.fov_max = fov_max
        self.a = float(a)          # 전방 최대 시야 포락선 장축 (m)
        self.b = float(b)          # 측면 트랙 한계 포락선 단축 (m)
        self.alpha = float(alpha)  # 시간적 평활 가중치

        # 360 각도 및 타원형 포락선 사전 계산
        self.angles = np.linspace(fov_min, fov_max, num_bins)
        cos_a = np.cos(self.angles)
        sin_a = np.sin(self.angles)
        self.ellipse_r = (self.a * self.b) / np.sqrt((self.b * cos_a) ** 2 + (self.a * sin_a) ** 2)

        # 전방 코너 및 측면 검사 마스크 (±10° ~ ±100°)
        self.ext_right = (self.angles < -0.1745) & (self.angles > -1.7453)
        self.ext_left = (self.angles > 0.1745) & (self.angles < 1.7453)

        self.prev_filtered = None

    def reset(self):
        """에피소드 시작/정지 시 이전 필터 상태 초기화."""
        self.prev_filtered = None

    def filter(self, scan_features: np.ndarray) -> np.ndarray:
        """360-feature 라이다 배열을 입력받아 2중벽 이상치를 제거한 배열을 반환."""
        scan_features = np.asarray(scan_features, dtype=np.float64)
        if self.prev_filtered is None:
            self.prev_filtered = np.copy(scan_features)
            return np.copy(scan_features)

        filtered = np.copy(scan_features)

        for mask in (self.ext_right, self.ext_left):
            # 1. [공간 참조]: 같은 프레임 내에서 실제 배리어(< ellipse_r)를 정상 감지 중인 빔 탐색
            valid_b = filtered[mask & (filtered < self.ellipse_r)]
            if len(valid_b) >= 10:
                ref_b = float(np.median(valid_b))
            else:
                # 현재 프레임이 전면 월담 시 직전 프레임의 유효 배리어 참조
                prev_valid = self.prev_filtered[mask & (self.prev_filtered < self.ellipse_r)]
                ref_b = float(np.median(prev_valid)) if len(prev_valid) > 0 else 1.5

            # 2. 포락선을 넘어 저 뒤쪽 연구실 벽을 보는 빔 선별
            leak = mask & (filtered > self.ellipse_r)
            if np.any(leak):
                target = np.minimum(ref_b, self.ellipse_r[leak])
                filtered[leak] = self.alpha * self.prev_filtered[leak] + (1.0 - self.alpha) * target

        self.prev_filtered = filtered
        return filtered

    def __call__(self, scan_features: np.ndarray) -> np.ndarray:
        return self.filter(scan_features)


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

