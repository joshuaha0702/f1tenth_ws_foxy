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
    """에지 검출 기반 국소 선형 보간 라이다 2중벽(외벽 투과 및 월담) 제거 필터 (Zero-Lag).

    - 롤링/피칭 시 15~20cm 폼 배리어 위로 광선이 넘어가 3.5m~18m의 연구실 외벽을 때리는
      월담 펄스(_|==|_) 이상치를 에지 검출로 감지.
    - Positive Edge(_|) 시작점과 Negative Edge(|_) 종료점 사이의 튀어버린 외벽 빔들을
      양단 정상 배리어 거리(r_left, r_right)로 국소 선형 보간(Linear Interpolation)하여 즉시 복원.
    - 시간 평활화(EMA)를 사용하지 않아 위상지연이 0ms이며,
      완만한 실제 트랙 코너 및 직선로는 에지가 발생하지 않으므로 원형 그대로 100% 보존.
    """

    def __init__(self, num_bins=NUM_FEATURES, fov_min=FOV_MIN, fov_max=FOV_MAX,
                 a=6.0, b=2.2, alpha=0.85, jump_thresh=1.2, max_barrier_dist=3.2,
                 min_overflow_dist=4.5, max_obstacle_width=30, **kwargs):
        self.num_bins = num_bins
        self.fov_min = fov_min
        self.fov_max = fov_max
        self.jump_thresh = float(jump_thresh)              # 배리어 월담 스텝 점프 임계값 (m)
        self.max_barrier_dist = float(max_barrier_dist)    # 배리어로 인정 가능한 최대 거리 (m)
        self.min_overflow_dist = float(min_overflow_dist)  # 월담 외벽으로 인정할 최소 도달 거리 (m)
        self.max_obstacle_width = int(max_obstacle_width)  # 선행차/장애물로 판단할 최대 빔 너비
        self.angles = np.linspace(fov_min, fov_max, num_bins)

    def reset(self):
        """에피소드 시작/정지 시 필터 상태 초기화 (단일 프레임 처리로 무상태)."""
        pass

    def _is_narrow_obstacle(self, filt: np.ndarray, idx: int, dist_tol: float = 0.4) -> bool:
        """Condition 3: idx가 좁은 장애물(선행차 등 <= max_obstacle_width 빔)에 속하는지 판별."""
        n = len(filt)
        r = filt[idx]
        left = idx
        while left > 0 and abs(filt[left - 1] - r) <= dist_tol:
            left -= 1
        right = idx
        while right < n - 1 and abs(filt[right + 1] - r) <= dist_tol:
            right += 1
        width = right - left + 1
        is_trough = (left > 0 and filt[left - 1] - r > 0.8) or (right < n - 1 and filt[right + 1] - r > 0.8)
        return (width <= self.max_obstacle_width) and is_trough

    def filter(self, scan_features: np.ndarray) -> np.ndarray:
        """360-feature 라이다 배열을 입력받아 월담 펄스를 에지 보간하여 반환 (Zero-Lag)."""
        scan = np.asarray(scan_features, dtype=np.float64)
        filt = np.copy(scan)
        n = len(filt)

        # 1. Forward Scan: Positive Edge (_|) 감지 후 Negative Edge (|_) 탐색
        i = 1
        while i < n - 1:
            # Condition 3 우선 판단: 점프 직전 값이 좁은 장애물(앞차 등 <= 30빔)이면 스킵
            if self._is_narrow_obstacle(filt, i - 1):
                i += 1
                continue

            # Condition 1 판단: 배리어(< max_barrier_dist)에서 일정 거리(>= min_overflow_dist) 이상 외벽으로 폭등 시
            if (filt[i] - filt[i - 1] > self.jump_thresh
                    and filt[i - 1] < self.max_barrier_dist
                    and filt[i] >= self.min_overflow_dist):
                start_idx = i
                r_left = filt[i - 1]
                end_idx = None
                r_right = r_left

                # 이후 하강 에지 (외벽 -> 배리어 복귀) 탐색
                for j in range(i, n - 1):
                    if filt[j] - filt[j + 1] > self.jump_thresh and filt[j + 1] < self.max_barrier_dist:
                        # 하강 도착점이 좁은 장애물(앞차)이면 배리어 복귀가 아니므로 무시
                        if self._is_narrow_obstacle(filt, j + 1):
                            continue
                        end_idx = j
                        r_right = filt[j + 1]
                        break

                if end_idx is not None:
                    # _|==|_ 펄스 구간 발견: 양단 배리어 값으로 국소 선형 보간
                    steps = end_idx - start_idx + 1
                    filt[start_idx:end_idx + 1] = np.linspace(r_left, r_right, steps)
                    i = end_idx + 1
                else:
                    # 배열 끝(후방)까지 외벽이 이어진 경우: r_left 값으로 평탄화
                    filt[start_idx:] = r_left
                    break
            else:
                i += 1

        # 2. Backward Scan: 배열 시작점(우측 후방)부터 외벽으로 시작하여 배리어로 떨어지는 경우 처리
        for j in range(0, n - 1):
            if filt[j] - filt[j + 1] > self.jump_thresh and filt[j + 1] < self.max_barrier_dist:
                if not self._is_narrow_obstacle(filt, j + 1) and filt[0] >= self.min_overflow_dist:
                    filt[0:j + 1] = filt[j + 1]
                break

        return filt

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

