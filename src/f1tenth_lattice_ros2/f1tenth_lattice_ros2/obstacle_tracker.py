import math
import numpy as np


class SingleObstacleTrack:
    """단일 장애물 추적을 위한 2D Constant Velocity (CV) 칼만 필터 트랙"""

    def __init__(self, track_id: int, init_x: float, init_y: float, init_theta: float = 0.0,
                 pos_std: float = 0.1, vel_std: float = 1.0, proc_accel_std: float = 1.5):
        self.track_id = track_id

        # 상태 벡터: [x, y, vx, vy]
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float64)

        # 오차 공분산 행렬
        self.P = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2]).astype(np.float64)

        # 센서 측정 노이즈 공분산 (LiDAR 클러스터 중심점 오차 ~0.08m)
        self.R = np.diag([0.08 ** 2, 0.08 ** 2]).astype(np.float64)

        # 프로세스 노이즈 모델링 파라미터 (가속도 노이즈 표준편차)
        self.proc_accel_std = proc_accel_std

        # 헤딩 각도 (rad)
        self.theta = init_theta

        # 생명주기 관리 변수
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self, dt: float):
        if dt <= 1e-4:
            return

        # 상태 전이 행렬 F
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 연속 시간 백색 가속도 노이즈 모델의 프로세스 노이즈 Q
        q = self.proc_accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = q * np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2]
        ], dtype=np.float64)

        # 예측 단계
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.age += 1
        self.time_since_update += 1

    def update(self, meas_x: float, meas_y: float, default_heading: float = 0.0):
        z = np.array([meas_x, meas_y], dtype=np.float64)

        # 측정 행렬 H
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        self.hits += 1
        self.time_since_update = 0

        # 헤딩 업데이트: 속도가 0.2m/s 이상이면 이동 방향을 헤딩으로 채택, 아니면 default_heading 유지
        speed = math.hypot(self.x[2], self.x[3])
        if speed > 0.2:
            self.theta = math.atan2(self.x[3], self.x[2])
        elif default_heading != 0.0:
            self.theta = default_heading


class MultiObstacleTracker:
    """다중 장애물 추적 및 속도 벡터 추정기 (Multi-Object Kalman Filter Tracker)"""

    def __init__(self,
                 max_match_dist: float = 1.0,
                 min_hits_to_confirm: int = 3,
                 max_misses_to_delete: int = 5):
        self.max_match_dist = max_match_dist
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_misses_to_delete = max_misses_to_delete

        self.tracks = []
        self.next_track_id = 1
        self.last_stamp = None

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
        self.last_stamp = None

    def update(self, detected_points: np.ndarray, current_stamp_sec: float) -> list:
        """
        detected_points: shape (M, 3) -> [x, y, default_heading]
        current_stamp_sec: 현재 ROS 시간(초)

        Returns:
            확인된(confirmed) 장애물 정보 리스트:
            [
                {
                    'id': int,
                    'x': float, 'y': float, 'theta': float,
                    'vx': float, 'vy': float, 'speed': float
                }, ...
            ]
        """
        if self.last_stamp is None:
            dt = 0.1
        else:
            dt = current_stamp_sec - self.last_stamp
            # 시간이 뒤로 가거나(시뮬레이션 리셋) 비정상적으로 크면 리셋
            if dt < 0.0 or dt > 1.0:
                self.reset()
                dt = 0.1
        self.last_stamp = current_stamp_sec

        # 1. 기존 트랙들의 예측 (Predict)
        for tr in self.tracks:
            tr.predict(dt)

        num_tracks = len(self.tracks)
        num_meas = len(detected_points)

        # 2. 데이터 연관 (Data Association) - Greedy Nearest Neighbor
        matched_tracks = set()
        matched_meas = set()

        if num_tracks > 0 and num_meas > 0:
            cost_matrix = np.zeros((num_tracks, num_meas), dtype=np.float64)
            for i, tr in enumerate(self.tracks):
                for j in range(num_meas):
                    dist = math.hypot(tr.x[0] - detected_points[j, 0],
                                      tr.x[1] - detected_points[j, 1])
                    cost_matrix[i, j] = dist

            # 가장 거리가 가까운 쌍부터 매칭
            sorted_indices = np.dstack(np.unravel_index(np.argsort(cost_matrix.ravel()), (num_tracks, num_meas)))[0]
            for tr_idx, m_idx in sorted_indices:
                if tr_idx in matched_tracks or m_idx in matched_meas:
                    continue
                if cost_matrix[tr_idx, m_idx] <= self.max_match_dist:
                    matched_tracks.add(tr_idx)
                    matched_meas.add(m_idx)
                    default_h = float(detected_points[m_idx, 2]) if detected_points.shape[1] > 2 else 0.0
                    self.tracks[tr_idx].update(
                        float(detected_points[m_idx, 0]),
                        float(detected_points[m_idx, 1]),
                        default_h
                    )

        # 3. 매칭되지 않은 신규 측정값 -> 새 트랙 등록
        for j in range(num_meas):
            if j not in matched_meas:
                default_h = float(detected_points[j, 2]) if detected_points.shape[1] > 2 else 0.0
                new_track = SingleObstacleTrack(
                    track_id=self.next_track_id,
                    init_x=float(detected_points[j, 0]),
                    init_y=float(detected_points[j, 1]),
                    init_theta=default_h
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # 4. 수명이 다한 트랙 정리 (Delete stale tracks)
        alive_tracks = []
        for tr in self.tracks:
            if tr.time_since_update <= self.max_misses_to_delete:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        # 5. 활성화/확인(Confirmed)된 트랙 결과 반환
        results = []
        for tr in self.tracks:
            if tr.hits >= self.min_hits_to_confirm:
                speed = math.hypot(tr.x[2], tr.x[3])
                results.append({
                    'id': tr.track_id,
                    'x': float(tr.x[0]),
                    'y': float(tr.x[1]),
                    'theta': float(tr.theta),
                    'vx': float(tr.x[2]),
                    'vy': float(tr.x[3]),
                    'speed': float(speed)
                })

        return results
