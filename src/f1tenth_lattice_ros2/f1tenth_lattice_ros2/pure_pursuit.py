import numpy as np
from numba import njit
from f1tenth_lattice_ros2.planner_utils import (
    nearest_point, intersect_point, get_actuation_PD
)


class PurePursuitPlanner:
    def __init__(self, conf, wpt_path, wb=0.33):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.0

        self.wpt_xind = conf.wpt_xind
        self.wpt_yind = conf.wpt_yind
        self.wpt_vind = conf.wpt_vind
        self.waypoints = None
        self.waypoints_xyv = None
        self._load_waypoints(wpt_path)
        self.wpNum = self.waypoints.shape[0]

        self.minL = conf.minL
        self.maxL = conf.maxL
        self.Lscale = conf.Lscale
        self.minP = conf.minP
        self.maxP = conf.maxP
        self.Pscale = conf.Pscale
        self.D = conf.D
        self.prev_error = 0.0
        self.vel_scale = conf.vel_scale
        self.interpScale = conf.interpScale

        # 실차용 추종 보강(opt-in). 기본값은 꺼짐 = 시뮬 데이터 수집 당시의 추종 동작.
        # 시뮬 config(sim_lattice_config.yaml, maps/*/lattice_config.yaml)는 키를 생략해 기존
        # 라벨 정책을 유지하고, 실차 lattice_config.yaml만 명시적으로 켠다.
        self.adaptive_lookahead = bool(getattr(conf, 'adaptive_lookahead', False))
        self.adaptive_lookahead_gain = float(getattr(conf, 'adaptive_lookahead_gain', 1.8))
        self.adaptive_lookahead_min = float(getattr(conf, 'adaptive_lookahead_min', 0.65))
        self.k_heading = float(getattr(conf, 'k_heading', 0.0))

    def _load_waypoints(self, wpt_path):
        # raceline CSV: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        raw = np.loadtxt(wpt_path, delimiter=';', skiprows=2)
        # reorder to (x, y, v, heading, s)
        self.waypoints = np.vstack(
            (raw[:, 1], raw[:, 2], raw[:, 5], raw[:, 3], raw[:, 0])
        ).T
        self.waypoints_xyv = self.waypoints[:, :3]

    def get_L(self, curr_v):
        return curr_v * (self.maxL - self.minL) / self.Lscale + self.minL

    def plan(self, pose_x, pose_y, pose_theta, curr_v, waypoints):
        L = curr_v * (self.maxL - self.minL) / self.Lscale + self.minL
        L = max(L, self.minL)

        if self.adaptive_lookahead:
            # Adaptive Lookahead: 코너링(조향각)에 따라 lookahead를 적응형 단축
            # 직선(|prev_error| ~ 0): 원래 속도 연동 L 유지 -> 고속 직선 흔들림 방지
            # 코너(|prev_error| 커질수록): L을 단축 -> 코너 호(arc)를 타이트하게 물고 탈출
            steer_factor = 1.0 / (1.0 + self.adaptive_lookahead_gain * abs(self.prev_error))
            L = max(L * steer_factor, self.adaptive_lookahead_min)

        P = self.maxP - curr_v * (self.maxP - self.minP) / self.Pscale
        P = max(min(P, self.maxP), self.minP)

        position = np.array([pose_x, pose_y])
        lookahead_point, new_L, nearest_dist = get_wp_xyv_with_interp(
            L, position, pose_theta, waypoints, waypoints.shape[0], self.interpScale
        )
        self.nearest_dist = nearest_dist

        speed, steering, error = get_actuation_PD(
            pose_theta, lookahead_point, position, new_L, self.wheelbase, self.prev_error, P, self.D,
            self.k_heading
        )
        speed = speed * self.vel_scale
        self.prev_error = error
        return steering, speed


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    nearest_idx = np.argmin(traj_distances)
    nearest_dist = traj_distances[nearest_idx]
    segment_end = nearest_idx

    if traj_distances[-1] < L:
        segment_end = wpNum - 1
    else:
        while traj_distances[segment_end] < L:
            segment_end = (segment_end + 1) % wpNum

    segment_begin = (segment_end - 1 + wpNum) % wpNum
    x_array = np.linspace(waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale)
    y_array = np.linspace(waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale)
    v_array = np.linspace(waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale)
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)

    dx = waypoints[segment_end, 0] - waypoints[segment_begin, 0]
    dy = waypoints[segment_end, 1] - waypoints[segment_begin, 1]
    if np.abs(dx) < 1e-6 and np.abs(dy) < 1e-6:
        target_heading = theta
    else:
        target_heading = np.arctan2(dy, dx)

    return np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp], target_heading)), new_L, nearest_dist
