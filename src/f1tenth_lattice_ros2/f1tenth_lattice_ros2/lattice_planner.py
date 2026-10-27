import os
import logging
import time
import math
import yaml
import numpy as np
from PIL import Image
from numba import njit
from scipy.ndimage import distance_transform_edt as edt
from pyclothoids import Clothoid

import multiprocessing
from f1tenth_lattice_ros2.planner_utils import (
    nearest_point, intersect_point, get_rotation_matrix,
    zero_2_2pi, sample_traj, map_collision, get_vertices, collision,
)
from f1tenth_lattice_ros2.pure_pursuit import PurePursuitPlanner

logger = logging.getLogger(__name__)


def _compute_clothoid_chunk(args):
    pose_x, pose_y, pose_theta, npts, chunk = args
    results = []
    for pt in chunk:
        c = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, pt[0], pt[1], pt[2])
        traj = sample_traj(c, npts, pt[3])
        results.append((traj, np.array(c.Parameters)))
    return results


class LatticePlanner:
    def __init__(self, conf, map_path, wpt_path, wb=0.325):
        self.wheelbase = float(getattr(conf, 'wheelbase', wb))

        # load waypoints: reorder to (x, y, v, heading, s)
        raw = np.loadtxt(wpt_path, delimiter=';', skiprows=2)
        self.waypoints = np.vstack(
            (raw[:, 1], raw[:, 2], raw[:, 5], raw[:, 3], raw[:, 0])
        ).T

        self.lh_grid_rows = 5
        self.sampling_width_max = getattr(conf, 'sampling_width_max', 0.6)
        self.sampling_width_num = getattr(conf, 'sampling_width_num', 13)
        self.sampling_widths = np.linspace(
            -self.sampling_width_max, self.sampling_width_max, num=self.sampling_width_num
        )
        self.traj_num = self.lh_grid_rows * len(self.sampling_widths)
        self.lh_grid_lb = conf.lh_grid_lb
        self.lh_grid_ub = conf.lh_grid_ub
        self.traj_points = conf.traj_points
        self.traj_v_scale = conf.traj_v_scale
        self.s_max = self.waypoints[-1, 4]

        self.shape_cost_funcs = [get_follow_optim_cost]
        self.constant_cost_funcs = [get_map_collision]
        self.selection_func = None

        self.params_num = list(conf.params_num)
        self.params_name = list(conf.params_name)
        self.params_idx = {}
        count = 0
        for name, num in zip(self.params_name, self.params_num):
            self.params_idx[name] = count
            count += num

        self.cost_weights_num = getattr(conf, 'weights_num', 4)
        configured_weights = getattr(conf, 'cost_weights', None)
        if configured_weights is not None:
            self.set_cost_weights(np.array(configured_weights, dtype=float))
        else:
            self.set_cost_weights(self.cost_weights_num)

        self.v_lattice_span = np.linspace(
            conf.traj_v_span_min, conf.traj_v_span_max, conf.traj_v_span_num
        )
        self.v_lattice_num = conf.traj_v_span_num

        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.prev_traj_local = np.zeros((self.traj_points, 2))
        self.prev_opp_pose = np.zeros((1, 2))
        self.goal_grid = None
        self.state_i = None
        self.state_t = None
        self.step_all_cost = {}
        self.all_costs = None
        # 선두(상대) 차량 속도 추정에 쓰는 dt. 노드에서 실제 plan() 호출 간격을
        # 넘겨주면 그 값을 쓰고, 못 받으면 이 기본값으로 폴백한다.
        self.time_interval = conf.tracker_steps * 0.01
        self.opp_dt = self.time_interval
        self.last_s = 0.0
        self.step = 0
        self.last_timing = None

        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=self.wheelbase)
        self.conf = conf

        # --- [수정] 맵 YAML 및 이미지 로딩 경로 예외 처리 보완 ---
        yaml_path = map_path + '.yaml' if not map_path.endswith('.yaml') else map_path
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(map_path, os.path.basename(map_path) + '_map.yaml')

        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
            
        # load map image using the filename specified in yaml
        image_filename = meta['image']
        map_img_path = os.path.join(os.path.dirname(yaml_path), image_filename)
        img = np.array(
            Image.open(map_img_path).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        img[img <= 128.] = 0.
        img[img > 128.] = 255.
        self.map_height = img.shape[0]
        self.map_width = img.shape[1]
        self.map_resolution = meta['resolution']
        self.origin = meta['origin']
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        self.dt = self.map_resolution * edt(img)
        self.map_metainfo = (
            self.orig_x, self.orig_y, self.orig_c, self.orig_s,
            self.map_height, self.map_width, self.map_resolution
        )

        self.scan_num = conf.scan_num
        self.angle_span = np.linspace(-0.75 * np.pi, 0.75 * np.pi, self.scan_num)
        self.ittc_thres = conf.ittc_thres
        self.collision_thres = getattr(conf, 'collision_thres', 0.26)
        self.curve_v_penalty_gain = getattr(conf, 'curve_v_penalty_gain', 2.2)

        # === 선두 차량 추종(following) 모드 설정 ===
        follow_cfg = getattr(conf, 'follow', None)
        self.follow_enabled = False
        self.follow_zones = []
        self.follow_active = False  # 히스테리시스 상태(진입/해제 떨림 방지)
        self.follow_lateral_align = 3.0
        self.follow_desired_gap = 1.2
        self.follow_kp_gap = 0.8
        self.follow_max_speed = 1.0
        self.follow_horizon = 3.0
        if isinstance(follow_cfg, dict):
            zones = follow_cfg.get('zones') or []
            self.follow_zones = [(int(z['min']), int(z['max'])) for z in zones]
            self.follow_lateral_align = float(follow_cfg.get('lateral_align_m', self.follow_lateral_align))
            self.follow_desired_gap = float(follow_cfg.get('desired_gap_m', self.follow_desired_gap))
            self.follow_kp_gap = float(follow_cfg.get('kp_gap', self.follow_kp_gap))
            self.follow_max_speed = float(follow_cfg.get('max_follow_speed', self.follow_max_speed))
            self.follow_horizon = float(follow_cfg.get('horizon_m', self.follow_horizon))
            if self.follow_zones:
                self.follow_enabled = True
                logger.info(
                    f'[follow] zone-based enabled — zones={self.follow_zones}, '
                    f'lateral_align={self.follow_lateral_align}m, '
                    f'desired_gap={self.follow_desired_gap}m, max_speed={self.follow_max_speed}m/s'
                )

        # Multi-core process pool for parallel Clothoid generation
        self.num_workers = min(multiprocessing.cpu_count(), 4)
        self.pool = multiprocessing.Pool(processes=self.num_workers)

    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.terminate()
            self.pool.join()

    def set_cost_weights(self, cost_weights):
        if isinstance(cost_weights, int):
            n = cost_weights
            cost_weights = np.array([1 / n] * n)
        self.cost_weights = cost_weights

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None, dt=None, opp_vels=None):
        self.step += 1
        if waypoints is None:
            waypoints = self.waypoints

        self.opp_dt = dt if (dt is not None and dt > 1e-6) else self.time_interval

        # opp_poses 형식 지원: (N, 5) 형태 [x, y, theta, vx, vy] 또는 (N, 3) + opp_vels
        if opp_poses is not None and len(opp_poses) > 0:
            if opp_poses.shape[1] >= 5 and opp_vels is None:
                opp_vels = opp_poses[:, 3:5]
                opp_poses = opp_poses[:, :3]
            elif opp_poses.shape[1] > 3:
                opp_poses = opp_poses[:, :3]
        else:
            opp_poses = np.empty((0, 3))
            opp_vels = np.empty((0, 2))

        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, t, nearest_i = nearest_point(ego_pose[:2], waypoints[:, 0:2])
        self.state_i = nearest_i
        self.state_t = t

        # === 선두 차량 추종 모드 판정 ===
        follow_traj = self._maybe_follow(ego_pose, opp_poses, waypoints)
        if follow_traj is not None:
            self.best_traj = follow_traj
            self.best_traj_ref_v = float(follow_traj[0, 2])
            self.prev_opp_pose = opp_poses[:, :2] if opp_poses.shape[0] > 0 else np.zeros((1, 2))
            self.last_timing = None
            return self.best_traj, 0.0, 0.0, 0.0, 0.0

        min_L = self.tracker.get_L(velocity)
        lh_grid = np.linspace(
            min_L + self.lh_grid_lb, min_L + self.lh_grid_ub, self.lh_grid_rows
        )
        self.goal_grid = sample_lookahead_square(
            pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, self.sampling_widths
        )

        _t_clothoid0 = time.perf_counter()
        chunks = np.array_split(self.goal_grid, self.num_workers)
        tasks = [(pose_x, pose_y, pose_theta, self.traj_points, chunk) for chunk in chunks if len(chunk) > 0]
        chunk_results = self.pool.map(_compute_clothoid_chunk, tasks)

        all_traj = []
        all_traj_clothoid = []
        for res in chunk_results:
            for traj, clothoid_params in res:
                all_traj.append(traj)
                all_traj_clothoid.append(clothoid_params)

        all_traj = np.array(all_traj)
        all_traj_clothoid = np.array(all_traj_clothoid)
        _t_clothoid1 = time.perf_counter()
        traj_cost, abs_v_cost, collision_cost = self._eval(
            all_traj, all_traj_clothoid, opp_poses, ego_pose, opp_vels=opp_vels
        )
        _t_eval1 = time.perf_counter()
        clothoid_ms = (_t_clothoid1 - _t_clothoid0) * 1000.0
        eval_ms = (_t_eval1 - _t_clothoid1) * 1000.0
        if clothoid_ms + eval_ms > 30.0:
            self.last_timing = (clothoid_ms, eval_ms)
        else:
            self.last_timing = None
        self.all_costs = traj_cost + abs_v_cost + collision_cost

        best_traj_idx = np.argmin(self.all_costs)
        self.best_traj_idx = best_traj_idx
        row_idx, col_idx = divmod(best_traj_idx, self.v_lattice_num)

        best_cost = self.all_costs[row_idx, col_idx]
        best_collision = collision_cost[row_idx, col_idx]

        # === 추월 불가(좁은 길, 벽 충돌 또는 장애물 차단으로 안전 회피 궤적 부재) 시 서행 추종 폴백 ===
        # 벽 충돌 없는(traj_cost < 2500) 궤적 중 하드 장애물 충돌(collision_cost < 600)이 아닌 안전한 궤적이 존재하는지 확인
        safe_trajs = (traj_cost[:, col_idx] < 2500.0) & (collision_cost[:, col_idx] < 600.0)
        has_safe_traj = np.any(safe_trajs)

        # 안전한 회피 궤적이 아예 없고 전방에 장애물이 있는 경우에만 서행 추종 폴백!
        if opp_poses.shape[0] > 0 and not has_safe_traj:
            dyn_follow_traj = self._build_dynamic_follow_traj(ego_pose, opp_poses, waypoints)
            if dyn_follow_traj is not None:
                self.best_traj = dyn_follow_traj
                self.best_traj_ref_v = float(dyn_follow_traj[0, 2])
                self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])
                self.prev_opp_pose = opp_poses[:, :2]
                logger.info(
                    f'[follow] All paths blocked (no safe clear traj, best_cost={best_cost:.1f}). '
                    f'Following leader at safe distance (v={self.best_traj_ref_v:.2f} m/s)'
                )
                return self.best_traj, best_cost, traj_cost[row_idx, col_idx], abs_v_cost[row_idx, col_idx], collision_cost[row_idx, col_idx]

        self.best_traj = all_traj[row_idx].copy()
        self.best_traj_ref_v = self.best_traj[-1, 2]
        self.best_traj[:, 2] *= (self.v_lattice_span[col_idx] * self.traj_v_scale)
        self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])
        self.prev_opp_pose = opp_poses[:, :2] if opp_poses.shape[0] > 0 else np.zeros((1, 2))
        return self.best_traj, best_cost, traj_cost[row_idx, col_idx], abs_v_cost[row_idx, col_idx], collision_cost[row_idx, col_idx]

    def _in_follow_zone(self, idx):
        for lo, hi in self.follow_zones:
            if lo <= idx <= hi:
                return True
        return False

    def _build_dynamic_follow_traj(self, ego_pose, opp_poses, waypoints):
        """추월이 불가능할 때 전방 장애물/차량 뒤에서 안전거리를 유지하며 서행 추종하는 궤적 생성"""
        if opp_poses.shape[0] == 0:
            return None
        ego_i = self.state_i
        opp_xy = opp_poses[0, :2]
        _, _, _, opp_i = nearest_point(opp_xy, waypoints[:, 0:2])

        s_ego = waypoints[ego_i, 4]
        s_opp = waypoints[opp_i, 4]
        gap = s_opp - s_ego
        if gap < 0.0:
            gap += self.s_max

        # 전방 0.45m ~ 3.5m 범위에 있는 전방 장애물에 대해서만 유효
        # (장애물과 나란히 있거나 이미 통과 중인 경우 추종 모드 진입 방지)
        if not (0.45 < gap < 3.5):
            return None

        v_leader = 0.0
        if (self.prev_opp_pose.shape[0] == opp_poses.shape[0]
                and np.sum(np.abs(self.prev_opp_pose)) > 1e-6 and self.opp_dt > 1e-6):
            v_leader = float(np.linalg.norm(opp_xy - self.prev_opp_pose[0]) / self.opp_dt)

        raceline_v_ego = float(waypoints[ego_i, 2])
        if gap < 0.55:
            # 안전거리 미만으로 너무 가까우면 정지하여 추돌 방지
            v_cmd = 0.0
        else:
            # 목표 거리(follow_desired_gap)를 유지하며 선두차 속도에 맞춰 추종
            v_cmd = v_leader + self.follow_kp_gap * (gap - self.follow_desired_gap)
            v_cmd = max(0.5, min(v_cmd, raceline_v_ego, self.follow_max_speed))

        return self._build_follow_traj(ego_i, v_cmd)

    def _maybe_follow(self, ego_pose, opp_poses, waypoints):
        if not self.follow_enabled or opp_poses.shape[0] == 0:
            self.follow_active = False
            return None

        ego_i = self.state_i
        opp_xy = opp_poses[0, :2]
        _, _, _, opp_i = nearest_point(opp_xy, waypoints[:, 0:2])

        s_ego = waypoints[ego_i, 4]
        s_opp = waypoints[opp_i, 4]
        gap = s_opp - s_ego
        if gap < 0.0:
            gap += self.s_max

        in_zone = self._in_follow_zone(ego_i)
        active = in_zone and (0.0 < gap < self.follow_lateral_align)

        if active and not self.follow_active:
            logger.info(f'[follow] ENTER — ego_i={ego_i}, gap={gap:.2f}m')
        elif not active and self.follow_active:
            logger.info(f'[follow] EXIT — ego_i={ego_i}, gap={gap:.2f}m, in_zone={in_zone}')
        self.follow_active = active

        if not active:
            return None

        v_leader = 0.0
        if (self.prev_opp_pose.shape[0] == opp_poses.shape[0]
                and np.sum(np.abs(self.prev_opp_pose)) > 1e-6 and self.opp_dt > 1e-6):
            v_leader = float(np.linalg.norm(opp_xy - self.prev_opp_pose[0]) / self.opp_dt)

        raceline_v_ego = float(waypoints[ego_i, 2])
        v_cmd = v_leader + self.follow_kp_gap * (gap - self.follow_desired_gap)
        v_cmd = max(0.0, min(v_cmd, raceline_v_ego, self.follow_max_speed))

        return self._build_follow_traj(ego_i, v_cmd)

    def _build_follow_traj(self, start_i, v_cmd):
        wp = self.waypoints
        n = wp.shape[0]
        idxs = [int(start_i)]
        acc = 0.0
        i = int(start_i)
        while acc < self.follow_horizon and len(idxs) < n:
            nxt = (i + 1) % n
            acc += float(np.linalg.norm(wp[nxt, :2] - wp[i, :2]))
            idxs.append(nxt)
            i = nxt
        idx = np.array(idxs)
        traj = np.zeros((len(idx), 5))
        traj[:, 0] = wp[idx, 0]
        traj[:, 1] = wp[idx, 1]
        traj[:, 2] = v_cmd
        traj[:, 3] = wp[idx, 3]
        traj[:, 4] = wp[idx, 4]
        return traj

    def _eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose, opp_vels=None):
        cost_weights = self.cost_weights
        n, k = self.traj_num, self.v_lattice_num

        mean_k, _ = get_curvature(all_traj, all_traj_clothoid, wheelbase=self.wheelbase)
        cost = np.zeros(self.traj_num)

        for func in self.shape_cost_funcs:
            cur_cost = func(
                all_traj, all_traj_clothoid, opp_poses, ego_pose,
                self.prev_traj_local, self.dt, self.map_metainfo
            )
            cost += cost_weights[0] * cur_cost

        for func in self.constant_cost_funcs:
            cur_cost = func(
                all_traj, all_traj_clothoid, opp_poses, ego_pose,
                self.prev_traj_local, self.dt, self.map_metainfo, self.collision_thres
            )
            cost += cur_cost

        all_traj_min_mean_k = np.min(mean_k)
        mean_k_lattice = np.repeat(mean_k, k).reshape(n, k)
        all_traj_v = all_traj[:, -1, 2]
        traj_v_lattice = (
            np.repeat(all_traj_v, k).reshape(n, k) * self.v_lattice_span * self.traj_v_scale
        )
        # 곡률에 따른 자동 감속 가중치 적용 (코너 벽 밀림 방지)
        abs_v_cost = (
            -cost_weights[-3] * np.log(1 + traj_v_lattice)
            + cost_weights[-2] * self.curve_v_penalty_gain * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        )

        # opp_vels 준비: 외부에서 안정적으로 추정된 속도가 있으면 사용, 없으면 폴백 계산
        if opp_vels is None or len(opp_vels) != len(opp_poses):
            has_prev = (np.sum(np.abs(self.prev_opp_pose)) > 1e-6 and
                        self.prev_opp_pose.shape[0] == opp_poses.shape[0])
            vels = np.zeros((opp_poses.shape[0], 2), dtype=np.float64)
            if has_prev and self.opp_dt > 1e-6:
                vels = (opp_poses[:, :2] - self.prev_opp_pose[:, :2]) / self.opp_dt
        else:
            vels = np.ascontiguousarray(opp_vels, dtype=np.float64)

        collision_cost = cost_weights[-1] * get_obstacle_collision_with_v(
            all_traj, all_traj_clothoid, traj_v_lattice,
            opp_poses, vels, max_pred_time=2.0
        )

        cost = np.repeat(cost, k).reshape(n, k)
        return cost, abs_v_cost, collision_cost


@njit(cache=True)
def sample_lookahead_square(pose_x, pose_y, pose_theta, velocity, waypoints,
                             lookahead_distances,
                             widths=np.linspace(-1.25, 1.25, num=11)):
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    local_span = np.vstack((np.zeros_like(widths), widths))
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(
            np.ascontiguousarray(nearest_p), d, waypoints[:, 0:2], t + nearest_i, wrap=True
        )
        if i2 is None:
            i2_int = 0
        else:
            i2_int = int(i2)
        lh_pt_theta = waypoints[i2_int, 3]
        lh_pt_v = waypoints[i2_int, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span + lh_pt.reshape(2, -1)
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v
    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, len(widths)).reshape(1, -1)
    v_grid = np.repeat(v_grid, len(widths)).reshape(1, -1)
    grid = np.vstack((xy_grid, theta_grid, v_grid)).T
    return grid


@njit(cache=True)
def traj_global2local(ego_pose, traj):
    new_traj = np.zeros_like(traj)
    pose_x, pose_y, pose_theta = ego_pose
    c = np.cos(pose_theta)
    s = np.sin(pose_theta)
    new_traj[..., 0] = c * (traj[..., 0] - pose_x) + s * (traj[..., 1] - pose_y)
    new_traj[..., 1] = -s * (traj[..., 0] - pose_x) + c * (traj[..., 1] - pose_y)
    return new_traj


@njit(cache=True)
def get_follow_optim_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None,
                           prev_traj=None, dt=None, map_metainfo=None):
    n = traj.shape[0]
    n_widths = 11
    if n % 5 == 0:
        n_widths = n // 5
    mid = n_widths // 2
    centers = np.empty(5, dtype=np.int64)
    for r in range(5):
        centers[r] = r * n_widths + mid
    center = np.repeat(centers, n_widths)
    traj_idx = np.arange(0, n, 1)
    idx_diff = traj_idx - center
    cost = idx_diff * idx_diff
    return cost


def get_curvature(traj, traj_clothoid, wheelbase=0.325):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)
    s = traj_clothoid[:, -1]
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T
    traj_k = k0 + dk * s_pts
    traj_k_abs = np.abs(traj_k)
    traj_steer = np.arctan(wheelbase * traj_k)
    max_steer = np.max(np.abs(traj_steer), axis=1)
    mean_k = np.mean(traj_k_abs, axis=1)
    for i in range(len(mean_k)):
        # 실제 차량 물리 조향 한계(0.26 rad = 14.9°) 이전인 0.25 rad 초과 시 강한 페널티로 클리핑/언더스티어 차단
        if max_steer[i] > 0.25:
            mean_k[i] *= 5.0
    max_k = np.max(traj_k_abs, axis=1)
    return mean_k, max_k


@njit(cache=True)
def get_map_collision(traj, traj_clothoid, opp_poses=None, ego_pose=None,
                       prev_traj=None, dt=None, map_metainfo=None, collision_thres=0.35):
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)
    collisions = map_collision(all_traj_pts[:, 0:2], dt, map_metainfo, eps=collision_thres)
    n_trajs = len(traj) 
    collisions = collisions.reshape(n_trajs, -1)
    cost = np.zeros(n_trajs, dtype=np.float64)
    
    for i in range(n_trajs):
        has_collision = False
        for j in range(collisions.shape[1]):
            if collisions[i, j]:
                has_collision = True
                break

        if has_collision:
            cost[i] = 3000.0
        else:
            cost[i] = 0.0

    return cost


@njit(cache=True)
def get_obstacle_collision_with_v(traj, traj_clothoid, v_lattice, opp_poses,
                                   opp_vels, max_pred_time=2.0):
    n, m, _ = traj.shape
    k = v_lattice.shape[1]
    if opp_poses.shape[0] == 0:
        return np.zeros((n, k))

    max_cost = 25.0
    min_cost = 12.0
    width, length = 0.31, 0.58
    # 좁은 트랙(1.2m~1.5m) 실차 주행을 위한 현실적 규격 마진 적용:
    ego_col_length = 0.52
    ego_col_width = 0.28
    opp_col_length = 0.55
    opp_col_width = 0.28
    # 두 차량 반폭 합: 0.14 + 0.14 = 0.28m (30cm 이상 이격 시 통과 가능)

    # 1) 소프트 사전 회피: 장애물 2.2m 전방부터 충분한 회피 폭(40cm) 확보 유도
    soft_lookahead_dist = 2.2
    soft_lat_width = 0.40  # 중심선 좌우 40cm 이내에 페널티 -> 40cm 이상 시원하게 비켜나도록 강제

    # 2) 추월 후 안전 복귀 마진: 장애물 통과 후 칼치기/접촉 방지를 위해 전방 1.2m 여유 확보
    clearance_ahead = 1.2
    clearance_lat_width = 0.35

    traj_xyt = np.empty((traj.shape[0], traj.shape[1], 3), dtype=traj.dtype)
    traj_xyt[:, :, 0] = traj[:, :, 0]
    traj_xyt[:, :, 1] = traj[:, :, 1]
    traj_xyt[:, :, 2] = traj[:, :, 3]

    traj_arc_lengths = np.zeros((n, m))
    for i in range(n):
        arc_len = traj_clothoid[i, 5]
        for j in range(m):
            traj_arc_lengths[i, j] = arc_len * j / max(m - 1, 1)

    cost = np.zeros((n, k))

    for i in range(n):
        tr = traj_xyt[i]
        for ki in range(k):
            ego_v = v_lattice[i, ki]
            if ego_v < 1e-3:
                ego_v = 1e-3
            max_collision_cost = 0.0
            # j=0(현재 에고 위치)은 제외하고 미래 이동 궤적(j>=1)에 대해서만 충돌 검사
            for j in range(1, m):
                t_reach = traj_arc_lengths[i, j] / ego_v
                if t_reach > max_pred_time:
                    t_reach = max_pred_time
                ego_point = tr[j]
                ego_box = get_vertices(ego_point, ego_col_length, ego_col_width)
                for oi in range(opp_poses.shape[0]):
                    predicted_opp = np.empty(3)
                    predicted_opp[0] = opp_poses[oi, 0] + opp_vels[oi, 0] * t_reach
                    predicted_opp[1] = opp_poses[oi, 1] + opp_vels[oi, 1] * t_reach
                    opp_speed = math.hypot(opp_vels[oi, 0], opp_vels[oi, 1])
                    if opp_speed > 0.2:
                        predicted_opp[2] = math.atan2(opp_vels[oi, 1], opp_vels[oi, 0])
                    else:
                        predicted_opp[2] = opp_poses[oi, 2]

                    # 1. 하드 충돌 박스 검사 (직접 접촉 위험 영역에 대한 강력한 페널티)
                    opp_box = get_vertices(predicted_opp, opp_col_length, opp_col_width)
                    if collision(opp_box, ego_box):
                        point_cost = max_cost - j * (max_cost - min_cost) / m
                        if point_cost > max_collision_cost:
                            max_collision_cost = point_cost
                    else:
                        dx = ego_point[0] - predicted_opp[0]
                        dy = ego_point[1] - predicted_opp[1]
                        cos_ot = math.cos(predicted_opp[2])
                        sin_ot = math.sin(predicted_opp[2])
                        d_long = cos_ot * dx + sin_ot * dy
                        d_lat = -sin_ot * dx + cos_ot * dy

                        # 2. 소프트 사전 회피 비용 (장애물 후방 접근 시 40cm 이상 확실한 회피 유도)
                        if -soft_lookahead_dist <= d_long <= 0.0 and abs(d_lat) < soft_lat_width:
                            lat_factor = 1.0 - (abs(d_lat) / soft_lat_width)
                            long_ratio = (d_long + soft_lookahead_dist) / soft_lookahead_dist
                            long_factor = 0.35 + 0.65 * max(0.0, min(1.0, long_ratio))
                            soft_cost = 9.0 * lat_factor * long_factor
                            if soft_cost > max_collision_cost:
                                max_collision_cost = soft_cost

                        # 3. 추월 후 안전 복귀 마진 (장애물 전방 1.2m 구간 칼치기/접촉 방지)
                        elif 0.0 < d_long <= clearance_ahead and abs(d_lat) < clearance_lat_width:
                            ret_factor = (1.0 - (d_long / clearance_ahead)) * (1.0 - (abs(d_lat) / clearance_lat_width))
                            ret_cost = 12.0 * ret_factor
                            if ret_cost > max_collision_cost:
                                max_collision_cost = ret_cost
            cost[i, ki] = max_collision_cost
    return cost