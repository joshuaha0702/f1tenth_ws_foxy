import os
import logging
import time
import yaml
import numpy as np
from PIL import Image
from numba import njit
from scipy.ndimage import distance_transform_edt as edt
from pyclothoids import Clothoid

from f1tenth_lattice_ros2.planner_utils import (
    nearest_point, intersect_point, get_rotation_matrix,
    zero_2_2pi, sample_traj, map_collision, get_vertices, collision,
)
from f1tenth_lattice_ros2.pure_pursuit import PurePursuitPlanner

logger = logging.getLogger(__name__)


class LatticePlanner:
    def __init__(self, conf, map_path, wpt_path, wb=0.33):
        self.wheelbase = wb

        # load waypoints: reorder to (x, y, v, heading, s)
        raw = np.loadtxt(wpt_path, delimiter=';', skiprows=2)
        self.waypoints = np.vstack(
            (raw[:, 1], raw[:, 2], raw[:, 5], raw[:, 3], raw[:, 0])
        ).T

        self.traj_num = 55
        self.lh_grid_rows = 5
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

        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)
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
        self.collision_thres = 0.35

        # === 선두 차량 추종(following) 모드 설정 ===
        follow_cfg = getattr(conf, 'follow', None)
        self.follow_enabled = False
        self.follow_zones = []
        self.follow_active = False  # 히스테리시스 상태(진입/해제 떨림 방지)
        if isinstance(follow_cfg, dict):
            zones = follow_cfg.get('zones') or []
            self.follow_zones = [(int(z['min']), int(z['max'])) for z in zones]
            if self.follow_zones:
                self.follow_enabled = True
                self.follow_lateral_align = float(follow_cfg.get('lateral_align_m', 2.0))
                self.follow_desired_gap = float(follow_cfg.get('desired_gap_m', 1.5))
                self.follow_kp_gap = float(follow_cfg.get('kp_gap', 0.8))
                self.follow_max_speed = float(follow_cfg.get('max_follow_speed', 1.5))
                self.follow_horizon = float(follow_cfg.get('horizon_m', 3.0))
                logger.info(
                    f'[follow] enabled — zones={self.follow_zones}, '
                    f'lateral_align={self.follow_lateral_align}m, '
                    f'desired_gap={self.follow_desired_gap}m, max_speed={self.follow_max_speed}m/s'
                )

    def set_cost_weights(self, cost_weights):
        if isinstance(cost_weights, int):
            n = cost_weights
            cost_weights = np.array([1 / n] * n)
        self.cost_weights = cost_weights

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None, dt=None):
        self.step += 1
        if waypoints is None:
            waypoints = self.waypoints

        self.opp_dt = dt if (dt is not None and dt > 1e-6) else self.time_interval

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
            pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid
        )

        _t_clothoid0 = time.perf_counter()
        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(
                pose_x, pose_y, pose_theta, point[0], point[1], point[2]
            )
            traj = sample_traj(clothoid, self.traj_points, point[3])
            all_traj.append(traj)
            all_traj_clothoid.append(np.array(clothoid.Parameters))

        all_traj = np.array(all_traj)
        all_traj_clothoid = np.array(all_traj_clothoid)
        _t_clothoid1 = time.perf_counter()
        traj_cost, abs_v_cost, collision_cost = self._eval(all_traj, all_traj_clothoid, opp_poses, ego_pose)
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

        self.best_traj = all_traj[row_idx].copy()
        self.best_traj_ref_v = self.best_traj[-1, 2]
        self.best_traj[:, 2] *= (self.v_lattice_span[col_idx] * self.traj_v_scale)
        self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])
        self.prev_opp_pose = opp_poses[:, :2] if opp_poses.shape[0] > 0 else np.zeros((1, 2))
        best_cost = self.all_costs[row_idx, col_idx]
        return self.best_traj, best_cost, traj_cost[row_idx, col_idx], abs_v_cost[row_idx, col_idx], collision_cost[row_idx, col_idx]

    def _in_follow_zone(self, idx):
        for lo, hi in self.follow_zones:
            if lo <= idx <= hi:
                return True
        return False

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

    def _eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose):
        cost_weights = self.cost_weights
        n, k = self.traj_num, self.v_lattice_num

        mean_k, _ = get_curvature(all_traj, all_traj_clothoid)
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
        abs_v_cost = (
            -cost_weights[-3] * np.log(1 + traj_v_lattice)
            + cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        )

        collision_cost = cost_weights[-1] * get_obstacle_collision_with_v(
            all_traj, all_traj_clothoid, traj_v_lattice,
            opp_poses, self.prev_opp_pose, self.opp_dt
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
    center = np.array((5, 16, 27, 38, 49))
    center = np.repeat(center, 11)
    traj_idx = np.arange(0, n, 1)
    idx_diff = traj_idx - center
    cost = idx_diff * idx_diff
    return cost


def get_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)
    s = traj_clothoid[:, -1]
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T
    traj_k = k0 + dk * s_pts
    traj_k_abs = np.abs(traj_k)
    wheelbase = 0.33
    traj_steer = np.arctan(wheelbase * traj_k)
    max_steer = np.max(np.abs(traj_steer), axis=1)
    mean_k = np.mean(traj_k_abs, axis=1)
    for i in range(len(mean_k)):
        if max_steer[i] > 0.32:
            mean_k[i] *= 2.0
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
                                   prev_oppo_pose, dt=None):
    max_cost = 20.0
    min_cost = 10.0
    width, length = 0.31, 0.58
    safety_width_distance = 0.15
    safety_length_distance = 0.2
    n, m, _ = traj.shape
    k = v_lattice.shape[1]

    traj_xyt = np.empty((traj.shape[0], traj.shape[1], 3), dtype=traj.dtype)
    traj_xyt[:, :, 0] = traj[:, :, 0]
    traj_xyt[:, :, 1] = traj[:, :, 1]
    traj_xyt[:, :, 2] = traj[:, :, 3]

    has_prev = (np.sum(np.abs(prev_oppo_pose)) > 1e-6 and
                prev_oppo_pose.shape[0] == opp_poses.shape[0])
    opp_vels = np.zeros((opp_poses.shape[0], 2))
    if has_prev and dt > 1e-6:
        for oi in range(opp_poses.shape[0]):
            opp_vels[oi, 0] = (opp_poses[oi, 0] - prev_oppo_pose[oi, 0]) / dt
            opp_vels[oi, 1] = (opp_poses[oi, 1] - prev_oppo_pose[oi, 1]) / dt

    traj_arc_lengths = np.zeros((n, m))
    for i in range(n):
        arc_len = traj_clothoid[i, 5]
        for j in range(m):
            traj_arc_lengths[i, j] = arc_len * j / max(m - 1, 1)

    cost = np.zeros((n, k))
    col_length = length + safety_length_distance
    col_width = width + safety_width_distance

    for i in range(n):
        tr = traj_xyt[i]
        for ki in range(k):
            ego_v = v_lattice[i, ki]
            if ego_v < 1e-3:
                ego_v = 1e-3
            max_collision_cost = 0.0
            for j in range(m):
                t_reach = traj_arc_lengths[i, j] / ego_v
                if t_reach > 0.5:
                    t_reach = 0.5
                ego_point = tr[j]
                ego_box = get_vertices(ego_point, col_length, col_width)
                for oi in range(opp_poses.shape[0]):
                    predicted_opp = np.empty(3)
                    predicted_opp[0] = opp_poses[oi, 0] + opp_vels[oi, 0] * t_reach
                    predicted_opp[1] = opp_poses[oi, 1] + opp_vels[oi, 1] * t_reach
                    predicted_opp[2] = opp_poses[oi, 2]
                    opp_box = get_vertices(predicted_opp, col_length, col_width)
                    if collision(opp_box, ego_box):
                        point_cost = max_cost - j * (max_cost - min_cost) / m
                        if point_cost > max_collision_cost:
                            max_collision_cost = point_cost
            cost[i, ki] = max_collision_cost
    return cost