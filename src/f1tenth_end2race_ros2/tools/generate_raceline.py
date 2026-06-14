#!/usr/bin/env python3
"""
Race line generator for F1Tenth maps.

Adapted from: https://github.com/KimDaevvon/Fix-End2Race

Usage:
    python3 generate_raceline.py --map_dir /path/to/maps/Simple \
        --map_name Simple --num_lanes 1 --v_max 3.0

Output:
    lane0.csv      - track boundary offsets
    raceline0.csv  - optimized raceline with velocity profile
"""

import cv2
import numpy as np
import os
import argparse
import yaml
import sys
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate raceline from 2D map')
    parser.add_argument('--map_dir', type=str, required=True,
                        help='Directory containing {map_name}_map.yaml and {map_name}_map.png')
    parser.add_argument('--map_name', type=str, required=True,
                        help='Map name (e.g. Simple)')
    parser.add_argument('--map_img_ext', type=str, default='.png')
    parser.add_argument('--num_lanes', type=int, default=1,
                        help='Number of lane lines to generate (raceline per lane)')
    parser.add_argument('--clockwise', action='store_true', default=True,
                        help='Track direction is clockwise')
    parser.add_argument('--inner_safe_dist', type=float, default=0.4,
                        help='Safety margin from inner wall (m)')
    parser.add_argument('--outer_safe_dist', type=float, default=0.4,
                        help='Safety margin from outer wall (m)')
    parser.add_argument('--v_max', type=float, default=3.0,
                        help='Maximum vehicle speed (m/s)')
    parser.add_argument('--vehicle_width', type=float, default=0.31)
    parser.add_argument('--vehicle_mass', type=float, default=3.362)
    parser.add_argument('--drag_coeff', type=float, default=0.0075)
    parser.add_argument('--num_laps', type=int, default=2)
    return parser.parse_args()


def reorder_vertex(image, lane, total_lane_image):
    """Reconstruct a continuous path from unordered pixel coordinates."""
    path_img = np.zeros_like(image)
    for idx in range(len(lane)):
        cv2.circle(path_img, (int(lane[idx, 0]), int(lane[idx, 1])), 1, 255, 1)
    curr_kernel = np.ones((2, 2), np.uint8)
    iter_cnt = 0
    while True:
        if iter_cnt > 10:
            print('ERROR: reorder_vertex did not converge', file=sys.stderr)
            sys.exit(1)
        curr_contours, curr_hierarchy = cv2.findContours(
            path_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(curr_contours) == 2 and curr_hierarchy[0][-1][-1] == 0:
            break
        path_img = cv2.dilate(path_img, curr_kernel, iterations=1)
        iter_cnt += 1
    path_img = cv2.ximgproc.thinning(path_img)
    curr_contours, _ = cv2.findContours(
        path_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if total_lane_image is not None:
        total_lane_image = cv2.add(total_lane_image, path_img)
    return np.squeeze(curr_contours[0]), total_lane_image


def transform_coords(path, height, s, tx, ty):
    """Transform pixel coordinates to world coordinates (meters)."""
    new_x = path[:, 0] * s + tx
    new_y = (height - path[:, 1]) * s + ty
    if path.shape[1] > 2:
        return np.vstack((new_x, new_y, path[:, 2] * s, path[:, 3] * s)).T
    return np.vstack((new_x, new_y)).T


def save_csv(data, csv_path, header=None):
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for row in data:
            writer.writerow(row.tolist())
    print(f'Saved: {csv_path}')


def generate_lanes(args, map_dir):
    """Extract lane center lines from the occupancy map image."""
    yaml_path = os.path.join(map_dir, args.map_name + '_map.yaml')
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    scale = meta['resolution']
    offset_x = meta['origin'][0]
    offset_y = meta['origin'][1]

    img_path = os.path.join(map_dir, args.map_name + '_map' + args.map_img_ext)
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise FileNotFoundError(f'Map image not found: {img_path}')
    h, w = input_img.shape

    # Invert, threshold, remove noise
    output_img = ~input_img
    _, output_img = cv2.threshold(output_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(output_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 70:
            cv2.fillPoly(output_img, pts=[contour], color=0)

    # Morphological thinning to get skeleton
    kernel = np.ones((5, 5), np.uint8)
    output_img = cv2.dilate(output_img, kernel, iterations=1)
    output_img = cv2.ximgproc.thinning(output_img)

    # Find inner / outer boundaries
    contours, hierarchy = cv2.findContours(output_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parents = hierarchy[0][:, 3]
    node = np.argmax(parents)
    tree_indices = []
    while node != -1:
        tree_indices.append(node)
        node = parents[node]
    tree_indices.reverse()
    outer_bound = contours[tree_indices[1]]
    inner_bound = contours[tree_indices[2]]

    # Compute distance ratios for every pixel
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = X.flatten().tolist()
    Y = Y.flatten().tolist()
    valid_pts = []
    for x, y in zip(X, Y):
        outer_dist = cv2.pointPolygonTest(outer_bound, (x, y), True)
        inner_dist = cv2.pointPolygonTest(inner_bound, (x, y), True)
        if (outer_dist > args.outer_safe_dist / scale
                and inner_dist < -args.inner_safe_dist / scale):
            ratio = abs(inner_dist) / (abs(outer_dist) + 1e-8)
            valid_pts.append([x, y, inner_dist, outer_dist, ratio])
    valid_pts = np.array(valid_pts)

    lane_ratios = np.arange(1, args.num_lanes + 1) / np.arange(args.num_lanes, 0, -1)
    if not np.any(lane_ratios == 1.0):
        lane_ratios = np.append(lane_ratios, 1.0)

    total_lane_image = output_img.copy()
    lanes, lane_names = [], []

    for idx, ratio in enumerate(lane_ratios):
        valid_mask = np.abs(valid_pts[:, -1] - ratio) < ratio / 10
        lane = valid_pts[valid_mask, 0:2].astype(int)
        lane, total_lane_image = reorder_vertex(output_img, lane, total_lane_image)
        if args.clockwise:
            lane = np.flipud(lane)

        left_dists, right_dists = [], []
        for x, y in lane:
            od = cv2.pointPolygonTest(outer_bound, (int(x), int(y)), True)
            id_ = cv2.pointPolygonTest(inner_bound, (int(x), int(y)), True)
            od = od - args.outer_safe_dist / scale
            id_ = abs(id_) - args.inner_safe_dist / scale
            if args.clockwise:
                left_dists.append(od)
                right_dists.append(id_)
            else:
                left_dists.append(id_)
                right_dists.append(od)

        lane = np.vstack((lane.T, right_dists, left_dists)).T
        lane = transform_coords(lane, h, scale, offset_x, offset_y)

        lane_name = f'lane{idx}'
        csv_path = os.path.join(map_dir, f'{lane_name}.csv')
        save_csv(lane, csv_path, header=['#x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
        lanes.append(lane)
        lane_names.append(lane_name)

    return lanes, lane_names


def prep_track(reftrack_imp, reg_smooth_opts, stepsize_opts, debug=False, min_width=None):
    reftrack_interp = tph.spline_approximation.spline_approximation(
        track=reftrack_imp,
        k_reg=reg_smooth_opts['k_reg'],
        s_reg=reg_smooth_opts['s_reg'],
        stepsize_prep=stepsize_opts['stepsize_prep'],
        stepsize_reg=stepsize_opts['stepsize_reg'],
        debug=debug,
    )
    refpath_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x, coeffs_y, a_interp, normvec = tph.calc_splines.calc_splines(path=refpath_cl)

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]
            if cur_width < min_width:
                delta = (min_width - cur_width) / 2
                reftrack_interp[i, 2] += delta
                reftrack_interp[i, 3] += delta

    return reftrack_interp, normvec, a_interp, coeffs_x, coeffs_y


def generate_raceline(lane_data, lane_name, args, map_dir):
    """Generate a velocity-profiled raceline from a lane centerline."""
    stepsize_opts = {
        'stepsize_prep': 0.5,
        'stepsize_reg': 2.0,
        'stepsize_interp_after_opt': 0.2,
    }
    reg_smooth_opts = {'k_reg': 3, 's_reg': 0.0}
    vel_calc_opts = {'dyn_model_exp': 1.0, 'vel_profile_conv_filt_window': 31}

    # vehicle GGV / ax_max_machines – use built-in defaults if files not found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    veh_info_dir = os.path.join(script_dir, '..', 'vehicle_dynamic_info')
    ggv_path = os.path.join(veh_info_dir, 'ggv.csv')
    ax_path = os.path.join(veh_info_dir, 'ax_max_machines.csv')

    if not os.path.exists(ggv_path) or not os.path.exists(ax_path):
        # Generate simple default GGV and ax_max_machines
        _write_default_vehicle_files(veh_info_dir, args.v_max)

    ggv, ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
        ggv_import_path=ggv_path,
        ax_max_machines_import_path=ax_path,
    )

    v_max = min(args.v_max, np.max(ggv[:, 0]) * 0.95)
    min_width = args.vehicle_width * 2.0

    reftrack_interp, normvec, a_interp, coeffs_x, coeffs_y = prep_track(
        reftrack_imp=lane_data,
        reg_smooth_opts=reg_smooth_opts,
        stepsize_opts=stepsize_opts,
        min_width=min_width,
    )

    alpha_opt = np.zeros(reftrack_interp.shape[0])

    (raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt,
     spline_inds, t_vals, s_points, spline_lengths, el_lengths) = tph.create_raceline.create_raceline(
        refline=reftrack_interp[:, :2],
        normvectors=normvec,
        alpha=alpha_opt,
        stepsize_interp=stepsize_opts['stepsize_interp_after_opt'],
    )

    psi, kappa = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt,
        coeffs_y=coeffs_y_opt,
        ind_spls=spline_inds,
        t_spls=t_vals,
        calc_curv=True,
    )

    vx_profile = tph.calc_vel_profile.calc_vel_profile(
        ax_max_machines=ax_max_machines,
        kappa=kappa,
        el_lengths=el_lengths,
        closed=True,
        drag_coeff=args.drag_coeff,
        m_veh=args.vehicle_mass,
        ggv=ggv,
        v_max=v_max,
        **vel_calc_opts,
    )

    ax_profile = tph.calc_ax_profile.calc_ax_profile(
        vx_profile=vx_profile,
        el_lengths=el_lengths,
        eq_length_output=False,
    )

    s_cumsum = np.cumsum(el_lengths)
    s_cumsum = np.insert(s_cumsum, 0, 0.0)[:-1]

    raceline_out = np.column_stack([
        s_cumsum,
        raceline_interp[:, 0],
        raceline_interp[:, 1],
        psi,
        kappa,
        vx_profile,
        ax_profile,
    ])

    raceline_name = lane_name.replace('lane', 'raceline')
    out_path = os.path.join(map_dir, f'{raceline_name}.csv')
    header = '# s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2'
    np.savetxt(out_path, raceline_out, delimiter=';', header=header, comments='', fmt='%.6f')
    print(f'Saved raceline: {out_path}')


def _write_default_vehicle_files(out_dir, v_max):
    """Write minimal GGV and ax_max_machines for an F1Tenth car."""
    os.makedirs(out_dir, exist_ok=True)
    v_steps = np.linspace(0.0, v_max, 10)
    ax_max = 3.0
    ay_max = 3.0
    ggv = np.column_stack([v_steps, np.full_like(v_steps, ax_max), np.full_like(v_steps, ay_max)])
    np.savetxt(os.path.join(out_dir, 'ggv.csv'), ggv, delimiter=';',
               header='# v_mps;ax_max_mps2;ay_max_mps2', comments='', fmt='%.3f')

    ax_max_machines = np.column_stack([v_steps, np.full_like(v_steps, ax_max)])
    np.savetxt(os.path.join(out_dir, 'ax_max_machines.csv'), ax_max_machines, delimiter=';',
               header='# v_mps;ax_max_mps2', comments='', fmt='%.3f')


def main():
    args = parse_arguments()
    map_dir = args.map_dir

    print(f'Generating lanes for map: {args.map_name} in {map_dir}')
    lanes, lane_names = generate_lanes(args, map_dir)

    for lane_data, lane_name in zip(lanes, lane_names):
        print(f'Generating raceline for {lane_name} ...')
        generate_raceline(lane_data, lane_name, args, map_dir)

    print('Done!')


if __name__ == '__main__':
    main()
