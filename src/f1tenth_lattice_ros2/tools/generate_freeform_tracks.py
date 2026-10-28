#!/usr/bin/env python3
"""Generate free-form (non-oval) tracks for End2Race training: S-bends, hairpins, mixed corners.

The Simple_v* variants are all stadium ovals with raceline curvature <= 0.38 1/m, while the
original Simple map reaches 0.53 1/m (22 % of the line above 0.4). Each track here is a closed
centre line built from constant-curvature corners (random turn angles, left and right, so
hairpins, 90-degree corners and S-bends appear) joined by straights whose lengths are solved so
the loop closes, with walls at +/- width/2. Outputs per map match generate_simple_augmented.py: occupancy map, wall
mesh, Gazebo world, raceline{0,1,2}{,_ccw}.csv (raceline1 = centre line), lattice_config.yaml
and episodes.yaml. Track list: config/freeform_tracks.yaml. Run from the workspace root:

    python3 src/f1tenth_lattice_ros2/tools/generate_freeform_tracks.py [--only NAME ...]

`mode: spline` tracks (config/spline_tracks.yaml, Spline_s*) instead make the centre-line
curvature a periodic cubic spline of arc length, so it varies smoothly and continuously along the
lap like the original Simple map (whose curvature has no arc/straight steps):

    python3 src/f1tenth_lattice_ros2/tools/generate_freeform_tracks.py \
        --config src/f1tenth_lattice_ros2/config/spline_tracks.yaml      # -> maps/Spline_overview.png
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.ndimage import distance_transform_edt

from generate_simple_augmented import (MIN_LANE_CLEARANCE, POINT_STEP, RESOLUTION, SPEED,
                                       WALL_HEIGHT, WALL_THICKNESS, _sub_once, write_world)

DENSE_STEP = 0.05          # centre-line resampling (m)
MAP_MARGIN = 1.5           # free space between the outer wall and the image border (m)
MIN_IMAGE_SIZE = 2000
WALL_GAP = 0.6             # minimum solid gap between two parts of the track (m)
INNER_WALL_MIN_RADIUS = 0.3   # inner wall radius at the tightest corner (m)
LAP_RANGE = (45.0, 110.0)
LANE_FRACTIONS = (-0.2, 0.0, 0.2)   # lane offsets as a share of the width (+ = left)
FOLLOW_KAPPA = 0.15        # raceline1 rows above this |curvature| form the follow zones
# Gazebo's ackermann plugin brakes weakly (velocity P gain 1.0: a 0 m/s command still leaves the
# car near 1.3 m/s for ~1 s), and these tracks force following behind slow leaders far more
# often than the ovals, so following starts earlier and leaders are not as slow.
FOLLOW_ALIGN_M = 3.0       # same as the oval configs (was 5.0 while the sim could not brake, vel_p_gain 1.0)
LEADER_V_SCALE_MIN = 0.3   # same as the oval configs (was 0.45 for the same reason)
MIN_OVERTAKE_ROWS = 10     # straights shorter than this (rows of 0.2 m = 2 m) stay in follow mode (was 25 = 5 m):
                           # leaving follow on a short straight right behind the leader and
                           # switching back at the next corner caused rear-end contacts


def resample_closed(pts, step):
    closed = np.vstack((pts, pts[:1]))
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(8, int(round(s[-1] / step)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack((np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])))


def curvature(pts):
    prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    h1 = np.arctan2(*(pts - prev).T[::-1])
    h2 = np.arctan2(*(nxt - pts).T[::-1])
    dpsi = np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))
    ds = (np.linalg.norm(pts - prev, axis=1) + np.linalg.norm(nxt - pts, axis=1)) / 2
    return dpsi / ds


def normals(pts):
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))      # left normal


class FreeformTrack:
    def __init__(self, name, width, kappa_max, seed, mode='corners', max_tries=300000,
                 kappa_min=None, n_corners=None, left_share=None, straight=None,
                 n_ctrl=None, straight_share=0.3, kappa_spread=0.3):
        self.name = name
        self.width = width
        self.kappa_max = kappa_max
        self.seed = seed
        self.mode = mode
        limit = 1.0 / (width / 2 + INNER_WALL_MIN_RADIUS)
        if kappa_max > limit:
            raise ValueError(f'{name}: kappa_max {kappa_max} > {limit:.2f} (inner wall would fold)')
        self.kappa_min = kappa_min
        if mode == 'corners':
            def draw():
                return self._corner_centre_line(rng, n_corners, left_share, tuple(straight))
        elif mode == 'spline':
            def draw():
                return self._spline_centre_line(rng, n_ctrl, straight_share, left_share or 0.0, kappa_spread)
        else:
            raise ValueError(f'{name}: unknown mode {mode!r} (corners or spline)')
        rng = np.random.default_rng(seed)
        for attempt in range(max_tries):
            centre = draw()
            if centre is not None:
                self.centre = centre
                self.attempts = attempt + 1
                break
        else:
            raise ValueError(f'{name}: no valid shape in {max_tries} tries')

    def _corner_centre_line(self, rng, n_corners, left_share, straight):
        """Corners of constant curvature in [kappa_min, kappa_max]; turn angles sum to -360 deg
        (clockwise). Straight lengths (one before every corner) are solved so the loop closes."""
        # left turns 30-110 deg (S-bends); right turns drawn 40-190 deg, then scaled so all
        # turns sum to -360 deg; a right turn outside 30-200 deg rejects the draw
        left = rng.uniform(size=n_corners) < left_share
        left[rng.integers(n_corners)] = False                    # at least one right turn
        angles = np.where(left, rng.uniform(30, 110, n_corners), -rng.uniform(40, 190, n_corners))
        need = -360.0 - angles[left].sum()
        angles[~left] *= need / angles[~left].sum()
        if (np.abs(angles[~left]) < 30).any() or (np.abs(angles[~left]) > 200).any():
            return None
        angles = [math.radians(a) for a in angles]
        kappa = rng.uniform(self.kappa_min, self.kappa_max, n_corners)
        kappa[np.argmax(np.abs(angles))] = self.kappa_max     # the sharpest turn uses the peak
        # heading at each straight and the displacement of each arc
        heading, h = [], rng.uniform(0, 2 * math.pi)
        arc_vec = np.zeros(2)
        for a, k in zip(angles, kappa):
            heading.append(h)
            r = 1.0 / k
            # chord of an arc turning by a with radius r, starting at heading h
            arc_vec += r * np.array([math.sin(h + a) - math.sin(h), -(math.cos(h + a) - math.cos(h))]) * np.sign(a)
            h += a
        A = np.array([[math.cos(t) for t in heading], [math.sin(t) for t in heading]])
        lo, hi = straight
        target = rng.uniform(lo, hi, n_corners)
        # closure A L = -arc_vec, plus a weak pull of L towards random target lengths
        w = 0.05
        M = np.vstack((A, w * np.eye(n_corners)))
        y = np.concatenate((-arc_vec, w * target))
        res = lsq_linear(M, y, bounds=(lo, hi * 2.5))
        if np.linalg.norm(A @ res.x + arc_vec) > 1e-3:
            return None
        pts = self._build(res.x, angles, kappa, heading[0])
        return self._valid(pts)

    def _spline_centre_line(self, rng, n_ctrl, straight_share, left_share, kappa_spread):
        """Curvature is a periodic cubic spline of arc length through n_ctrl control values, so it
        varies smoothly and continuously along the lap (like the original Simple map) instead of
        stepping between arcs and straights. Control points sit at jittered, roughly even spacing;
        each is a straight (|k| ~ 0, share straight_share), a left bend (S-bend, left_share of the
        rest, 20-70 % of the peak) or a right corner (peak x [1 - kappa_spread, 1]). The lap length
        follows from the -360 deg total turn; a smooth one-period correction a cos + b sin of the
        curvature (zero net turn) closes the loop, and the shape is scaled so peak |k| = kappa_max.
        (A position spline through jittered points around an ellipse was tried first: its
        curvature overshoots at every control point, so |k| > 0.75 peak never covered more than
        ~6 % of the lap, against 22 % on Simple.)"""
        m = 2048                                        # integration grid over one lap
        u_ctrl = (np.arange(n_ctrl) + rng.uniform(-0.3, 0.3, n_ctrl)) / n_ctrl
        kind = rng.uniform(size=n_ctrl)
        value = np.where(kind < straight_share, rng.uniform(-0.08, 0.08, n_ctrl),
                         -rng.uniform(1.0 - kappa_spread, 1.0, n_ctrl))
        left = (kind >= straight_share) & (rng.uniform(size=n_ctrl) < left_share)
        value[left] = rng.uniform(0.2, 0.7, left.sum())
        order = np.argsort(u_ctrl)
        knots = np.append(u_ctrl[order], u_ctrl[order][0] + 1.0)
        f = CubicSpline(knots, np.append(value[order], value[order][0]), bc_type='periodic')
        u = (np.arange(m) + 0.5) / m                     # cell centres
        k = f(np.mod(u - knots[0], 1.0) + knots[0])
        k /= np.abs(k).max()                             # peak 1 before closure / scaling
        if k.mean() >= -0.02:
            return None
        lap = -2 * math.pi / k.mean()                   # unit peak curvature: total turn -2 pi
        ds = lap / m
        cos_u, sin_u = np.cos(2 * math.pi * u), np.sin(2 * math.pi * u)
        corr = np.zeros(2)
        for _ in range(12):                              # Newton on the closure gap
            kk = k + corr[0] * cos_u + corr[1] * sin_u
            theta = np.cumsum(kk) * ds
            e = np.exp(1j * theta) * ds
            gap = e.sum()
            if abs(gap) < 1e-6 * lap:
                break
            # d theta_j / d a = sum_{i<=j} cos_u_i ds; d gap / d a = sum_j i e_j d theta_j / d a
            ja = (1j * e * np.cumsum(cos_u) * ds).sum()
            jb = (1j * e * np.cumsum(sin_u) * ds).sum()
            J = np.array([[ja.real, jb.real], [ja.imag, jb.imag]])
            try:
                corr = corr - np.linalg.solve(J, [gap.real, gap.imag])
            except np.linalg.LinAlgError:
                return None
            if np.abs(corr).max() > 0.6:
                return None
        else:
            return None
        if abs(gap) >= 1e-6 * lap:
            return None
        scale = np.abs(kk).max() / self.kappa_max       # curvature scales with 1 / size
        lap *= scale
        if not LAP_RANGE[0] <= lap <= LAP_RANGE[1]:     # cheap reject before resampling
            return None
        n = int(round(lap / DENSE_STEP))
        theta = np.concatenate(([0.0], theta))          # heading at cell edges
        t = (np.arange(n) + 0.5) / n                    # resample headings, integrate positions
        h = np.interp(t, np.arange(m + 1) / m, theta)
        step = np.exp(1j * h) * lap / n
        z = np.concatenate(([0.0], np.cumsum(step)[:-1]))
        z -= np.arange(n) / n * step.sum()              # remove the residual closure error
        return self._valid(np.column_stack((z.real, z.imag)))

    @staticmethod
    def _build(lengths, angles, kappa, h0):
        pts, p, h = [], np.zeros(2), h0
        for L, a, k in zip(lengths, angles, kappa):
            n = max(1, int(round(L / DENSE_STEP)))
            for i in range(n):
                pts.append(p + (i / n) * L * np.array([math.cos(h), math.sin(h)]))
            p = p + L * np.array([math.cos(h), math.sin(h)])
            r, sgn = 1.0 / k, np.sign(a)
            centre = p + sgn * r * np.array([-math.sin(h), math.cos(h)])
            n = max(2, int(round(abs(a) * r / DENSE_STEP)))
            for i in range(n):
                hh = h + a * i / n
                pts.append(centre - sgn * r * np.array([-math.sin(hh), math.cos(hh)]))
            h += a
            p = centre - sgn * r * np.array([-math.sin(h), math.cos(h)])
        return resample_closed(np.array(pts), DENSE_STEP)

    def _valid(self, pts):
        lap = len(pts) * DENSE_STEP
        if not LAP_RANGE[0] <= lap <= LAP_RANGE[1]:
            return None
        area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        if area > 0:          # must be clockwise like the Simple maps
            return None
        if np.abs(curvature(pts)).max() > self.kappa_max * 1.05:
            return None
        # two parts of the track further apart along the lap than a U-turn must keep the walls apart
        n = len(pts)
        sep = int(math.ceil((math.pi * (self.width / 2 + 1.0)) / DENSE_STEP))
        sub = pts[::4]
        idx = np.arange(n)[::4]
        d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
        along = np.abs(idx[:, None] - idx[None])
        along = np.minimum(along, n - along)
        if (d[along > sep] < self.width + WALL_GAP).any():
            return None
        return pts

    # --- geometry -----------------------------------------------------------

    def placed(self):
        """Centre line shifted into positive map coordinates."""
        lo = self.centre.min(axis=0) - (self.width / 2 + MAP_MARGIN)
        return self.centre - lo

    def walls(self):
        c = self.placed()
        nrm = normals(c)
        return c + nrm * self.width / 2, c - nrm * self.width / 2      # left wall, right wall

    def lane(self, fraction):
        c = self.placed()
        return resample_closed(c + normals(c) * fraction * self.width, POINT_STEP)

    def image_shape(self):
        hi = self.placed().max(axis=0) + self.width / 2 + MAP_MARGIN
        return (max(MIN_IMAGE_SIZE, math.ceil(hi[1] / RESOLUTION)),
                max(MIN_IMAGE_SIZE, math.ceil(hi[0] / RESOLUTION)))

    # --- outputs ------------------------------------------------------------

    def render_map(self):
        height, width = self.image_shape()
        image = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(image)
        c = self.placed()
        # free = within width/2 of the centre line: draw the band as overlapping quads
        left, right = self.walls()
        for i in range(len(c)):
            j = (i + 1) % len(c)
            quad = [left[i], left[j], right[j], right[i]]
            draw.polygon([(x / RESOLUTION, height - y / RESOLUTION) for x, y in quad], fill=255)
        return np.array(image)

    def write_walls(self, path):
        half = WALL_THICKNESS / 2.0
        triangles = []
        c = self.placed()
        nrm = normals(c)
        for side in (1.0, -1.0):
            outer = c + nrm * side * (self.width / 2 + half)
            inner = c + nrm * side * (self.width / 2 - half)
            for i in range(len(c)):
                j = (i + 1) % len(c)
                a, b, cc, d = outer[i], outer[j], inner[i], inner[j]
                h = WALL_HEIGHT

                def v(p, z):
                    return (float(p[0]), float(p[1]), z)
                triangles.extend((
                    (v(a, 0), v(b, 0), v(b, h)), (v(a, 0), v(b, h), v(a, h)),
                    (v(cc, 0), v(d, h), v(d, 0)), (v(cc, 0), v(cc, h), v(d, h)),
                    (v(a, h), v(b, h), v(d, h)), (v(a, h), v(d, h), v(cc, h)),
                    (v(a, 0), v(d, 0), v(b, 0)), (v(a, 0), v(cc, 0), v(d, 0)),
                ))
        with path.open('wb') as out:
            out.write(f'{self.name} walls'.encode().ljust(80, b' ')[:80])
            out.write(struct.pack('<I', len(triangles)))
            for a, b, cc in triangles:
                out.write(struct.pack('<12fH', 0, 0, 0, *a, *b, *cc, 0))
        return len(triangles)

    @staticmethod
    def write_raceline(path, pts, counter_clockwise=False):
        if counter_clockwise:
            pts = np.roll(pts[::-1], 1, axis=0)
        closed = np.vstack((pts, pts[0]))
        ds = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        kappa = curvature(pts)
        with path.open('w') as out:
            out.write('# s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2\n')
            for i in range(len(pts) + 1):
                j = i % len(pts)
                out.write(f'{s[i]:.6f};{pts[j,0]:.6f};{pts[j,1]:.6f};'
                          f'{heading[j]:.6f};{kappa[j]:.6f};{SPEED:.6f};0.000000\n')
        return float(s[-1]), kappa


def follow_zones(kappa):
    """Index ranges of raceline1 corners: four rows before to two rows after |kappa| > threshold."""
    n = len(kappa)
    corner = np.abs(kappa) > FOLLOW_KAPPA
    if corner.all() or not corner.any():
        return [(0, n - 1)] if corner.all() else []
    start = int(np.argmin(corner))          # begin scanning on a straight row
    zones, run = [], None
    for k in range(n):
        i = (start + k) % n
        if corner[i] and run is None:
            run = i
        if not corner[i] and run is not None:
            zones.append((run, (i - 1) % n))
            run = None
    if run is not None:
        zones.append((run, (start - 1) % n))
    # merge corners separated by short straights (indices are in scan order from `start`)
    rel = [((lo - start) % n, (hi - start) % n) for lo, hi in zones]
    merged = [list(rel[0])]
    for lo, hi in rel[1:]:
        if lo - merged[-1][1] - 1 - 6 < MIN_OVERTAKE_ROWS:     # 6 = zone extension below
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    if len(merged) > 1 and (merged[0][0] + n) - merged[-1][1] - 1 - 6 < MIN_OVERTAKE_ROWS:
        merged[0][0] = merged[-1][0] - n          # wraps past the scan start
        merged.pop()
    if len(merged) == 1 and n - (merged[0][1] - merged[0][0] + 1) - 6 < MIN_OVERTAKE_ROWS:
        return [(0, n - 1)]                       # no straight long enough: follow everywhere
    zones = [((lo + start) % n, (hi + start) % n) for lo, hi in merged]
    out = []
    for lo, hi in zones:
        if hi < lo:                                # wraps around the lap start
            hi += n
        lo, hi = lo - 4, hi + 2
        if lo < 0:
            out.append((n + lo, n - 1))
            lo = 0
        if hi > n - 1:
            out.append((0, hi - n))
            hi = n - 1
        out.append((lo, hi))
    return sorted(out)


def write_configs(ws, track, map_dir, raceline, kappa, lateral_offset):
    cfg_dir = ws / 'src/f1tenth_lattice_ros2/config'
    lattice = (f'# {track.name}: generate_freeform_tracks.py가 생성한 설정. '
               f'simple_augmented_lattice_config.yaml에서\n# 스폰 좌표와 follow zone만 이 맵에 맞게 바꿨다.\n'
               + (cfg_dir / 'simple_augmented_lattice_config.yaml').read_text())
    n = len(raceline)
    straight = np.convolve(np.abs(kappa) < FOLLOW_KAPPA, np.ones(30), 'same')   # start on a straight
    i1 = int(np.argmax(straight))
    for car, i in (('car1', i1), ('car2', (i1 + 20) % n)):
        j = (i + 1) % n
        yaw = math.degrees(math.atan2(raceline[j, 1] - raceline[i, 1], raceline[j, 0] - raceline[i, 0]))
        lattice = _sub_once(
            rf'({car}:.*?spawn:\n\s+x: )[-\d.]+(\n\s+y: )[-\d.]+(\n\s+yaw_deg: )[-\d.]+',
            rf'\g<1>{raceline[i, 0]:.2f}\g<2>{raceline[i, 1]:.2f}\g<3>{yaw:.1f}', lattice)
    zones = ''.join(f'      - {{ min: {lo}, max: {hi} }}\n' for lo, hi in follow_zones(kappa))
    lattice = _sub_once(r'(    zones:[^\n]*\n)(?:\s*#[^\n]*\n)?(?:\s*- \{[^}]*\}\n)+',
                        rf'\g<1>      # raceline1의 코너 구간(|kappa| > {FOLLOW_KAPPA}, 진입 4행 전 ~ 출구 2행 후)\n{zones}',
                        lattice)
    lattice = _sub_once(r'lateral_align_m: [\d.]+', f'lateral_align_m: {FOLLOW_ALIGN_M}', lattice)
    (map_dir / 'lattice_config.yaml').write_text(lattice)

    episodes = (cfg_dir / 'simple_augmented_episodes.yaml').read_text()
    container_map = f'/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps/{track.name}'
    episodes = _sub_once(r'raceline_path: [^\n]+', f'raceline_path: {container_map}/raceline1.csv', episodes)
    episodes = _sub_once(r'lateral_offset_m: [\d.]+[^\n]*',
                         f'lateral_offset_m: {lateral_offset:.1f}   # ±, 중앙 라인 최소 벽 여유 - 0.45 m', episodes)
    episodes = _sub_once(r'(car2: \{ min: )[\d.]+', rf'\g<1>{LEADER_V_SCALE_MIN}', episodes)
    episodes = _sub_once(r'base_dir: [^\n]+', f'base_dir: /root/f1tenth_ws/data/{track.name.lower()}_h2h', episodes)
    (map_dir / 'episodes.yaml').write_text(episodes)


def generate(ws, track):
    lower = track.name.lower()
    map_dir = ws / 'src/f1tenth_lattice_ros2/maps' / track.name
    map_dir.mkdir(parents=True, exist_ok=True)
    image = track.render_map()
    Image.fromarray(image).save(map_dir / f'{track.name}_map.png')
    (map_dir / f'{track.name}_map.yaml').write_text(
        f'image: {track.name}_map.png\nresolution: {RESOLUTION}\norigin: [0.0, 0.0, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n')
    triangles = track.write_walls(ws / f'src/racecar_description/meshes/{lower}.stl')
    write_world(ws / f'src/racecar_description/worlds/{lower}.world', track.name)

    edt = distance_transform_edt(image > 128) * RESOLUTION
    height = image.shape[0]
    info = {'name': track.name, 'triangles': triangles, 'lanes': []}
    for i, frac in enumerate(LANE_FRACTIONS):
        pts = track.lane(frac)
        for suffix, ccw in (('', False), ('_ccw', True)):
            length, kappa = track.write_raceline(map_dir / f'raceline{i}{suffix}.csv', pts, ccw)
        rows = np.clip((height - pts[:, 1] / RESOLUTION).astype(int), 0, height - 1)
        cols = (pts[:, 0] / RESOLUTION).astype(int)
        clearance = float(edt[rows, cols].min())
        if clearance < MIN_LANE_CLEARANCE:
            raise ValueError(f'{track.name}: raceline{i} wall clearance {clearance:.2f} m < {MIN_LANE_CLEARANCE} m')
        info['lanes'].append({'lap': length, 'points': len(pts), 'clearance': clearance,
                              'kappa_max': float(np.abs(kappa).max())})
        if i == 1:
            raceline, kappa1 = pts, kappa
    lateral_offset = max(0.3, math.floor((info['lanes'][1]['clearance'] - 0.45) * 10) / 10)
    write_configs(ws, track, map_dir, raceline, kappa1, lateral_offset)
    info.update(lateral_offset=lateral_offset, kappa=kappa1, raceline=raceline, image=image)
    return info


def overview(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = 4
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax, (track, info) in zip(axes.flat, results):
        img = info['image']
        ax.imshow(img, cmap='gray', extent=(0, img.shape[1] * RESOLUTION, 0, img.shape[0] * RESOLUTION))
        rl = info['raceline']
        sc = ax.scatter(rl[:, 0], rl[:, 1], c=np.abs(info['kappa']), s=3, cmap='viridis', vmin=0, vmax=0.6)
        ax.plot(rl[0, 0], rl[0, 1], 'r^')
        ax.set_title(f'{track.name}  w {track.width} m  |k|max {np.abs(info["kappa"]).max():.2f}\n'
                     f'lap {info["lanes"][1]["lap"]:.0f} m  share |k|>0.4: {np.mean(np.abs(info["kappa"]) > 0.4):.0%}',
                     fontsize=9)
        ax.set_aspect('equal'); ax.tick_params(labelsize=7)
        lo = rl.min(axis=0) - track.width; hi = rl.max(axis=0) + track.width
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    for ax in axes.flat[len(results):]:
        ax.axis('off')
    fig.colorbar(sc, ax=axes, shrink=0.6, label='|raceline1 curvature| (1/m)')
    fig.savefig(path, dpi=100)
    plt.close(fig)


def raceline_stats(path):
    """max |k|, share |k| > 0.4, mean and max |dk/ds| (consecutive rows, closing row included)."""
    data = np.loadtxt(path, delimiter=';', comments='#')
    k = data[:-1, 4]                       # last row repeats the first point
    dk = np.abs(np.diff(np.append(k, k[0]))) / np.diff(data[:, 0])
    return float(np.abs(k).max()), float(np.mean(np.abs(k) > 0.4)), float(dk.mean()), float(dk.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--overview', default=None,
                        help='overview image name in maps/ (default: <Config prefix>_overview.png, '
                             'e.g. Freeform_overview.png for freeform_tracks.yaml)')
    args = parser.parse_args()
    ws = args.workspace.resolve()
    cfg_path = args.config or ws / 'src/f1tenth_lattice_ros2/config/freeform_tracks.yaml'
    overview_name = args.overview or f'{cfg_path.stem.split("_")[0].capitalize()}_overview.png'
    cfg = yaml.safe_load(cfg_path.read_text())
    results, stats = [], []
    for spec in cfg['tracks']:
        if args.only and spec['name'] not in args.only:
            continue
        params = {**cfg.get('defaults', {}), **spec}
        shape = {k: v for k, v in params.items() if k not in ('name', 'width', 'kappa_max', 'seed')}
        track = FreeformTrack(params['name'], params['width'], params['kappa_max'], params['seed'], **shape)
        info = generate(ws, track)
        lane = info['lanes'][1]
        k = np.abs(info['kappa'])
        lane_k = ', '.join('%.2f' % l['kappa_max'] for l in info['lanes'])
        print(f'{track.name}: lap {lane["lap"]:.1f} m, |k| max {k.max():.2f} (lanes {lane_k}), '
              f'share>0.4 {np.mean(k > 0.4):.0%}, clearance {lane["clearance"]:.2f} m, '
              f'lateral_offset {info["lateral_offset"]}, tries {track.attempts}')
        stats.append((track.name, raceline_stats(ws / 'src/f1tenth_lattice_ros2/maps' / track.name / 'raceline1.csv')))
        results.append((track, info))
    print(f'\nraceline1 curvature    max|k|  |k|>0.4  mean|dk/ds|  max|dk/ds|')
    for name, st in stats + [('(Simple)', (0.53, 0.22, 0.064, 0.28))]:
        print(f'{name:<20} {st[0]:7.2f} {st[1]:8.0%} {st[2]:12.3f} {st[3]:11.2f}')
    overview(results, ws / 'src/f1tenth_lattice_ros2/maps' / overview_name)


if __name__ == '__main__':
    main()
