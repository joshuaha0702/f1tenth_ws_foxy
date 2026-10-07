#!/usr/bin/env python3
"""Generate a Simple-style stadium track: occupancy map, Gazebo walls, world and race lines.

Every output is derived from the same wall radius functions, so the 2D map, the
3D walls and the race lines always agree. With --wall-amplitude the inner and
outer walls get independent low-frequency undulations, so the track width
varies along the lap. Run from the workspace root; see README for the command.
"""

import argparse
import math
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


RESOLUTION = 0.02
MIN_IMAGE_SIZE = 2000
# Default centre of the bottom bend; moved outwards only for tracks too big to fit.
CX = 7.0
BOTTOM_Y = 6.5
MAP_MARGIN = 1.0
INNER_RADIUS = 1.0
WALL_HEIGHT = 0.30
WALL_THICKNESS = 0.05
SPEED = 2.0
POINT_STEP = 0.2          # race line spacing (m)
WALL_STEP = 0.05          # wall polyline spacing along the centre line (m)
LANE_FRACTIONS = (0.3, 0.5, 0.7)

# Undulation limits: long waves only, so walls stay drivable and realistic.
MAX_WAVELENGTH = 12.0
WAVE_COMPONENTS = 3
# Validity limits checked after generation.
MIN_LOCAL_WIDTH = 2.8
MIN_INNER_RADIUS = 0.5
MIN_LANE_CLEARANCE = 0.6  # car half width 0.15 + planner collision threshold 0.35 + margin


class Track:
    def __init__(self, name, track_width, straight_length, wall_amplitude=0.0,
                 min_wavelength=4.0, seed=0):
        self.name = name
        self.track_width = track_width
        self.straight_length = straight_length
        self.wall_amplitude = wall_amplitude
        self.min_wavelength = min_wavelength
        self.seed = seed
        self.inner_radius = INNER_RADIUS
        self.outer_radius = INNER_RADIUS + track_width
        self.centre_radius = INNER_RADIUS + track_width / 2.0
        reach = self.outer_radius + max(wall_amplitude, 0.0) + MAP_MARGIN
        self.cx = max(CX, reach)
        self.bottom_y = max(BOTTOM_Y, reach)
        self.top_y = self.bottom_y + straight_length
        rng = np.random.default_rng(seed)
        self.inner_wave = self._make_wave(rng)
        self.outer_wave = self._make_wave(rng)

    # --- geometry -----------------------------------------------------------

    def counts(self, step, radius=None):
        radius = self.centre_radius if radius is None else radius
        straight = max(2, math.ceil(self.straight_length / step))
        arc = max(3, math.ceil(math.pi * radius / step))
        return straight, arc

    def lap_fraction(self, straight_count, arc_count):
        """Position of each loop point as a fraction of the centre-line lap."""
        seg_s = self.straight_length / straight_count
        seg_a = math.pi * self.centre_radius / arc_count
        steps = ([seg_s] * straight_count + [seg_a] * arc_count) * 2
        s = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
        return s / sum(steps)

    def stadium_points(self, radius, straight_count, arc_count):
        """Clockwise closed loop, starting on the left straight at its bottom.

        radius is a scalar or one value per point (for undulating walls).
        """
        n = 2 * (straight_count + arc_count)
        r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,))
        pts = np.empty((n, 2))
        i = 0
        for j in range(straight_count):
            pts[i] = (self.cx - r[i], self.bottom_y + self.straight_length * j / straight_count)
            i += 1
        for j in range(arc_count):
            theta = math.pi - math.pi * j / arc_count
            pts[i] = (self.cx + r[i] * math.cos(theta), self.top_y + r[i] * math.sin(theta))
            i += 1
        for j in range(straight_count):
            pts[i] = (self.cx + r[i], self.top_y - self.straight_length * j / straight_count)
            i += 1
        for j in range(arc_count):
            theta = -math.pi * j / arc_count
            pts[i] = (self.cx + r[i] * math.cos(theta), self.bottom_y + r[i] * math.sin(theta))
            i += 1
        return pts

    def _make_wave(self, rng):
        if self.wall_amplitude <= 0.0:
            return None
        lap = 2 * self.straight_length + 2 * math.pi * self.centre_radius
        k_lo = max(1, math.ceil(lap / MAX_WAVELENGTH))
        k_hi = max(k_lo, math.floor(lap / self.min_wavelength))
        ks = rng.choice(np.arange(k_lo, k_hi + 1),
                        size=min(WAVE_COMPONENTS, k_hi - k_lo + 1), replace=False)
        phases = rng.uniform(0.0, 2 * math.pi, len(ks))
        weights = rng.uniform(0.5, 1.0, len(ks))
        # Scale so the peak displacement equals the requested amplitude.
        u = np.linspace(0.0, 1.0, 8192, endpoint=False)
        peak = np.max(np.abs(self._eval_wave((ks, phases, weights), u)))
        return ks, phases, weights * self.wall_amplitude / peak

    @staticmethod
    def _eval_wave(wave, u):
        if wave is None:
            return np.zeros_like(u)
        ks, phases, weights = wave
        return np.sum(weights[:, None] * np.sin(2 * math.pi * ks[:, None] * u[None, :]
                                                + phases[:, None]), axis=0)

    def wall_radii(self, straight_count, arc_count):
        u = self.lap_fraction(straight_count, arc_count)
        return (self.inner_radius + self._eval_wave(self.inner_wave, u),
                self.outer_radius + self._eval_wave(self.outer_wave, u))

    def walls(self):
        counts = self.counts(WALL_STEP)
        r_in, r_out = self.wall_radii(*counts)
        return counts, r_in, r_out

    def lane_radii(self):
        return [self.inner_radius + self.track_width * f for f in LANE_FRACTIONS]

    def follow_zones(self):
        """Corner index ranges of raceline1: four rows before to two rows after each corner.

        The bottom corner ends at the lap start, so its exit rows are the separate (0, 1) zone.
        """
        straight, arc = self.counts(POINT_STEP, self.lane_radii()[1])
        n = 2 * (straight + arc)
        return [(straight - 4, straight + arc + 1), (2 * straight + arc - 4, n - 1), (0, 1)]

    def image_shape(self, r_out):
        width = max(MIN_IMAGE_SIZE, math.ceil((self.cx + r_out.max() + 1.0) / RESOLUTION))
        height = max(MIN_IMAGE_SIZE, math.ceil((self.top_y + r_out.max() + 1.0) / RESOLUTION))
        return height, width

    # --- outputs ------------------------------------------------------------

    def render_map(self):
        """Occupancy image (row 0 = top), 255 free / 0 occupied."""
        _, r_in, r_out = self.walls()
        counts = self.counts(WALL_STEP)
        height, width = self.image_shape(r_out)

        def to_px(pts):
            return [(x / RESOLUTION, height - y / RESOLUTION) for x, y in pts]

        image = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(image)
        draw.polygon(to_px(self.stadium_points(r_out, *counts)), fill=255)
        draw.polygon(to_px(self.stadium_points(r_in, *counts)), fill=0)
        return np.array(image)

    def write_walls(self, path):
        counts, r_in, r_out = self.walls()
        half = WALL_THICKNESS / 2.0
        triangles = []
        for radius in (r_in, r_out):
            outer = self.stadium_points(radius + half, *counts)
            inner = self.stadium_points(radius - half, *counts)
            n = len(outer)

            def v(p, z):
                return (float(p[0]), float(p[1]), z)

            for i in range(n):
                j = (i + 1) % n
                a, b = outer[i], outer[j]
                c, d = inner[i], inner[j]
                h = WALL_HEIGHT
                triangles.extend((
                    (v(a, 0), v(b, 0), v(b, h)), (v(a, 0), v(b, h), v(a, h)),
                    (v(c, 0), v(d, h), v(d, 0)), (v(c, 0), v(c, h), v(d, h)),
                    (v(a, h), v(b, h), v(d, h)), (v(a, h), v(d, h), v(c, h)),
                    (v(a, 0), v(d, 0), v(b, 0)), (v(a, 0), v(c, 0), v(d, 0)),
                ))
        with path.open('wb') as out:
            out.write(f'{self.name} walls'.encode().ljust(80, b' ')[:80])
            out.write(struct.pack('<I', len(triangles)))
            for a, b, c in triangles:
                # Gazebo accepts zero normals and computes triangle normals.
                out.write(struct.pack('<12fH', 0, 0, 0, *a, *b, *c, 0))
        return len(triangles)

    def write_raceline(self, path, radius, counter_clockwise=False):
        pts = self.stadium_points(radius, *self.counts(POINT_STEP, radius))
        if counter_clockwise:
            # Same start point, opposite direction of travel.
            pts = np.roll(pts[::-1], 1, axis=0)
        closed = np.vstack((pts, pts[0]))
        ds = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        # Signed curvature: negative clockwise, positive counter-clockwise.
        dpsi = np.arctan2(np.sin(np.roll(heading, -1) - np.roll(heading, 1)),
                          np.cos(np.roll(heading, -1) - np.roll(heading, 1)))
        curvature = dpsi / (np.roll(ds, 1) + ds)
        with path.open('w') as out:
            out.write('# s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2\n')
            for i in range(len(pts) + 1):
                j = i % len(pts)
                out.write(f'{s[i]:.6f};{pts[j,0]:.6f};{pts[j,1]:.6f};'
                          f'{heading[j]:.6f};{curvature[j]:.6f};{SPEED:.6f};0.000000\n')
        return float(s[-1]), len(pts)


def write_world(path, name):
    lower = name.lower()
    path.write_text(f'''<?xml version="1.0"?>
<sdf version="1.4">
<world name="{lower}">
  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
    <ros><namespace>/gazebo</namespace></ros>
  </plugin>
  <physics name="default_physics" default="1" type="ode">
    <max_step_size>0.002</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>625</real_time_update_rate>
  </physics>
  <scene><ambient>0.5 0.5 0.5 1.0</ambient><shadows>0</shadows></scene>
  <include><uri>model://sun</uri><pose>0 0 15 0 0 0</pose><cast_shadows>false</cast_shadows></include>
  <model name="{lower}">
    <static>1</static>
    <link name="walls">
      <collision name="collision">
        <geometry><mesh><uri>model://racecar_description/meshes/{lower}.stl</uri></mesh></geometry>
      </collision>
      <visual name="visual">
        <geometry><mesh><uri>model://racecar_description/meshes/{lower}.stl</uri></mesh></geometry>
      </visual>
    </link>
  </model>
  <model name="ground_plane"><static>1</static><link name="link">
    <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
    <surface><friction><ode><mu>1000</mu><mu2>500</mu2></ode></friction></surface></collision>
    <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></visual>
  </link></model>
</world>
</sdf>
''')


def _sub_once(pattern, repl, text):
    new, count = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f'config template no longer matches: {pattern}')
    return new


def write_configs(ws, track, map_dir, lateral_offset):
    """Per-map lattice/episode configs, based on the Simple_augmented sim-collection configs."""
    cfg_dir = ws / 'src/f1tenth_lattice_ros2/config'
    lattice = (cfg_dir / 'simple_augmented_lattice_config.yaml').read_text()
    header = (f'# {track.name}: generate_simple_augmented.py가 생성한 설정. '
              f'simple_augmented_lattice_config.yaml에서\n# 스폰 좌표와 follow zone만 이 맵에 맞게 바꿨다.\n')
    lattice = header + lattice
    x = track.cx - track.lane_radii()[1]
    for car, y in (('car1', track.bottom_y + 1.5), ('car2', track.bottom_y + 5.5)):
        lattice = _sub_once(
            rf'({car}:.*?spawn:\n\s+x: )[-\d.]+(\n\s+y: )[-\d.]+(\n\s+yaw_deg: )[-\d.]+',
            rf'\g<1>{x:.2f}\g<2>{y:.2f}\g<3>90.0', lattice)
    zones = ''.join(f'      - {{ min: {lo}, max: {hi} }}\n' for lo, hi in track.follow_zones())
    lattice = _sub_once(r'(    zones:[^\n]*\n)(?:\s*#[^\n]*\n)?(?:\s*- \{[^}]*\}\n)+',
                        rf'\g<1>      # raceline1의 코너 구간(진입 4행 전 ~ 출구 2행 후)\n{zones}', lattice)
    (map_dir / 'lattice_config.yaml').write_text(lattice)

    episodes = (cfg_dir / 'simple_augmented_episodes.yaml').read_text()
    container_map = f'/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps/{track.name}'
    episodes = _sub_once(r'raceline_path: [^\n]+', f'raceline_path: {container_map}/raceline1.csv', episodes)
    episodes = _sub_once(r'lateral_offset_m: [\d.]+[^\n]*',
                         f'lateral_offset_m: {lateral_offset:.1f}   # ±, 중앙 라인 최소 벽 여유 - 0.45 m',
                         episodes)
    episodes = _sub_once(r'base_dir: [^\n]+',
                         f'base_dir: /root/f1tenth_ws/data/{track.name.lower()}_h2h', episodes)
    (map_dir / 'episodes.yaml').write_text(episodes)


def generate(ws, track, with_configs=False):
    """Write all outputs for one track; raises ValueError when the track is not drivable."""
    _, r_in, r_out = track.walls()
    local_width = r_out - r_in
    if local_width.min() < MIN_LOCAL_WIDTH:
        raise ValueError(f'{track.name}: local width {local_width.min():.2f} m < {MIN_LOCAL_WIDTH} m')
    if r_in.min() < MIN_INNER_RADIUS:
        raise ValueError(f'{track.name}: inner wall radius {r_in.min():.2f} m < {MIN_INNER_RADIUS} m')
    if track.cx - r_out.max() <= 0.5 or track.bottom_y - r_out.max() <= 0.5:
        raise ValueError(f'{track.name}: outer wall leaves the map')

    lower = track.name.lower()
    map_dir = ws / 'src/f1tenth_lattice_ros2/maps' / track.name
    mesh = ws / f'src/racecar_description/meshes/{lower}.stl'
    world = ws / f'src/racecar_description/worlds/{lower}.world'
    map_dir.mkdir(parents=True, exist_ok=True)

    image = track.render_map()
    Image.fromarray(image).save(map_dir / f'{track.name}_map.png')
    (map_dir / f'{track.name}_map.yaml').write_text(
        f'image: {track.name}_map.png\nresolution: {RESOLUTION}\norigin: [0.0, 0.0, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n')
    triangles = track.write_walls(mesh)
    write_world(world, track.name)

    # Clearance of every race line, measured on the rasterised map itself.
    edt = distance_transform_edt(image > 128) * RESOLUTION
    height = image.shape[0]
    lanes = []
    for i, radius in enumerate(track.lane_radii()):
        for suffix, ccw in (('', False), ('_ccw', True)):
            path = map_dir / f'raceline{i}{suffix}.csv'
            length, rows = track.write_raceline(path, radius, ccw)
        pts = np.loadtxt(map_dir / f'raceline{i}.csv', delimiter=';', comments='#')[:, 1:3]
        rows_px = np.clip((height - pts[:, 1] / RESOLUTION).astype(int), 0, height - 1)
        cols_px = (pts[:, 0] / RESOLUTION).astype(int)
        clearance = float(edt[rows_px, cols_px].min())
        if clearance < MIN_LANE_CLEARANCE:
            raise ValueError(f'{track.name}: raceline{i} wall clearance {clearance:.2f} m '
                             f'< {MIN_LANE_CLEARANCE} m')
        lanes.append({'radius': radius, 'lap': length, 'points': rows, 'clearance': clearance})

    lateral_offset = max(0.3, math.floor((lanes[1]['clearance'] - 0.45) * 10) / 10)
    if with_configs:
        write_configs(ws, track, map_dir, lateral_offset)
    return {
        'name': track.name, 'map_dir': map_dir, 'mesh': mesh, 'world': world,
        'triangles': triangles, 'lanes': lanes, 'lateral_offset': lateral_offset,
        'width_min': float(local_width.min()), 'width_max': float(local_width.max()),
        'follow_zones': track.follow_zones(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--name', default='Simple_augmented',
                        help='Map name; outputs go to maps/<name>, meshes/<name lower>.stl, '
                             'worlds/<name lower>.world')
    parser.add_argument('--track-width', type=float, default=4.0,
                        help='Nominal free track width in metres (default: 4.0)')
    parser.add_argument('--straight-length', type=float, default=20.5,
                        help='Distance between bend centres in metres (default: 20.5)')
    parser.add_argument('--wall-amplitude', type=float, default=0.0,
                        help='Peak wall displacement of the undulations in metres (default: 0)')
    parser.add_argument('--min-wavelength', type=float, default=4.0,
                        help='Shortest undulation wavelength in metres (default: 4.0)')
    parser.add_argument('--seed', type=int, default=0, help='Undulation random seed')
    parser.add_argument('--configs', action='store_true',
                        help='Also write maps/<name>/lattice_config.yaml and episodes.yaml')
    args = parser.parse_args()
    if args.track_width <= 1.0 or args.straight_length <= 1.0:
        parser.error('track width and straight length must exceed 1 m')
    track = Track(args.name, args.track_width, args.straight_length,
                  args.wall_amplitude, args.min_wavelength, args.seed)
    try:
        info = generate(args.workspace.resolve(), track, args.configs)
    except ValueError as e:
        parser.error(str(e))
    for i, lane in enumerate(info['lanes']):
        print(f'raceline{i}(+_ccw): radius={lane["radius"]:.2f} m, lap={lane["lap"]:.2f} m, '
              f'points={lane["points"]}, wall clearance={lane["clearance"]:.2f} m')
    print(f'{info["name"]}: width {info["width_min"]:.2f}~{info["width_max"]:.2f} m, '
          f'straight={args.straight_length:.1f} m, mesh={info["mesh"]} ({info["triangles"]} triangles)')


if __name__ == '__main__':
    main()
