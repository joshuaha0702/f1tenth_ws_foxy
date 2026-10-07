#!/usr/bin/env python3
"""Generate every Simple variant listed in config/simple_variants.yaml and render each one.

For each variant this writes the occupancy map, Gazebo walls/world, race lines and
per-map lattice/episode configs (see generate_simple_augmented.py), then renders
maps/<name>/<name>_render.png: a 3D view of the wall mesh, a top view with race
lines, follow zones and spawn points, and the local track width along the lap.
maps/Simple_variants_overview.png shows all variants at the same scale.
Run from the workspace root; see README for the exact command.
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import yaml

from generate_simple_augmented import (POINT_STEP, RESOLUTION, WALL_HEIGHT,
                                       Track, generate)


SURFACE = '#fcfcfb'
INK = '#2b2b28'
INK_MUTED = '#6b6a64'
GRID = '#e4e3dc'
FREE = '#ecebe5'
WALL = '#8d8b82'
LANE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')   # categorical slots 1-3
Z_EXAGGERATION = 3.0
VIEW_AZIMUTH = -125.0    # camera direction seen from the track centre (deg)
VIEW_ELEVATION = 35.0

plt.rcParams.update({
    'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK_MUTED,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'axes.edgecolor': GRID,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})


def load_lanes(map_dir):
    return [np.loadtxt(map_dir / f'raceline{i}.csv', delimiter=';', comments='#')
            for i in range(3)]


def read_stl(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], dtype='<u4')[0])
    rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([
        ('normal', '<3f4'), ('v', '<9f4'), ('attr', '<u2')]))
    return rec['v'].reshape(n, 3, 3).astype(np.float64)


def draw_3d(ax, track, info, lanes):
    """Orthographic render of the wall mesh (painter's algorithm, Lambert shading).

    Done by hand instead of mplot3d so the axes keep true proportions.
    """
    tris = read_stl(info['mesh'])
    tris[:, :, 2] *= Z_EXAGGERATION
    centre = np.array([track.cx, (track.bottom_y + track.top_y) / 2.0, 0.0])
    az, el = math.radians(VIEW_AZIMUTH), math.radians(VIEW_ELEVATION)
    to_camera = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(to_camera, right)

    def project(points):
        rel = points - centre
        return np.stack((rel @ right, rel @ up), axis=-1), rel @ to_camera

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(normals @ light)
    base = np.array(matplotlib.colors.to_rgb(WALL))
    colors = np.clip(base[None, :] * shade[:, None] * 1.25, 0, 1)

    screen, depth = project(tris)
    order = np.argsort(depth.mean(axis=1))          # far to near

    x0, x1 = tris[:, :, 0].min() - 1.5, tris[:, :, 0].max() + 1.5
    y0, y1 = tris[:, :, 1].min() - 1.5, tris[:, :, 1].max() + 1.5
    ground, _ = project(np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]))
    ax.add_collection(PolyCollection([ground], facecolors=FREE, edgecolors='none'))
    # Ground-level lines are drawn before the walls; walls in front then hide them correctly.
    for lane, color in zip(lanes, LANE_COLORS):
        pts, _ = project(np.column_stack((lane[:, 1], lane[:, 2], np.zeros(len(lane)))))
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.4)
    ax.add_collection(PolyCollection(screen[order], facecolors=colors[order],
                                     edgecolors=colors[order], linewidths=0.2))
    ax.set_xlim(ground[:, 0].min(), ground[:, 0].max())
    ax.set_ylim(ground[:, 1].min(), screen[:, :, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'3D wall mesh, orthographic view ({info["triangles"]:,} triangles, '
                 f'wall height x{Z_EXAGGERATION:.0f})', color=INK, fontsize=10, loc='left')


def draw_top(ax, track, info, lanes, image):
    height, width = image.shape
    extent = (0, width * RESOLUTION, 0, height * RESOLUTION)
    cmap = matplotlib.colors.ListedColormap([WALL, FREE])
    ax.imshow(image > 128, cmap=cmap, extent=extent, origin='upper', interpolation='nearest')
    for i, (lane, color) in enumerate(zip(lanes, LANE_COLORS)):
        ax.plot(lane[:, 1], lane[:, 2], color=color, lw=1.4,
                label=f'raceline{i} (lap {lane[-1, 0]:.1f} m)')
    centre = lanes[1]
    for k, (lo, hi) in enumerate(info['follow_zones']):
        ax.plot(centre[lo:hi + 1, 1], centre[lo:hi + 1, 2], color=LANE_COLORS[1], lw=6,
                alpha=0.25, solid_capstyle='butt', label='follow zone' if k == 0 else None)
    x = track.cx - track.lane_radii()[1]
    for label, y, marker in (('car1 spawn', track.bottom_y + 1.5, 'o'), ('car2 spawn', track.bottom_y + 5.5, 's')):
        ax.plot(x, y, marker=marker, ms=8, color=INK, mfc=SURFACE, mew=1.6, ls='none', label=label)
    xs = np.concatenate([lane[:, 1] for lane in lanes])
    ys = np.concatenate([lane[:, 2] for lane in lanes])
    pad = track.track_width / 2 + 1.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Top view: occupancy map and race lines', color=INK, fontsize=10, loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)


def draw_width(ax, track):
    counts = track.counts(POINT_STEP / 4)
    r_in, r_out = track.wall_radii(*counts)
    lap = 2 * track.straight_length + 2 * math.pi * track.centre_radius
    s = track.lap_fraction(*counts) * lap
    ax.plot(s, r_out - r_in, color=LANE_COLORS[0], lw=2)
    ax.axhline(track.track_width, color=INK_MUTED, lw=1, ls='--')
    ax.text(lap, track.track_width, ' nominal', color=INK_MUTED, va='center', fontsize=8)
    ax.axhline(2.8, color=INK_MUTED, lw=1, ls=':')
    ax.text(lap, 2.8, ' min allowed', color=INK_MUTED, va='center', fontsize=8)
    straight, arc = counts
    for idx in (straight, straight + arc, 2 * straight + arc):
        ax.axvline(s[idx], color=GRID, lw=1)
    ax.set_xlim(0, lap)
    ax.set_ylim(2.6, max(track.track_width + track.wall_amplitude * 2 + 0.2, 3.6))
    ax.set_xlabel('distance along centre line from lap start (m)  |  grey lines: straight/corner boundaries')
    ax.set_ylabel('track width (m)')
    ax.grid(axis='y', color=GRID, lw=0.8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_title('Local track width along the lap', color=INK, fontsize=10, loc='left')


def render(track, info, out_path):
    lanes = load_lanes(info['map_dir'])
    image = np.array(plt.imread(str(info['map_dir'] / f'{track.name}_map.png')))
    image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=(4, 1.2), width_ratios=(1.4, 1.4, 1.0),
                            hspace=0.28, wspace=0.05)
    draw_3d(fig.add_subplot(grid[0, 0:2]), track, info, lanes)
    draw_top(fig.add_subplot(grid[0, 2]), track, info, lanes, image)
    draw_width(fig.add_subplot(grid[1, :]), track)
    fig.suptitle(
        f'{track.name}   width {track.track_width:.1f} m (local {info["width_min"]:.2f}~'
        f'{info["width_max"]:.2f} m)   straight {track.straight_length:.1f} m   '
        f'wall undulation ±{track.wall_amplitude:.2f} m (λ≥{track.min_wavelength:.0f} m, seed {track.seed})',
        x=0.02, ha='left', fontsize=13, color=INK)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def render_overview(results, out_path):
    cols = 10
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 5.4 * rows), squeeze=False)
    y_max = max(t.top_y + t.outer_radius + t.wall_amplitude for t, _ in results) + 0.5
    x_max = max(t.cx + t.outer_radius + t.wall_amplitude for t, _ in results) + 0.5
    x_min = min(t.cx - t.outer_radius - t.wall_amplitude for t, _ in results) - 0.5
    y_min = min(t.bottom_y - t.outer_radius - t.wall_amplitude for t, _ in results) - 0.5
    cmap = matplotlib.colors.ListedColormap([WALL, FREE])
    for ax, (track, info) in zip(axes.flat, results):
        image = plt.imread(str(info['map_dir'] / f'{track.name}_map.png'))
        h, w = image.shape[:2]
        ax.imshow(image > 0.5, cmap=cmap, extent=(0, w * RESOLUTION, 0, h * RESOLUTION),
                  origin='upper', interpolation='nearest')
        lane = load_lanes(info['map_dir'])[1]
        ax.plot(lane[:, 1], lane[:, 2], color=LANE_COLORS[1], lw=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(WALL)   # same backdrop where a smaller map image ends
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'{track.name[-3:]}  W{track.track_width:.1f}  L{track.straight_length:.1f}\n'
                     f'±{track.wall_amplitude:.2f}  lap {lane[-1, 0]:.1f} m', fontsize=9, color=INK)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.suptitle('Simple variants (same scale; W = nominal width, L = straight length, '
                 '± = wall undulation, all in m; orange = raceline1)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.08)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--manifest', type=Path, default=None,
                        help='default: <workspace>/src/f1tenth_lattice_ros2/config/simple_variants.yaml')
    parser.add_argument('--only', nargs='*', help='Generate only these variant names')
    args = parser.parse_args()
    ws = args.workspace.resolve()
    manifest = args.manifest or ws / 'src/f1tenth_lattice_ros2/config/simple_variants.yaml'
    spec = yaml.safe_load(manifest.read_text())
    defaults = spec.get('defaults', {})

    results = []
    for entry in spec['variants']:
        if args.only and entry['name'] not in args.only:
            continue
        params = {**defaults, **entry}
        track = Track(params['name'], float(params['track_width']), float(params['straight_length']),
                      float(params.get('wall_amplitude', 0.0)),
                      float(params.get('min_wavelength', 4.0)), int(params.get('seed', 0)))
        info = generate(ws, track, with_configs=True)
        render(track, info, info['map_dir'] / f'{track.name}_render.png')
        results.append((track, info))
        clear = ', '.join(f'{lane["clearance"]:.2f}' for lane in info['lanes'])
        print(f'{track.name}: width {info["width_min"]:.2f}~{info["width_max"]:.2f} m, '
              f'straight {track.straight_length:.1f} m, lap {info["lanes"][1]["lap"]:.1f} m, '
              f'lane clearance [{clear}] m, zones {info["follow_zones"]}, '
              f'lateral ±{info["lateral_offset"]:.1f} m')
    if results and not args.only:
        render_overview(results, ws / 'src/f1tenth_lattice_ros2/maps/Simple_variants_overview.png')
    print(f'generated {len(results)} variants')


if __name__ == '__main__':
    main()
