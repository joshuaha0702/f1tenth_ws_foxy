#!/usr/bin/env python3
"""
Interactive raceline inspector.

Overlays a raceline (and optional lane centerlines) on top of the 2D occupancy
map in world coordinates. Hovering the mouse over the raceline highlights the
nearest point and shows its row index and (x, y) value.

Usage:
    # Auto-resolve map / raceline inside the maps directory
    python3 inspect_raceline.py --map_name Simple --raceline raceline0

    # Or give explicit paths
    python3 inspect_raceline.py \
        --map_yaml ../maps/Simple_map.yaml \
        --raceline_csv ../maps/raceline0.csv

Controls:
    - Move the mouse near the raceline -> shows row index, x, y (and vx/kappa).
    - Click on a point            -> prints the same info to the terminal.
"""

import argparse
import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_arguments():
    here = os.path.dirname(os.path.abspath(__file__))
    default_maps = os.path.normpath(os.path.join(here, '..', 'maps'))

    parser = argparse.ArgumentParser(description='Hover over a raceline to read its row index and x/y.')
    parser.add_argument('--maps_dir', type=str, default=default_maps,
                        help='Directory containing the map and raceline files.')
    parser.add_argument('--map_name', type=str, default='Simple',
                        help='Map name; resolves to {maps_dir}/{map_name}_map.yaml.')
    parser.add_argument('--map_yaml', type=str, default=None,
                        help='Explicit path to the map .yaml (overrides --map_name).')
    parser.add_argument('--raceline', type=str, default='raceline1',
                        help='Raceline file stem; resolves to {maps_dir}/{raceline}.csv.')
    parser.add_argument('--raceline_csv', type=str, default=None,
                        help='Explicit path to the raceline .csv (overrides --raceline).')
    parser.add_argument('--lanes', nargs='*', default=None,
                        help='Optional lane stems to overlay, e.g. --lanes lane0 lane1.')
    parser.add_argument('--pick_radius', type=float, default=0.5,
                        help='Max world-distance (m) to snap the cursor to a point.')
    return parser.parse_args()


def load_map(map_yaml_path):
    """Load the occupancy map image and compute its world-coordinate extent.

    Returns (image, extent) where extent = [xmin, xmax, ymin, ymax] in meters.
    The transform matches transform_coords() in generate_raceline.py:
        world_x = px * res + origin_x
        world_y = (height - py) * res + origin_y
    """
    with open(map_yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    res = meta['resolution']
    origin_x, origin_y = meta['origin'][0], meta['origin'][1]

    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(map_yaml_path), img_path)
    img = mpimg.imread(img_path)
    h, w = img.shape[0], img.shape[1]

    xmin = origin_x
    xmax = origin_x + w * res
    ymin = origin_y
    ymax = origin_y + h * res
    return img, [xmin, xmax, ymin, ymax]


def load_raceline(csv_path):
    """Load a raceline CSV (semicolon-delimited).

    Columns: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    Returns the full Nx7 array (row index = array row).
    """
    data = np.genfromtxt(csv_path, delimiter=';', comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_lane(csv_path):
    """Load a lane centerline CSV (comma-delimited): x_m, y_m, w_right, w_left."""
    return np.genfromtxt(csv_path, delimiter=',', comments='#')


def main():
    args = parse_arguments()

    map_yaml = args.map_yaml or os.path.join(args.maps_dir, f'{args.map_name}_map.yaml')
    raceline_csv = args.raceline_csv or os.path.join(args.maps_dir, f'{args.raceline}.csv')

    if not os.path.exists(map_yaml):
        raise FileNotFoundError(f'Map yaml not found: {map_yaml}')
    if not os.path.exists(raceline_csv):
        raise FileNotFoundError(f'Raceline csv not found: {raceline_csv}')

    img, extent = load_map(map_yaml)
    rl = load_raceline(raceline_csv)
    xs, ys = rl[:, 1], rl[:, 2]

    fig, ax = plt.subplots(figsize=(10, 10))
    # origin='upper': array row 0 sits at the top (max world y), matching the
    # (height - py) flip used when the raceline was generated.
    ax.imshow(img, extent=extent, origin='upper', cmap='gray', zorder=0)

    # Optional lane overlays
    if args.lanes:
        for stem in args.lanes:
            lane_path = os.path.join(args.maps_dir, f'{stem}.csv')
            if os.path.exists(lane_path):
                lane = load_lane(lane_path)
                ax.plot(lane[:, 0], lane[:, 1], '--', lw=0.8, alpha=0.6, label=stem)

    # Raceline colored by velocity if available
    sc = ax.scatter(xs, ys, c=rl[:, 5], cmap='viridis', s=8, zorder=2,
                    label=os.path.basename(raceline_csv))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('vx [m/s]')

    # Hover artifacts: a highlight marker + a text annotation
    highlight, = ax.plot([], [], 'o', ms=12, mfc='none', mec='red', mew=2, zorder=3)
    annot = ax.annotate(
        '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
        bbox=dict(boxstyle='round', fc='yellow', ec='black', alpha=0.9),
        arrowprops=dict(arrowstyle='->'), fontsize=9, zorder=4,
    )
    annot.set_visible(False)

    pts = np.column_stack([xs, ys])

    def nearest_index(mx, my):
        d2 = (pts[:, 0] - mx) ** 2 + (pts[:, 1] - my) ** 2
        i = int(np.argmin(d2))
        return i, np.sqrt(d2[i])

    def format_info(i):
        return (f'idx: {i}\n'
                f'x: {rl[i, 1]:.3f} m\n'
                f'y: {rl[i, 2]:.3f} m\n'
                f's: {rl[i, 0]:.3f} m\n'
                f'vx: {rl[i, 5]:.2f} m/s\n'
                f'kappa: {rl[i, 4]:.3f}')

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            if annot.get_visible():
                annot.set_visible(False)
                highlight.set_data([], [])
                fig.canvas.draw_idle()
            return
        i, dist = nearest_index(event.xdata, event.ydata)
        if dist <= args.pick_radius:
            annot.xy = (rl[i, 1], rl[i, 2])
            annot.set_text(format_info(i))
            annot.set_visible(True)
            highlight.set_data([rl[i, 1]], [rl[i, 2]])
        else:
            annot.set_visible(False)
            highlight.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        i, dist = nearest_index(event.xdata, event.ydata)
        if dist <= args.pick_radius:
            print(f'[click] idx={i}  x={rl[i, 1]:.4f}  y={rl[i, 2]:.4f}  '
                  f's={rl[i, 0]:.4f}  vx={rl[i, 5]:.3f}  kappa={rl[i, 4]:.4f}')

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{os.path.basename(raceline_csv)}  ({len(rl)} points)  '
                 f'- hover to read row index / x / y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
