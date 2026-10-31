"""Plot actual and open-loop predicted trajectories from run_model_for_div.py.

The script accepts either an individual direction directory (for example,
``asphalt_div/forward``) or a parent directory containing ``forward`` and
``backward`` runs.  With no paths, it discovers ``*_div`` directories in the
current working directory.  By default, each plot shows the first five
seconds of the saved open-loop rollout.

Examples:
    python -m experiments.exp_008_beamng.plot_model_divergence asphalt_div
    python -m experiments.exp_008_beamng.plot_model_divergence \
        --combine-environments
    python -m experiments.exp_008_beamng.plot_model_divergence snow_div/forward
    python -m experiments.exp_008_beamng.plot_model_divergence \
        asphalt_div/forward --expert learned_h4 --log-errors --show
    python -m experiments.exp_008_beamng.plot_model_divergence \
        asphalt_div/forward --full-rollout
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
import struct

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import to_rgba
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image
from experiments.exp_008_beamng.experiment_paths import LEVELS_DIR, RENDERED_BACKGROUNDS, resolve_saved_path


FONT_SCALE = 2.7
plt.rcParams.update(
    {
        "font.size": 10 * FONT_SCALE,
        "axes.titlesize": 12 * FONT_SCALE,
        "axes.labelsize": 10 * FONT_SCALE,
        "xtick.labelsize": 9 * FONT_SCALE,
        "ytick.labelsize": 9 * FONT_SCALE,
        "legend.fontsize": 9 * FONT_SCALE,
    }
)


PREDICTION_RE = re.compile(r"prediction_expert-(.+)_model-(.+)\.npz$")
REQUIRED_CHANNELS = (
    "x",
    "y",
    "yaw_tractor",
    "yaw_trailer",
    "vx",
    "vy",
)
MODEL_COLORS = {
    "learned_h1": "tab:purple",
    "learned_h4": "tab:blue",
    "learned": "tab:blue",
    "fiala_surface": "tab:orange",
    "fiala": "tab:green",
}
MODEL_LABELS = {
    "fiala": "Analytic",
    "fiala_surface": "Friction-aware Analytic",
    "learned_h4": "Learned",
    "learned": "Learned",
    "learned_h1": "Learned (H=1)",
}
DEFAULT_PLOT_SECONDS = 5.0
DEFAULT_LEVELS_DIR = LEVELS_DIR
TERRAIN_PATHS = {
    "slope": Path("tech_ground_slope/terrain8.ter"),
    "snow": Path("tech_ground_snow/terrain8.ter"),
    "both": Path("tech_ground_final/tech_ground/terrain8.ter"),
}
TERRAIN_EXTENT = (-1024.0, 1024.0, -1024.0, 1024.0)
TERRAIN_MARGIN_M = 20.0
SNOW_IDX = 4
COMBINED_ENVIRONMENTS = (
    ("asphalt_div", "Asphalt"),
    ("snow_div", "Snow"),
    ("slope_div", "Slope"),
    ("both4_div", "Snow + slope"),
)


@lru_cache(maxsize=8)
def load_rendered_background(environment: str, direction: str):
    """Load a true orthographic BeamNG render with its world-coordinate extent."""
    metadata_path = RENDERED_BACKGROUNDS / f'{environment}_{direction}.json'
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata['projection'] != 'orthographic' or metadata['origin'] != 'upper':
        raise ValueError(f'Unsupported background georeferencing: {metadata_path}')
    with Image.open(metadata_path.parent / metadata['image']) as image:
        rgb = np.asarray(image.convert('RGB'))
    return rgb, metadata


def add_rendered_background(ax, environment, direction, transform=None):
    """Draw the rendered landscape; return False when no capture is available."""
    rendered = load_rendered_background(environment, direction)
    if rendered is None:
        return False
    rgb, metadata = rendered
    ax.imshow(rgb, extent=metadata['extent'], origin='upper',
              transform=transform if transform is not None else ax.transData,
              interpolation='lanczos', zorder=0)
    return True


def endpoint_rig_geometry(pose, vehicle):
    """Schematic footprint from tractor XY, both yaws and the saved hitch geometry.

    Trailer XY is reconstructed, since it is not stored in the rollout states.
    All coordinates remain in metres; silhouettes are never enlarged or offset.
    """
    x, y, tractor_yaw, trailer_yaw = np.asarray(pose, dtype=float)
    length, width = vehicle.get('chassis_size', [3.2, 1.5])[:2]
    hitch_offset = float(vehicle.get('hitch_offset', 2.0))
    hitch_to_axle = float(vehicle.get('l2f', 2.8))
    rear_overhang = float(vehicle.get('l2r', .3))
    tractor = np.array([x, y])
    heading = np.array([np.cos(tractor_yaw), np.sin(tractor_yaw)])
    hitch = tractor - hitch_offset * heading

    def coordinates(points, origin, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.asarray(points) @ np.array([[c, s], [-s, c]]) + origin

    def rectangle(front, rear, half_width, origin, yaw):
        return coordinates([[front, half_width], [front, -half_width],
                            [rear, -half_width], [rear, half_width]], origin, yaw)

    front, rear = length / 2, -length / 2
    tractor_body = coordinates([[front, .35 * width], [front - .2, .5 * width],
                                [rear, .5 * width], [rear, -.5 * width],
                                [front - .2, -.5 * width], [front, -.35 * width]],
                               tractor, tractor_yaw)
    trailer_front = -.3 * hitch_to_axle
    trailer_rear = -(hitch_to_axle + rear_overhang)
    trailer_body = rectangle(trailer_front, trailer_rear, width / 2, hitch, trailer_yaw)
    cab = rectangle(.55 * front, -.05 * length, .4 * width, tractor, tractor_yaw)
    wheels = []
    for origin, yaw, axles in ((tractor, tractor_yaw, [vehicle.get('lf', 1.35), -vehicle.get('lr', 1.65)]),
                               (hitch, trailer_yaw, [-hitch_to_axle])):
        for axle in axles:
            for side in (-1, 1):
                wheels.append(coordinates([[axle + .3, side * width / 2 + .12],
                                            [axle + .3, side * width / 2 - .12],
                                            [axle - .3, side * width / 2 - .12],
                                            [axle - .3, side * width / 2 + .12]], origin, yaw))
    tongue = coordinates([[trailer_front, .35 * width], [0, 0],
                          [trailer_front, -.35 * width]], hitch, trailer_yaw)
    coupler = np.stack([coordinates([[rear, 0]], tractor, tractor_yaw)[0], hitch])
    return dict(bodies=[tractor_body, trailer_body], cab=cab, wheels=wheels,
                tongue=tongue, coupler=coupler)


def endpoint_rig_vertices(pose, vehicle):
    geometry = endpoint_rig_geometry(pose, vehicle)
    return np.concatenate(geometry['bodies'] + geometry['wheels'] +
                          [geometry['cab'], geometry['tongue'], geometry['coupler']])


def draw_endpoint_rig(ax, pose, vehicle, color, transform=None):
    """Draw one color-matched tractor/trailer at the unmodified final pose."""
    geometry = endpoint_rig_geometry(pose, vehicle)
    transform = ax.transData if transform is None else transform
    outline = [path_effects.Stroke(linewidth=3.0, foreground='white', alpha=.85),
               path_effects.Normal()]
    for points in geometry['bodies']:
        patch = Polygon(points, closed=True, facecolor=to_rgba(color, .22),
                        edgecolor=color, linewidth=1.8, transform=transform, zorder=12,
                        joinstyle='round')
        patch.set_path_effects(outline)
        ax.add_patch(patch)
    ax.add_patch(Polygon(geometry['cab'], closed=True, facecolor=to_rgba('white', .65),
                         edgecolor=color, linewidth=.9, transform=transform, zorder=13))
    for points in geometry['wheels']:
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor='white',
                             linewidth=.45, transform=transform, zorder=14))
    for points in (geometry['tongue'], geometry['coupler']):
        line, = ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.4,
                        transform=transform, zorder=11)
        line.set_path_effects(outline)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_terrain(path: Path, max_height: float = 256.0) -> tuple[np.ndarray, np.ndarray]:
    """Parse the BeamNG/Torque3D v9 height and material maps from a .ter file."""
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 1)[0]
    count = size * size
    offset = 5
    height = np.frombuffer(data, "<u2", count, offset).reshape(size, size).copy()
    offset += 2 * count
    layer = np.frombuffer(data, np.uint8, count, offset).reshape(size, size).copy()
    return height.astype(np.float32) * max_height / 65535.0, layer


def spatial_limits(states: list[np.ndarray], x: int, y: int, extra_points=None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return common trajectory limits with enough context for terrain contours."""
    xs = np.concatenate([state[:, x] for state in states])
    ys = np.concatenate([state[:, y] for state in states])
    if extra_points is not None:
        xs = np.r_[xs, extra_points[:, 0]]
        ys = np.r_[ys, extra_points[:, 1]]
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if not len(xs) or not len(ys):
        raise ValueError("spatial rollout contains no finite positions")
    x_pad = max(TERRAIN_MARGIN_M, 0.08 * np.ptp(xs))
    y_pad = max(TERRAIN_MARGIN_M, 0.08 * np.ptp(ys))
    xlim = (float(xs.min() - x_pad), float(xs.max() + x_pad))
    ylim = (float(ys.min() - y_pad), float(ys.max() + y_pad))
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    return (x_mid - span * 0.5, x_mid + span * 0.5), (
        y_mid - span * 0.5,
        y_mid + span * 0.5,
    )


def add_environment_background(
    ax: plt.Axes,
    environment: str,
    terrain: tuple[np.ndarray, np.ndarray] | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list:
    """Draw the slope contours and snow layer in make_env_fig.py's style."""
    if terrain is None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return []

    height, layer = terrain
    size = height.shape[0]
    world = np.linspace(TERRAIN_EXTENT[0], TERRAIN_EXTENT[1], size)
    col0 = max(0, int(np.searchsorted(world, xlim[0], side="left")) - 1)
    col1 = min(size, int(np.searchsorted(world, xlim[1], side="right")) + 1)
    row0 = max(0, int(np.searchsorted(world, ylim[0], side="left")) - 1)
    row1 = min(size, int(np.searchsorted(world, ylim[1], side="right")) + 1)
    cropped_height = height[row0:row1, col0:col1]
    cropped_layer = layer[row0:row1, col0:col1]
    crop_x = world[col0:col1]
    crop_y = world[row0:row1]

    if environment in ("slope", "both"):
        lo, hi = float(np.nanmin(height)), float(np.nanmax(height))
        ax.contour(
            crop_x,
            crop_y,
            cropped_height,
            levels=np.arange(lo, hi, 5.0),
            colors="#9f8d69",
            linewidths=0.7,
            alpha=0.9,
            zorder=1,
        )
        ax.contour(
            crop_x,
            crop_y,
            cropped_height,
            levels=np.arange(lo, hi, 25.0),
            colors="#6f5c3e",
            linewidths=1.4,
            alpha=1.0,
            zorder=2,
        )

    if environment in ("snow", "both"):
        snow = np.ma.masked_where(cropped_layer != SNOW_IDX, np.ones_like(cropped_height))
        image_extent = [crop_x[0], crop_x[-1], crop_y[0], crop_y[-1]]
        ax.imshow(
            snow,
            origin="lower",
            extent=image_extent,
            cmap="Blues_r",
            vmin=0,
            vmax=1.6,
            alpha=0.55,
            interpolation="nearest",
            zorder=3,
        )

    environment_handles = []
    if environment in ("snow", "both"):
        environment_handles.append(
            Patch(fc="#a8cfec", ec="0.6", lw=0.4, alpha=0.55, label="Snow")
        )
    if environment in ("slope", "both"):
        environment_handles.append(
            Line2D([], [], color="#6f5c3e", lw=3.0, label="Contours (5/25 m)")
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return environment_handles


def find_run_directories(paths: list[Path]) -> list[Path]:
    """Resolve input paths to directories containing prediction archives."""
    if not paths:
        paths = sorted(Path.cwd().glob("*_div"))
        default_root = Path("experiments/exp_008_beamng/model_divergence_out")
        if not paths and default_root.exists():
            paths = [default_root]

    run_dirs: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        if path.is_file():
            raise ValueError(f"expected a directory, got a file: {path}")
        if any(path.glob("prediction_expert-*_model-*.npz")):
            run_dirs.add(path.resolve())
            continue
        run_dirs.update(
            archive.parent.resolve()
            for archive in path.rglob("prediction_expert-*_model-*.npz")
        )

    if not run_dirs:
        shown = ", ".join(map(str, paths)) or "the current directory"
        raise FileNotFoundError(f"no prediction archives found under {shown}")
    return sorted(run_dirs)


def load_predictions(run_dir: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Return prediction data grouped as expert -> predictor -> arrays."""
    grouped: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for archive in sorted(run_dir.glob("prediction_expert-*_model-*.npz")):
        match = PREDICTION_RE.fullmatch(archive.name)
        if match is None:
            continue
        expert, predictor = match.groups()
        with np.load(archive) as saved:
            grouped.setdefault(expert, {})[predictor] = {
                key: np.asarray(saved[key]) for key in saved.files
            }
    return grouped


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_track(manifest: dict, run_dir: Path) -> np.ndarray:
    """Load the configured track centerline as an (N, 2) world-coordinate array."""
    configured = manifest.get("scenario_config", {}).get("track", {}).get("csv")
    if not configured:
        raise ValueError(f"{run_dir}: manifest does not contain scenario_config.track.csv")
    path = resolve_saved_path(configured)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"track centerline not found: {path}")
    centerline = np.genfromtxt(path, delimiter=",", names=True)
    if centerline.dtype.names is None or not {"x", "y"} <= set(centerline.dtype.names):
        raise ValueError(f"track centerline needs x and y columns: {path}")
    return np.column_stack((centerline["x"], centerline["y"]))


def track_ribbon(centerline: np.ndarray, width: float) -> np.ndarray:
    """Build closed quadrilateral segments around the track centerline."""
    previous = np.roll(centerline, 1, axis=0)
    following = np.roll(centerline, -1, axis=0)
    tangent = following - previous
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = centerline + normal * (width * 0.5)
    right = centerline - normal * (width * 0.5)
    return np.stack(
        (left, np.roll(left, -1, axis=0), np.roll(right, -1, axis=0), right),
        axis=1,
    )


def channel_indices(data: dict[str, np.ndarray]) -> dict[str, int]:
    channels = [str(name) for name in data["state_channels"]]
    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(f"archive is missing state channel(s): {', '.join(missing)}")
    return {name: channels.index(name) for name in REQUIRED_CHANNELS}


def color_for(model: str, index: int) -> str:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return MODEL_COLORS.get(model, cycle[index % len(cycle)])


def write_rmse_csv(path: Path, rows: list[dict]) -> None:
    """Write RMSE values calculated over the displayed rollout horizon."""
    fields = (
        "environment",
        "direction",
        "expert",
        "predictor",
        "horizon_s",
        "transitions",
        "position_rmse_m",
        "hitch_angle_rmse_deg",
        "velocity_rmse_mps",
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_expert_panels(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    run_dir: Path,
    expert: str,
    predictions: dict[str, dict[str, np.ndarray]],
    manifest: dict,
    terrain: tuple[np.ndarray, np.ndarray] | None,
    track: np.ndarray,
    log_errors: bool,
    max_seconds: float | None,
    full_rollout: bool,
    show_titles: bool = True,
    align_spatial: bool = False,
) -> tuple[list, list, list[dict], list]:
    """Draw spatial, hitch-angle error, and velocity error panels."""
    first = next(iter(predictions.values()))
    indices = channel_indices(first)
    actual = first["expert_states"]
    dt = float(manifest.get("dt", 0.05))

    if max_seconds is not None:
        sample_count = min(len(actual), int(max_seconds / dt) + 1)
    elif full_rollout:
        sample_count = len(actual)
    else:
        sample_count = min(len(actual), int(DEFAULT_PLOT_SECONDS / dt) + 1)
    actual = actual[:sample_count]
    time_s = np.arange(sample_count) * dt

    x = indices["x"]
    y = indices["y"]
    yaw_tractor = indices["yaw_tractor"]
    yaw_trailer = indices["yaw_trailer"]
    actual_hitch = wrap_angle(actual[:, yaw_tractor] - actual[:, yaw_trailer])
    ax_xy, ax_hitch_error, ax_velocity_error = axes
    environment = str(manifest.get("environment_label", run_dir.parent.name))
    vehicle = manifest.get('scenario_config', {}).get('vehicle', {})
    pose_indices = [x, y, yaw_tractor, yaw_trailer]
    endpoint_points = [endpoint_rig_vertices(actual[-1, pose_indices], vehicle)]
    rmse_rows: list[dict] = []

    spatial_states = [actual]
    ax_xy.plot(
        actual[:, x], actual[:, y], color="black", lw=5.0 if align_spatial else 4.0,
        label="Actual", zorder=7
    )
    ax_xy.scatter(
        actual[0, x], actual[0, y], color="black", s=28, marker="o", zorder=8
    )
    draw_endpoint_rig(ax_xy, actual[-1, pose_indices], vehicle, 'black')

    for model_index, (predictor, data) in enumerate(sorted(predictions.items())):
        archive_actual = data["expert_states"][:sample_count]
        if archive_actual.shape != actual.shape or not np.allclose(
            archive_actual, actual, equal_nan=True
        ):
            raise ValueError(
                f"expert states disagree between prediction archives for {expert}"
            )

        predicted = data["predicted_states"][:sample_count]
        endpoint_points.append(endpoint_rig_vertices(predicted[-1, pose_indices], vehicle))
        spatial_states.append(predicted)
        color = color_for(predictor, model_index)
        label = MODEL_LABELS.get(predictor, predictor)
        predicted_hitch = wrap_angle(
            predicted[:, yaw_tractor] - predicted[:, yaw_trailer]
        )

        hitch_error_deg = np.degrees(
            np.abs(wrap_angle(predicted_hitch - actual_hitch))
        )
        position_error = np.linalg.norm(
            predicted[:, [x, y]] - actual[:, [x, y]], axis=1
        )
        velocity_error = np.linalg.norm(
            predicted[:, [indices["vx"], indices["vy"]]]
            - actual[:, [indices["vx"], indices["vy"]]], axis=1
        )
        velocity_valid = np.isfinite(velocity_error)
        velocity_valid[0] = False
        velocity_rmse = (
            float(np.sqrt(np.mean(velocity_error[velocity_valid] ** 2)))
            if velocity_valid.any() else np.nan
        )
        valid = np.isfinite(position_error) & np.isfinite(hitch_error_deg)
        valid[0] = False
        if not valid.any():
            position_rmse = np.nan
            hitch_rmse = np.nan
        else:
            position_rmse = float(np.sqrt(np.mean(position_error[valid] ** 2)))
            hitch_rmse = float(np.sqrt(np.mean(hitch_error_deg[valid] ** 2)))
        rmse_rows.append(
            {
                "environment": environment,
                "direction": run_dir.name,
                "expert": expert,
                "predictor": predictor,
                "horizon_s": float(time_s[-1]),
                "transitions": sample_count - 1,
                "position_rmse_m": position_rmse,
                "hitch_angle_rmse_deg": hitch_rmse,
                "velocity_rmse_mps": velocity_rmse,
            }
        )

        ax_xy.plot(
            predicted[:, x],
            predicted[:, y],
            color=color,
            lw=4.0 if align_spatial else 3.0,
            label=label,
            zorder=6,
        )
        draw_endpoint_rig(ax_xy, predicted[-1, pose_indices], vehicle, color)
        ax_hitch_error.plot(
            time_s, hitch_error_deg, color=color, lw=3.0, label=predictor
        )

        ax_velocity_error.plot(
            time_s, velocity_error, color=color, lw=3.0, label=predictor
        )

    if show_titles:
        ax_xy.set_title("Spatial rollout")
        ax_hitch_error.set_title("Hitch-angle divergence")
        ax_velocity_error.set_title("Velocity divergence")
    ax_xy.set_aspect("equal", adjustable="box")
    if not align_spatial:
        ax_xy.set_box_aspect(1)
    ax_xy.set_anchor("C")
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])
    track_width = float(
        manifest.get("scenario_config", {}).get("track", {}).get("width", 15.0)
    )
    ribbon = track_ribbon(track, track_width)
    xlim, ylim = spatial_limits(spatial_states, x, y, np.concatenate(endpoint_points))
    spatial_transform = None
    if align_spatial:
        # Align the long direction of the recorded road traversal horizontally.
        # Apply the same rigid transform to every path and background artist.
        actual_xy = actual[:, [x, y]]
        _, _, directions = np.linalg.svd(actual_xy - actual_xy.mean(axis=0), full_matrices=False)
        heading = directions[0]
        if np.dot(heading, actual_xy[-1] - actual_xy[0]) < 0:
            heading = -heading
        spatial_transform = Affine2D().translate(*(-actual_xy[0])).rotate(
            -np.arctan2(heading[1], heading[0]))
        # Include both road edges along the recorded segment when framing the
        # trajectories, rather than cropping to the vehicle paths alone.
        nearest = np.argmin(np.sum((actual_xy[:, None, :] - track[None, :, :]) ** 2, axis=2), axis=1)
        road_segments = np.unique(np.concatenate((nearest, (nearest - 1) % len(track))))
        positions = spatial_transform.transform(np.concatenate(
            [s[:, [x, y]] for s in spatial_states] +
            [ribbon[road_segments].reshape(-1, 2), *endpoint_points]))
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        padding = np.maximum(2.0, .05 * (hi - lo))
        lo, hi = lo - padding, hi + padding
        # Keep the view landscape without changing distances or cropping paths.
        extra = max(0.0, 2.4 * (hi[1] - lo[1]) - (hi[0] - lo[0])) / 2
        lo[0] -= extra
        hi[0] += extra
        view_xlim, view_ylim = (lo[0], hi[0]), (lo[1], hi[1])
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        world_corners = spatial_transform.inverted().transform(corners)
        xlim = (world_corners[:, 0].min(), world_corners[:, 0].max())
        ylim = (world_corners[:, 1].min(), world_corners[:, 1].max())
    rendered_background = add_rendered_background(ax_xy, environment, run_dir.name)
    environment_handles = []
    if not rendered_background:
        environment_handles = add_environment_background(
            ax_xy, environment, terrain, xlim, ylim
        )
        ax_xy.add_collection(
            PolyCollection(
                ribbon,
                facecolors="#78838f",
                edgecolors="none",
                alpha=0.35 if align_spatial else 0.18,
                zorder=4,
            )
        )
    if spatial_transform is not None:
        transform = spatial_transform + ax_xy.transData
        for artist in (*ax_xy.lines, *ax_xy.images, *ax_xy.collections, *ax_xy.patches):
            if isinstance(artist, PathCollection):
                artist.set_offset_transform(transform)
            else:
                artist.set_transform(transform)
        ax_xy.set_xlim(*view_xlim)
        ax_xy.set_ylim(*view_ylim)
    else:
        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
    ax_hitch_error.set_ylabel("Hitch Angle Error (deg)")
    ax_velocity_error.set_ylabel("Velocity Error (m/s)")
    ax_velocity_error.set_xlabel("Time (s)")
    ax_velocity_error.grid(alpha=0.28)
    ax_hitch_error.set_xlabel("Time (s)")
    ax_hitch_error.grid(alpha=0.28)
    ax_xy.grid(False)
    for spine in ax_xy.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.4")
    if log_errors:
        ax_hitch_error.set_yscale("symlog", linthresh=1e-3)
        ax_velocity_error.set_yscale("symlog", linthresh=1e-3)

    handles, labels = ax_xy.get_legend_handles_labels()
    return handles, labels, rmse_rows, environment_handles


def plot_expert(
    run_dir: Path,
    expert: str,
    predictions: dict[str, dict[str, np.ndarray]],
    manifest: dict,
    output_dir: Path,
    terrain: tuple[np.ndarray, np.ndarray] | None,
    track: np.ndarray,
    log_errors: bool,
    max_seconds: float | None,
    full_rollout: bool,
) -> Path:
    """Plot spatial, hitch-angle, and velocity divergence."""
    fig = plt.figure(figsize=(22, 14))
    axes = tuple(fig.add_axes((.1, .3, .8, .5)) for _ in range(3))
    handles, labels, rmse_rows, environment_handles = draw_expert_panels(
        tuple(axes),
        run_dir,
        expert,
        predictions,
        manifest,
        terrain,
        track,
        log_errors,
        max_seconds,
        full_rollout,
        align_spatial=True,
    )
    # Size the first row from its metric bounds so equal aspect fills the entire
    # available width. Fixed GridSpec heights would shrink it into a small box.
    figure_width = 22.0
    left, right, column_gap = 1.3, .35, 1.8
    panel_width = figure_width - left - right
    xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
    spatial_height = panel_width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    bottom = 2.05 if not environment_handles else 2.75
    error_height, row_gap, top = 3.6, .8, .75
    spatial_bottom = bottom + error_height + row_gap
    figure_height = spatial_bottom + spatial_height + top
    fig.set_size_inches(figure_width, figure_height)
    axes[0].set_position((left / figure_width, spatial_bottom / figure_height,
                          panel_width / figure_width, spatial_height / figure_height))
    error_width = (panel_width - column_gap) / 2
    for column, ax in enumerate(axes[1:]):
        ax.set_position(((left + column * (error_width + column_gap)) / figure_width,
                         bottom / figure_height, error_width / figure_width,
                         error_height / figure_height))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, (.2 if not environment_handles else .9) / figure_height),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    if environment_handles:
        fig.legend(environment_handles, [h.get_label() for h in environment_handles],
                   loc="lower center", bbox_to_anchor=(0.5, .2 / figure_height),
                   ncol=len(environment_handles), frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"divergence_expert-{expert}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix('.pdf'), bbox_inches="tight")
    write_rmse_csv(output_dir / f"rmse_expert-{expert}.csv", rmse_rows)
    for row in rmse_rows:
        print(
            f"RMSE {row['environment']}/{row['direction']} "
            f"expert={expert} predictor={row['predictor']}: "
            f"position={row['position_rmse_m']:.3f} m, "
            f"hitch={row['hitch_angle_rmse_deg']:.3f} deg, "
            f"velocity={row['velocity_rmse_mps']:.3f} m/s"
        )
    return output


def plot_combined_expert(
    rows: list[
        tuple[
            str,
            Path,
            dict[str, dict[str, np.ndarray]],
            dict,
            tuple[np.ndarray, np.ndarray] | None,
            np.ndarray,
        ]
    ],
    expert: str,
    output_dir: Path,
    log_errors: bool,
    max_seconds: float | None,
    full_rollout: bool,
    figure_title: str | None = None,
) -> Path:
    """Plot four environments as columns of a three-row comparison figure."""
    fig, axes = plt.subplots(
        3,
        len(rows),
        figsize=(5.1 * len(rows), 19),
        squeeze=False,
        gridspec_kw={"height_ratios": (1.08, 1.0, 1.0)},
    )
    if figure_title:
        fig.suptitle(figure_title, fontsize=12 * FONT_SCALE, y=0.985)
    legend_handles: list = []
    legend_labels: list = []
    environment_legend: dict[str, object] = {}
    combined_rmse: list[dict] = []
    for column in range(1, len(rows)):
        axes[1, column].sharey(axes[1, 0])
        axes[2, column].sharey(axes[2, 0])
    for column, (row_label, run_dir, predictions, manifest, terrain, track) in enumerate(rows):
        handles, labels, rmse_rows, environment_handles = draw_expert_panels(
            (axes[0, column], axes[1, column], axes[2, column]),
            run_dir,
            expert,
            predictions,
            manifest,
            terrain,
            track,
            log_errors,
            max_seconds,
            full_rollout,
            show_titles=False,
        )
        axes[0, column].set_title(
            row_label, fontsize=12 * FONT_SCALE, fontweight="normal"
        )
        for row in rmse_rows:
            row["environment"] = row_label
        combined_rmse.extend(rmse_rows)
        for handle in environment_handles:
            environment_legend.setdefault(handle.get_label(), handle)
        if column:
            axes[1, column].set_ylabel("")
            axes[1, column].tick_params(axis="y", left=False, labelleft=False)
            axes[2, column].tick_params(axis="y", left=False, labelleft=False)
        if not legend_handles:
            legend_handles, legend_labels = handles, labels

    for column in range(len(rows)):
        axes[1, column].set_ylabel("")
        axes[2, column].set_ylabel("")
        axes[1, column].set_xlabel("")
    legend_handles.extend(environment_legend.values())
    legend_labels.extend(environment_legend.keys())

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(legend_labels)),
        frameon=False,
        fontsize=9 * FONT_SCALE,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.18,
        top=0.92,
        wspace=0.12,
        hspace=0.18,
    )

    # Let Matplotlib place labels beside the first column's tick labels.
    for row_index, label in enumerate(
        ("Spatial Rollout", "Hitch Angle Error\n(deg)", "Velocity Error\n(m/s)")
    ):
        axes[row_index, 0].set_ylabel(
            label, fontsize=12 * FONT_SCALE, labelpad=8,
        )
    fig.align_ylabels(axes[:, 0])

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"divergence_expert-{expert}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    write_rmse_csv(output_dir / f"rmse_expert-{expert}.csv", combined_rmse)
    for row in combined_rmse:
        print(
            f"RMSE {row['environment']}/{row['direction']} "
            f"expert={expert} predictor={row['predictor']}: "
            f"position={row['position_rmse_m']:.3f} m, "
            f"hitch={row['hitch_angle_rmse_deg']:.3f} deg, "
            f"velocity={row['velocity_rmse_mps']:.3f} m/s"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="direction or parent directories; default: ./*_div",
    )
    parser.add_argument(
        "--expert",
        action="append",
        help="expert controller to plot; repeat for multiple (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output root; default: plots inside each run, or combined_div in combined mode",
    )
    parser.add_argument(
        "--combine-environments",
        action="store_true",
        help="combine asphalt, snow, slope, and both4 into four-column figures",
    )
    parser.add_argument(
        "--levels-dir",
        type=Path,
        default=DEFAULT_LEVELS_DIR,
        help=f"BeamNG levels directory containing terrain files (default: {DEFAULT_LEVELS_DIR})",
    )
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--max-seconds",
        type=float,
        help="override the manifest horizon and plot the first N seconds",
    )
    window_group.add_argument(
        "--full-rollout",
        action="store_true",
        help="plot every saved step instead of the default five seconds",
    )
    parser.add_argument(
        "--log-errors",
        action="store_true",
        help="use a symmetric-log scale for the error panels",
    )
    parser.add_argument("--show", action="store_true", help="open the figures interactively")
    return parser.parse_args()


def terrain_for_environment(
    environment: str,
    levels_dir: Path,
    cache: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load each environment terrain at most once."""
    if environment not in cache:
        relative_terrain = TERRAIN_PATHS.get(environment)
        if relative_terrain is None:
            cache[environment] = None
        else:
            terrain_path = levels_dir.expanduser() / relative_terrain
            if not terrain_path.is_file():
                raise FileNotFoundError(
                    f"terrain for {environment} not found: {terrain_path}"
                )
            cache[environment] = load_terrain(terrain_path)
    return cache[environment]


def plot_combined_runs(
    args: argparse.Namespace,
    run_dirs: list[Path],
    terrain_cache: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> list[Path]:
    """Create one four-environment figure per direction and expert."""
    requested_roots = [name for name, _ in COMBINED_ENVIRONMENTS]
    requested_labels = dict(COMBINED_ENVIRONMENTS)
    indexed = {(run_dir.parent.name, run_dir.name): run_dir for run_dir in run_dirs}
    directions = sorted(
        {
            direction
            for root, direction in indexed
            if root in requested_roots
        }
    )
    if not directions:
        raise ValueError("no asphalt, snow, slope, or both4 runs were found")

    outputs: list[Path] = []
    for direction in directions:
        missing_roots = [
            root for root in requested_roots if (root, direction) not in indexed
        ]
        if missing_roots:
            raise ValueError(
                f"{direction}: missing combined environment run(s): "
                f"{', '.join(missing_roots)}"
            )

        loaded_rows = []
        for root in requested_roots:
            run_dir = indexed[root, direction]
            grouped = load_predictions(run_dir)
            manifest = read_manifest(run_dir)
            environment = str(
                manifest.get("environment_label", run_dir.parent.name)
            )
            loaded_rows.append(
                (
                    requested_labels[root],
                    run_dir,
                    grouped,
                    manifest,
                    terrain_for_environment(
                        environment, args.levels_dir, terrain_cache
                    ),
                    load_track(manifest, run_dir),
                )
            )

        available_experts = set(loaded_rows[0][2])
        for row in loaded_rows[1:]:
            available_experts.intersection_update(row[2])
        selected = sorted(available_experts) if not args.expert else args.expert
        missing_experts = [
            expert for expert in selected if expert not in available_experts
        ]
        if missing_experts:
            raise ValueError(
                f"{direction}: expert(s) unavailable in every environment: "
                f"{', '.join(missing_experts)}"
            )

        output_dir = (
            Path("combined_div") / direction / "plots"
            if args.output is None
            else args.output / direction
        )
        for expert in selected:
            rows = [
                (label, run_dir, grouped[expert], manifest, terrain, track)
                for label, run_dir, grouped, manifest, terrain, track in loaded_rows
            ]
            output = plot_combined_expert(
                rows,
                expert,
                output_dir,
                args.log_errors,
                args.max_seconds,
                args.full_rollout,
            )
            outputs.append(output)
            print(f"saved {output}")
            if not args.show:
                plt.close(plt.gcf())
    return outputs


def main() -> None:
    args = parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")

    outputs: list[Path] = []
    terrain_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    discovery_paths = args.paths
    if args.combine_environments and not discovery_paths:
        discovery_paths = [Path(root) for root, _ in COMBINED_ENVIRONMENTS]
    run_dirs = find_run_directories(discovery_paths)
    if args.combine_environments:
        outputs = plot_combined_runs(args, run_dirs, terrain_cache)
    else:
        for run_dir in run_dirs:
            grouped = load_predictions(run_dir)
            manifest = read_manifest(run_dir)
            track = load_track(manifest, run_dir)
            environment = str(manifest.get("environment_label", run_dir.parent.name))
            terrain = terrain_for_environment(
                environment, args.levels_dir, terrain_cache
            )
            selected = sorted(grouped) if not args.expert else args.expert
            missing = [expert for expert in selected if expert not in grouped]
            if missing:
                available = ", ".join(sorted(grouped))
                raise ValueError(
                    f"{run_dir}: unknown expert(s) {', '.join(missing)}; "
                    f"available: {available}"
                )

            if args.output is None:
                output_dir = run_dir / "plots"
            else:
                output_dir = args.output / str(environment) / run_dir.name

            for expert in selected:
                output = plot_expert(
                    run_dir,
                    expert,
                    grouped[expert],
                    manifest,
                    output_dir,
                    terrain,
                    track,
                    args.log_errors,
                    args.max_seconds,
                    args.full_rollout,
                )
                outputs.append(output)
                print(f"saved {output}")
                if not args.show:
                    plt.close(plt.gcf())

    if args.show:
        plt.show()
    else:
        plt.close("all")
    print(f"generated {len(outputs)} figure(s)")


if __name__ == "__main__":
    main()
