"""Reinitialize open-loop divergence tests at shared physical waypoints.

This reuses the saved model-controlled BeamNG trajectory from each environment.
For each direction, the closest saved state to a common XY waypoint becomes the
new initial condition.  The learned model receives the four real history rows
ending at that state, and every predictor replays the following saved controls.

No BeamNG session is started and the source archives are never modified.

Example:
    python -m experiments.exp_008_beamng.rerun_div_from_waypoints
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

# The development environment may contain CUDA 12 and CUDA 13 PJRT plugins
# even when no NVIDIA device is exposed.  Avoid loading those conflicting
# plugins for this CPU-only replay; retain normal discovery on GPU machines.
CPU_ONLY = (
    os.environ.get("WAYPOINT_REPLAY_CPU") == "1"
    or not Path("/dev/nvidiactl").exists()
)
if CPU_ONLY:
    from jax._src import xla_bridge

    xla_bridge.discover_pjrt_plugins = lambda: None

import experiments.exp_008_beamng.run_model_for_div as div_runner
from experiments.exp_008_beamng.run_model_for_div import (
    CONTROL_CHANNELS,
    DEFAULT_MODELS,
    DT,
    STATE_CHANNELS,
    build_runners,
    compute_metrics,
    learned_row,
    save_expert,
    save_prediction,
    write_summary,
)
from src.simulation.beamng_trailer_env import bng_pickup_trailer_cfg
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    SimulationConfig,
    TrackConfig,
)
from experiments.exp_008_beamng.experiment_paths import DATA_ROOT, resolve_saved_path


SOURCE_ENVIRONMENTS = ("asphalt_div", "snow_div", "slope_div", "both4_div")
WAYPOINTS = {
    "forward": np.array([477.642, -62.550], dtype=np.float64),
    "backward": np.array([-341.269166873739, 463.82229449686844], dtype=np.float64),
}
BASELINE_EXPERT = "learned_h4"
LEARNED_HISTORY = 4


def restore_model_from_local_store(model, checkpoint: Path) -> None:
    """Restore a GPU-sharded Orbax checkpoint directly onto the CPU.

    Orbax attempts to reconstruct the checkpoint's saved ``cuda:0`` sharding
    before applying the CPU target tree in environments with no CUDA device.
    Reading the local OCDBT/Zarr arrays directly avoids that backend lookup.
    """
    checkpoint = checkpoint.resolve()
    for layer_index, layer in enumerate(model.model.layers):
        if not hasattr(layer, "kernel"):
            continue
        for field in ("bias", "kernel"):
            key = f"model.layers.{layer_index}.{field}.value"
            spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "ocdbt",
                    "base": f"file://{checkpoint}",
                    "path": key,
                },
            }
            value = ts.open(spec, open=True).result().read().result()
            getattr(layer, field).value = jnp.asarray(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DATA_ROOT,
        help="directory containing asphalt_div, snow_div, slope_div, and both4_div",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("turn_snow_div"),
        help="new output root (must not already exist)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of open-loop transitions after the waypoint (default: 100 = 5 s)",
    )
    parser.add_argument(
        "--max-waypoint-distance",
        type=float,
        default=3.0,
        help="reject a source trajectory farther than this many metres from its waypoint",
    )
    return parser.parse_args()


def load_source_expert(run_dir: Path) -> dict:
    archive = run_dir / f"expert_{BASELINE_EXPERT}.npz"
    if not archive.is_file():
        raise FileNotFoundError(f"missing baseline expert archive: {archive}")
    with np.load(archive) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def history_at_state(
    states: np.ndarray,
    controls: np.ndarray,
    state_index: int,
    history: int,
) -> np.ndarray:
    """Return real learned-model rows ending at ``states[state_index]``."""
    first_state = state_index - history + 1
    if first_state < 1:
        raise ValueError(
            f"state index {state_index} does not have {history} recorded history rows"
        )
    return np.stack(
        [
            learned_row(states[index], controls[index - 1])
            for index in range(first_state, state_index + 1)
        ]
    )


def select_segment(
    source: dict,
    waypoint: np.ndarray,
    requested_steps: int,
    max_waypoint_distance: float,
) -> tuple[dict, dict]:
    states = np.asarray(source["states"], dtype=np.float64)
    controls = np.asarray(source["controls"], dtype=np.float64)
    channels = [str(name) for name in source["state_channels"]]
    nearest = int(np.argmin(np.linalg.norm(states[:, :2] - waypoint, axis=1)))
    distance = float(np.linalg.norm(states[nearest, :2] - waypoint))
    if distance > max_waypoint_distance:
        raise ValueError(
            f"nearest saved state is {distance:.2f} m from waypoint "
            f"{tuple(waypoint)} (limit {max_waypoint_distance:.2f} m)"
        )
    steps = min(requested_steps, len(controls) - nearest)
    if steps < requested_steps:
        raise ValueError(
            f"waypoint at state {nearest} leaves only {steps} controls; "
            f"{requested_steps} requested"
        )

    segment_states = states[nearest : nearest + steps + 1]
    segment_controls = controls[nearest : nearest + steps]
    initial_history = history_at_state(
        states, controls, nearest, LEARNED_HISTORY
    )
    speed_kph = float(
        np.hypot(
            states[nearest, channels.index("vx")],
            states[nearest, channels.index("vy")],
        )
        * 3.6
    )

    solve_ms = np.asarray(source.get("solve_ms", np.empty(0)))
    solve_segment = solve_ms[nearest : nearest + steps]
    terminations = np.asarray(
        source.get("termination_steps", np.empty(0, dtype=np.int32)),
        dtype=np.int32,
    )
    terminations = terminations[
        (terminations > nearest) & (terminations <= nearest + steps)
    ] - nearest

    expert = {
        "states": segment_states,
        "controls": segment_controls,
        "initial_history": initial_history,
        "solve_ms": solve_segment,
        "status": str(np.asarray(source.get("status", "completed")).item()),
        "termination_steps": terminations,
    }
    selection = {
        "query_xy": waypoint.tolist(),
        "nearest_state_index": nearest,
        "nearest_state_xy": segment_states[0, :2].tolist(),
        "nearest_distance_m": distance,
        "initial_speed_kph": speed_kph,
        "history_state_indices": list(
            range(nearest - LEARNED_HISTORY + 1, nearest + 1)
        ),
        "control_indices": [nearest, nearest + steps - 1],
        "state_indices": [nearest, nearest + steps],
        "steps": steps,
        "duration_s": steps * DT,
    }
    return expert, selection


def scenario_from_manifest(manifest: dict) -> BeamNGTrailerEnvConfig:
    track = manifest["scenario_config"]["track"]
    return BeamNGTrailerEnvConfig(
        ".",
        TrackConfig(
            csv=str(resolve_saved_path(track["csv"])),
            width=float(track["width"]),
            friction_csv=(str(resolve_saved_path(track["friction_csv"]))
                          if track["friction_csv"] else None),
            closed=bool(track["closed"]),
            mu=float(track["mu"]),
        ),
        bng_pickup_trailer_cfg,
        SimulationConfig(dt=DT),
    )


def learned_models_from_manifest(manifest: dict) -> dict:
    config = manifest["learned"][BASELINE_EXPERT]
    model = (
        int(config["history"]),
        resolve_saved_path(config["stats"]),
        resolve_saved_path(config["checkpoint"]),
    )
    return {BASELINE_EXPERT: model, "learned": model}


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.max_waypoint_distance <= 0:
        raise ValueError("--max-waypoint-distance must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"output already exists: {args.output}; choose a new --output path"
        )

    source_root = args.source_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True)
    if CPU_ONLY:
        div_runner.restore_model = restore_model_from_local_store

    for direction, waypoint in WAYPOINTS.items():
        first_run = source_root / SOURCE_ENVIRONMENTS[0] / direction
        first_manifest = json.loads((first_run / "manifest.json").read_text())
        scenario = scenario_from_manifest(first_manifest)
        print(f"restoring predictors for {direction}...", flush=True)
        runners, _, _ = build_runners(
            DEFAULT_MODELS,
            scenario,
            float(first_manifest["target_kph"]),
            learned_models_from_manifest(first_manifest),
        )

        for environment_root in SOURCE_ENVIRONMENTS:
            source_run = source_root / environment_root / direction
            source_manifest = json.loads((source_run / "manifest.json").read_text())
            source_expert = load_source_expert(source_run)
            expert, selection = select_segment(
                source_expert,
                waypoint,
                args.steps,
                args.max_waypoint_distance,
            )

            output = output_root / environment_root / direction
            output.mkdir(parents=True)
            manifest = {
                **source_manifest,
                "scenario_config": asdict(scenario),
                "models": list(DEFAULT_MODELS),
                "steps": selection["steps"],
                "state_channels": STATE_CHANNELS,
                "control_channels": CONTROL_CHANNELS,
                "baseline_expert": BASELINE_EXPERT,
                "source_run": str(source_run.resolve()),
                "waypoint_selection": selection,
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            save_expert(output, BASELINE_EXPERT, expert)

            summary = []
            print(
                f"{environment_root}/{direction}: state "
                f"{selection['nearest_state_index']}, "
                f"distance={selection['nearest_distance_m']:.2f} m, "
                f"speed={selection['initial_speed_kph']:.1f} km/h",
                flush=True,
            )
            for predictor_name, runner in runners.items():
                predicted = runner.open_loop(
                    expert["states"],
                    expert["controls"],
                    expert["initial_history"],
                    DT,
                )
                metrics = compute_metrics(expert["states"], predicted)
                summary.append(
                    {
                        "expert": BASELINE_EXPERT,
                        "predictor": predictor_name,
                        "steps": len(expert["controls"]),
                        **metrics,
                    }
                )
                save_prediction(
                    output,
                    BASELINE_EXPERT,
                    predictor_name,
                    expert,
                    predicted,
                )
                print(
                    f"  {predictor_name:<13} "
                    f"position RMSE={metrics.get('position_rmse_m', np.nan):8.3f} m  "
                    f"hitch RMSE={metrics.get('hitch_rmse_deg', np.nan):8.3f} deg",
                    flush=True,
                )
            write_summary(output, summary)

    print(f"saved waypoint divergence data -> {output_root}")


if __name__ == "__main__":
    main()
