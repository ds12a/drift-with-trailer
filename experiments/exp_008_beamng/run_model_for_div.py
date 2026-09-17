"""Generate MPPI expert trajectories and cross-model open-loop predictions.

By default, run 1000 forward (+80 km/h) and 1000 backward (-30 km/h)
steps for H=1, H=4, measured-friction Fiala, and fixed-mu=1 Fiala.
Each configured dynamics model first controls BeamNG in closed loop.  Every
model is then replayed open loop from every expert trajectory's initial state
using the recorded BeamNG command sequence.  The resulting expert and fake
trajectories are saved as compressed NumPy archives.

Examples:
    python -m experiments.exp_008_beamng.run_model_for_div
    python -m experiments.exp_008_beamng.run_model_for_div --steps 1000 --samples 1000
    python -m experiments.exp_008_beamng.run_model_for_div --target-kph -30
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import csv
from dataclasses import asdict, astuple, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.dynamics.trailer.beamng_dynamics import gen_util_funs as learned_util
from src.dynamics.trailer.trailer_bicycle_fiala_surface import (
    gen_util_funs as surface_fiala_util,
)
from src.learning.models import beamng_model_spec
from src.learning.models.beamng_trailer_spec import IN_COLS, fiala_dyn
from src.learning.models.trailer_nn import TrailerModel
from src.simulation.beamng_trailer_env import (
    BeamNGTrailerEnv,
    bng_pickup_trailer_cfg,
)
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    SimulationConfig,
    TrackConfig,
)


logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False


DT = 0.05
ROW_WIDTH = 13
STATE_CHANNELS = (
    "x",
    "y",
    "yaw_tractor",
    "yaw_trailer",
    "vx",
    "vy",
    "yaw_rate_tractor",
    "yaw_rate_trailer",
    "steer_realized",
    "accel_realized",
    "mu_tractor",
    "mu_trailer",
    "arc_length",
)
CONTROL_CHANNELS = ("steer_command", "accel_command")

I_YAW_TRACTOR = 2
I_YAW_TRAILER = 3
I_MU_TRACTOR = 10
I_MU_TRAILER = 11
I_ARC = 12

DEFAULT_MODELS = ("learned_h1", "learned_h4", "fiala_surface", "fiala")
STATS_PATH = Path("experiments/exp_008_beamng/data_proc_test10-new_stats.json")
CHECKPOINT_PATH = (
    Path.cwd() / "src/learning/models/trained/beamng-l4-128-test10-new_rollout10"
)
LEARNED_MODELS = {
    "learned_h1": (1, Path("experiments/exp_008_beamng/data_proc_h1_stats.json"),
                   Path("src/learning/models/trained/beamng-l4-h1")),
    "learned_h4": (4, STATS_PATH, CHECKPOINT_PATH),
    "learned": (4, STATS_PATH, CHECKPOINT_PATH),  # compatibility alias
}
OUT_ROOT = Path("experiments/exp_008_beamng/model_divergence_out")

FWD_WEIGHTS = {
    "p_weight": 2e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 1e2,
    "reverse": False,
}
REV_WEIGHTS = {
    "p_weight": 5e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 2e2,
    "reverse": False,
}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def learned_row(common_state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Convert the common saved state to beamng_dynamics' 13-wide row."""
    return np.concatenate(
        [common_state[:10], np.asarray(control, dtype=float), common_state[I_ARC : I_ARC + 1]]
    )


@dataclass
class DynamicsRunner:
    name: str
    dynamics: Callable
    cost: Callable
    bound: Callable
    history: int | None
    state_dim: int
    steering_sign_to_beamng: float

    def planner_state(
        self, common_state: np.ndarray, history_rows: list[np.ndarray]
    ) -> jax.Array:
        if self.history is not None:
            if len(history_rows) < self.history:
                raise ValueError(
                    f"{self.name} needs {self.history} history rows, got {len(history_rows)}"
                )
            return jnp.asarray(np.stack(history_rows[-self.history :]).reshape(-1))
        return jnp.asarray(self._native_state(common_state))

    def control_to_beamng(self, control: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.steering_sign_to_beamng * control[0], control[1]], dtype=float
        )

    def control_from_beamng(self, control: np.ndarray) -> jax.Array:
        return jnp.asarray(
            [self.steering_sign_to_beamng * control[0], control[1]]
        )

    def _native_state(self, common_state: np.ndarray) -> np.ndarray:
        if self.name in ("fiala_surface", "fiala"):
            return np.concatenate(
                [
                    common_state[:8],
                    (common_state[I_MU_TRACTOR : I_MU_TRAILER + 1]
                     if self.name == "fiala_surface" else np.ones(2)),
                    common_state[I_ARC : I_ARC + 1],
                ]
            )
        raise ValueError(f"{self.name} requires history rows")

    def open_loop(
        self,
        expert_states: np.ndarray,
        controls: np.ndarray,
        initial_history: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Replay controls without state correction after the initial condition."""
        predicted = np.empty_like(expert_states, dtype=np.float64)
        predicted[0] = expert_states[0]

        if self.history is not None:
            rows = jnp.asarray(initial_history[-self.history :])
            for i, control in enumerate(controls):
                model_u = self.control_from_beamng(control)
                flat = rows.reshape(-1)
                dx = self.dynamics(flat, model_u)
                next_row = rows[-1] + dx * dt
                rows = jnp.concatenate([rows[1:], next_row[None, :]], axis=0)

                raw = np.asarray(next_row, dtype=float)
                predicted[i + 1, :10] = raw[:10]
                predicted[i + 1, I_MU_TRACTOR : I_MU_TRAILER + 1] = (
                    expert_states[i + 1, I_MU_TRACTOR : I_MU_TRAILER + 1]
                )
                predicted[i + 1, I_ARC] = raw[-1]
        else:
            native = jnp.asarray(self._native_state(expert_states[0]))
            for i, control in enumerate(controls):
                # Friction is an observed exogenous input. Replay its measured
                # schedule while leaving all physical state open loop.
                native = native.at[8:10].set(
                    jnp.asarray(expert_states[i, I_MU_TRACTOR : I_MU_TRAILER + 1])
                    if self.name == "fiala_surface" else jnp.ones(2)
                )
                model_u = self.control_from_beamng(control)
                native = native + self.dynamics(native, model_u) * dt

                raw = np.asarray(native, dtype=float)
                predicted[i + 1, :8] = raw[:8]
                predicted[i + 1, 8:10] = control
                predicted[i + 1, I_MU_TRACTOR : I_MU_TRAILER + 1] = (
                    expert_states[i + 1, I_MU_TRACTOR : I_MU_TRAILER + 1]
                )
                predicted[i + 1, I_ARC] = raw[-1]

        return predicted


def restore_model(model: TrailerModel, checkpoint: Path) -> None:
    """Restore checkpoints saved either as nnx.State or a pure dict."""
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    try:
        nnx.update(model, checkpointer.restore(checkpoint, state))
        return
    except ValueError as exc:
        if "structures do not match" not in str(exc):
            raise
    pure = state.to_pure_dict()
    state.replace_by_pure_dict(checkpointer.restore(checkpoint, pure))
    nnx.update(model, state)


def build_runners(
    names: tuple[str, ...],
    scenario: BeamNGTrailerEnvConfig,
    target_kph: float,
    learned_models: dict | None = None,
) -> tuple[dict[str, DynamicsRunner], int, dict]:
    learned_models = LEARNED_MODELS if learned_models is None else learned_models
    weights = dict(FWD_WEIGHTS if target_kph > 0 else REV_WEIGHTS)
    weights["v_target"] = target_kph / 3.6
    runners: dict[str, DynamicsRunner] = {}
    learned_history = 1
    learned_config = {}

    for name in names:
        if name not in learned_models:
            continue
        expected_history, stats_path, checkpoint = learned_models[name]
        stats = json.loads(stats_path.read_text())
        input_width = len(stats["x_mean"])
        if input_width % len(IN_COLS):
            raise ValueError(
                f"stats input width {input_width} is not divisible by {len(IN_COLS)} columns"
            )
        history = input_width // len(IN_COLS)
        if history != expected_history:
            raise ValueError(f"{name}: expected H={expected_history}, stats have H={history}")
        learned_history = max(learned_history, history)
        spec = beamng_model_spec.make_main_spec(H=history, dt=DT)
        model = TrailerModel(input_width, len(stats["y_mean"]))
        restore_model(model, checkpoint.resolve())
        dynamics, cost, bound, _ = learned_util(
            scenario, spec, fiala_dyn, model, stats, **weights
        )
        runners[name] = DynamicsRunner(
            name=name,
            dynamics=jax.jit(dynamics),
            cost=cost,
            bound=bound,
            history=history,
            state_dim=ROW_WIDTH,
            steering_sign_to_beamng=1.0,
        )
        learned_config[name] = {
            "stats": str(stats_path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "history": history,
        }

    for name in names:
        if name not in ("fiala_surface", "fiala"):
            continue
        dynamics, cost, bound, _ = surface_fiala_util(
            scenario, s_weight=0, **weights
        )
        runners[name] = DynamicsRunner(
            name=name,
            dynamics=jax.jit(dynamics),
            cost=cost,
            bound=bound,
            history=None,
            state_dim=11,
            steering_sign_to_beamng=-1.0,
        )

    unknown = set(names) - set(runners)
    if unknown:
        raise ValueError(f"unknown dynamics model(s): {', '.join(sorted(unknown))}")
    return runners, learned_history, learned_config


def make_planner(
    runner: DynamicsRunner,
    samples: int,
    horizon: int,
    seed: int,
) -> MPPI_Jax:
    planner = MPPI_Jax(
        runner.state_dim,
        2,
        runner.dynamics,
        None,
        runner.cost,
        runner.bound,
        jnp.diag(jnp.array([2e-2, 0.2])),
        inverse_temp=100,
        K=samples,
        step=DT,
        T=horizon,
        alpha=0.01,
        gamma=0.0,
        history=runner.history,
    )
    # MPPI_Jax derives gamma internally; run_model.py explicitly uses zero.
    planner.gamma = 0.0
    planner.key = jax.random.key(seed)
    return planner


def capture_common_state(env: BeamNGTrailerEnv) -> np.ndarray:
    state = env.unwrapped._state
    friction = env.unwrapped.query_surface_friction()
    arc_length = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
    return np.asarray(
        [
            *astuple(state)[:10],
            *friction.dynamics_mu,
            arc_length,
        ],
        dtype=np.float64,
    )


def run_expert(
    env: BeamNGTrailerEnv,
    runner: DynamicsRunner,
    steps: int,
    samples: int,
    horizon: int,
    seed: int,
    target_kph: float,
    warmup_steps: int,
) -> dict:
    planner = make_planner(runner, samples, horizon, seed)
    env.reset()
    env.step(np.zeros(2))

    warmup_rows: list[np.ndarray] = []
    warm_control = np.asarray([0.0, 0.35 if target_kph > 0 else -0.35])
    for _ in range(warmup_steps):
        env.step(warm_control)
        common = capture_common_state(env)
        warmup_rows.append(learned_row(common, warm_control))

    states = [capture_common_state(env)]
    controls: list[np.ndarray] = []
    solve_ms: list[float] = []
    status = "completed"
    termination_steps = []

    for i in range(steps):
        planner_input = runner.planner_state(states[-1], warmup_rows)
        start = time.perf_counter()
        model_control = planner.run_mpc(planner_input)
        model_control.block_until_ready()
        solve_ms.append((time.perf_counter() - start) * 1e3)

        env_control = runner.control_to_beamng(np.asarray(model_control))
        if not np.isfinite(env_control).all():
            status = "nonfinite_control"
            break

        _, _, terminated, truncated, _ = env.step(env_control)
        common = capture_common_state(env)
        states.append(common)
        controls.append(env_control)
        warmup_rows.append(learned_row(common, env_control))

        speed_kph = np.hypot(common[4], common[5]) * 3.6
        hitch_deg = abs(wrap_angle(common[2] - common[3])) * 180.0 / np.pi
        print(
            f"\r[{runner.name:<13}] expert {i + 1:>4}/{steps}  "
            f"|v|={speed_kph:>6.1f} km/h  hitch={hitch_deg:>6.2f} deg  "
            f"solve={solve_ms[-1]:>6.1f} ms",
            end="",
            flush=True,
        )
        if terminated or truncated:
            # Keep the requested fixed-length experiment; record violations.
            termination_steps.append(i + 1)

    print()
    return {
        "states": np.asarray(states),
        "controls": np.asarray(controls).reshape(-1, 2),
        "initial_history": np.asarray(warmup_rows[:warmup_steps]),
        "solve_ms": np.asarray(solve_ms),
        "status": status,
        "termination_steps": np.asarray(termination_steps, dtype=np.int32),
    }


def compute_metrics(expert: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - expert
    error[:, I_YAW_TRACTOR] = wrap_angle(error[:, I_YAW_TRACTOR])
    error[:, I_YAW_TRAILER] = wrap_angle(error[:, I_YAW_TRAILER])
    hitch_error = wrap_angle(
        (predicted[:, I_YAW_TRACTOR] - predicted[:, I_YAW_TRAILER])
        - (expert[:, I_YAW_TRACTOR] - expert[:, I_YAW_TRAILER])
    )
    position_error = np.linalg.norm(error[:, :2], axis=1)
    finite = np.isfinite(error[:, :10]).all(axis=1)
    valid = finite.copy()
    valid[0] = False
    if not valid.any():
        return {"finite_fraction": float(finite.mean())}

    e = error[valid]
    terminal = np.flatnonzero(finite)[-1]
    metrics = {
        "finite_fraction": float(finite.mean()),
        "position_rmse_m": float(np.sqrt(np.mean(position_error[valid] ** 2))),
        "position_terminal_m": float(position_error[terminal]),
        "hitch_rmse_deg": float(np.sqrt(np.mean(hitch_error[valid] ** 2)) * 180 / np.pi),
        "hitch_terminal_deg": float(abs(hitch_error[terminal]) * 180 / np.pi),
        "vx_rmse_mps": float(np.sqrt(np.mean(e[:, 4] ** 2))),
        "vy_rmse_mps": float(np.sqrt(np.mean(e[:, 5] ** 2))),
        "yaw_rate_tractor_rmse": float(np.sqrt(np.mean(e[:, 6] ** 2))),
        "yaw_rate_trailer_rmse": float(np.sqrt(np.mean(e[:, 7] ** 2))),
    }
    return metrics


def save_expert(path: Path, name: str, expert: dict) -> None:
    np.savez_compressed(
        path / f"expert_{name}.npz",
        states=expert["states"].astype(np.float32),
        controls=expert["controls"].astype(np.float32),
        initial_history=expert["initial_history"].astype(np.float32),
        solve_ms=expert["solve_ms"].astype(np.float32),
        status=np.asarray(expert["status"]),
        termination_steps=expert["termination_steps"],
        state_channels=np.asarray(STATE_CHANNELS),
        control_channels=np.asarray(CONTROL_CHANNELS),
    )


def save_prediction(
    path: Path,
    expert_name: str,
    predictor_name: str,
    expert: dict,
    predicted: np.ndarray,
) -> None:
    error = predicted - expert["states"]
    error[:, I_YAW_TRACTOR] = wrap_angle(error[:, I_YAW_TRACTOR])
    error[:, I_YAW_TRAILER] = wrap_angle(error[:, I_YAW_TRAILER])
    np.savez_compressed(
        path / f"prediction_expert-{expert_name}_model-{predictor_name}.npz",
        expert_states=expert["states"].astype(np.float32),
        predicted_states=predicted.astype(np.float32),
        controls=expert["controls"].astype(np.float32),
        errors=error.astype(np.float32),
        state_channels=np.asarray(STATE_CHANNELS),
        control_channels=np.asarray(CONTROL_CHANNELS),
    )


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "expert",
        "predictor",
        "steps",
        "finite_fraction",
        "position_rmse_m",
        "position_terminal_m",
        "hitch_rmse_deg",
        "hitch_terminal_deg",
        "vx_rmse_mps",
        "vy_rmse_mps",
        "yaw_rate_tractor_rmse",
        "yaw_rate_trailer_rmse",
    ]
    with (path / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"comma-separated subset of {','.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000, help="MPPI samples K")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--target-kph", type=float, default=None,
                        help="single direction; default runs +80 and -30 km/h")
    parser.add_argument("--forward-kph", type=float, default=80.0)
    parser.add_argument("--backward-kph", type=float, default=-30.0)
    parser.add_argument("--environment-label", default="default",
                        help="environment identifier saved in the manifest")
    parser.add_argument("--track-csv", default=TrackConfig().csv)
    for history in (1, 4):
        _, stats, checkpoint = LEARNED_MODELS[f"learned_h{history}"]
        parser.add_argument(f"--h{history}-stats", type=Path, default=stats)
        parser.add_argument(f"--h{history}-checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn-index", type=int, default=800)
    parser.add_argument("--track-width", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_direction(args):
    names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    if not names:
        raise ValueError("--models must contain at least one model")

    scenario = BeamNGTrailerEnvConfig(
        ".",
        TrackConfig(mu=1.0, width=args.track_width, csv=args.track_csv),
        bng_pickup_trailer_cfg,
        SimulationConfig(dt=DT),
    )
    learned_models = {
        f"learned_h{h}": (h, getattr(args, f"h{h}_stats"), getattr(args, f"h{h}_checkpoint"))
        for h in (1, 4)
    }
    learned_models["learned"] = learned_models["learned_h4"]
    runners, learned_history, learned_config = build_runners(
        names, scenario, args.target_kph, learned_models
    )
    warmup_steps = max(learned_history, 1)

    output = args.output or OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "environment_label": args.environment_label,
        "scenario_config": asdict(scenario),
        "models": list(names),
        "steps": args.steps,
        "mppi_samples": args.samples,
        "mppi_horizon": args.horizon,
        "target_kph": args.target_kph,
        "seed": args.seed,
        "spawn_index": args.spawn_index,
        "dt": DT,
        "state_channels": STATE_CHANNELS,
        "control_channels": CONTROL_CHANNELS,
        "friction_source": "BeamNG wheel contacts; expert schedule replayed open loop",
        "learned": learned_config,
        "prior_friction": {"fiala": [1.0, 1.0], "fiala_surface": "measured"},
        "termination_policy": "record violations and continue to requested steps",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"output -> {output}")

    env = BeamNGTrailerEnv(
        config=scenario,
        use_custom_mu=False,
        spidx=args.spawn_index,
    )
    experts = {}
    try:
        for offset, name in enumerate(names):
            print(f"\n=== expert: {name} ===")
            experts[name] = run_expert(
                env,
                runners[name],
                args.steps,
                args.samples,
                args.horizon,
                args.seed + offset,
                args.target_kph,
                warmup_steps,
            )
            save_expert(output, name, experts[name])
    finally:
        env.close()

    print("\n=== open-loop cross-model replay ===")
    summary = []
    for expert_name, expert in experts.items():
        for predictor_name, runner in runners.items():
            predicted = runner.open_loop(
                expert["states"],
                expert["controls"],
                expert["initial_history"],
                DT,
            )
            metrics = compute_metrics(expert["states"], predicted)
            row = {
                "expert": expert_name,
                "predictor": predictor_name,
                "steps": len(expert["controls"]),
                **metrics,
            }
            summary.append(row)
            save_prediction(output, expert_name, predictor_name, expert, predicted)
            print(
                f"  expert={expert_name:<13} predictor={predictor_name:<13} "
                f"pos_rmse={metrics.get('position_rmse_m', np.nan):8.3f} m  "
                f"hitch_rmse={metrics.get('hitch_rmse_deg', np.nan):8.3f} deg  "
                f"vx_rmse={metrics.get('vx_rmse_mps', np.nan):8.3f} m/s  "
                f"finite={metrics['finite_fraction']:.1%}"
            )
    write_summary(output, summary)
    print(f"\nsaved {len(experts)} experts and {len(summary)} predictions -> {output}")


def main():
    args = parse_args()
    if args.steps < 1 or args.samples < 1 or args.horizon < 1:
        raise ValueError("steps, samples and horizon must be positive")
    if args.target_kph is not None:
        run_direction(args)
        return
    from argparse import Namespace
    root = args.output or OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    if args.forward_kph <= 0 or args.backward_kph >= 0:
        raise ValueError("forward-kph must be positive and backward-kph negative")
    for direction, speed in (("forward", args.forward_kph), ("backward", args.backward_kph)):
        run_direction(Namespace(**{**vars(args), "output": root / direction,
                                   "target_kph": speed}))


if __name__ == "__main__":
    main()
