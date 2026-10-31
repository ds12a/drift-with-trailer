"""Fiala-trailer adaptation of Liniger, Domahidi, and Morari's MPCC.

The contouring formulation follows equations (11) and (13) in:

    A. Liniger, A. Domahidi, and M. Morari, "Optimization-based
    autonomous racing of 1:43 scale RC cars," Optimal Control
    Applications and Methods 36(5), 2015. doi:10.1002/oca.2123.

The paper's bicycle model is replaced by the repository's existing Fiala
tractor-trailer dynamics. Reverse progress on a closed track and the trailer
hitch constraint are explicit adaptations; see :class:`MPCC`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
from src.utils.track import TrackModel


MPCC_REFERENCE = {
    "authors": "Liniger, Domahidi, and Morari",
    "title": "Optimization-Based Autonomous Racing of 1:43 Scale RC Cars",
    "venue": "Optimal Control Applications and Methods 36(5), 2015",
    "doi": "10.1002/oca.2123",
    "arxiv": "1711.07300",
    "formulation": "Equations (11) and (13)",
    "official_implementation": "https://github.com/alexliniger/MPCC",
    "official_weight_config": "https://github.com/alexliniger/MPCC/blob/master/C%2B%2B/Params/cost.json",
    "implementation_note": "The official repository states that its current C++ version is not the paper's original implementation",
    "trailer_context": {
        "authors": "Aertssen, Huisman, Besselink, Elfring, and van de Molengraft",
        "title": "Trajectory Planning for an Articulated Commercial Vehicle using Model Predictive Contouring Control",
        "arxiv": "2604.24064",
        "scope": "Motivates explicit trailer safety treatment; this controller does not reproduce the paper's multi-anchor trajectory planner",
    },
}


@dataclass(frozen=True)
class MPCCWeights:
    """Liniger equation (13) weights plus trailer/ablation adaptations."""

    contour: float = 0.1
    lag: float = 500.0
    progress: float = 0.02
    steer_rate: float = 5e-3
    accel_rate: float = 1e-4
    progress_rate: float = 1e-5
    hitch: float = 0.0
    hitch_rate: float = 0.0
    speed: float = 0.0
    speed_violation: float = 0.0
    track_slack_linear: float = 1e4
    track_slack_quadratic: float = 1e5
    hitch_slack_linear: float = 1e4
    hitch_slack_quadratic: float = 1e5


class MPCC:
    """Nonlinear realization of Liniger et al.'s MPCC problem.

    Like equation (13), the physical system is augmented with virtual path
    progress, contouring and lag errors are penalized, progress is rewarded,
    and changes in physical and virtual inputs are penalized. The optimized
    input is ``[steer, accel, progress_speed]``; only the first two entries
    are sent to the vehicle.

    Differences from the paper:

    * the existing Fiala tractor-trailer model replaces the RC-car model;
    * a direction multiplier maps nonnegative virtual progress to reverse driving;
    * a hitch-angle soft constraint is added beside the paper's track slack;
    * optional hitch, hitch-rate, and target-speed costs adapt the controller to
      an articulated target-velocity ablation;
    * piecewise-constant input blocking reduces the online decision dimension;
    * the nonlinear problem is solved directly with JAX gradients and L-BFGS-B
      instead of the paper's one-step LTV-QP real-time iteration.

    Vehicle inputs use the same normalized analytic steering frame as the Fiala
    MPPI and IPOPT baselines.
    """

    def __init__(
        self,
        config,
        v_target: float,
        horizon: int = 40,
        control_horizon: int | None = None,
        weights: MPCCWeights | dict | None = None,
        max_iter: int = 35,
        ftol: float = 1e-4,
        max_line_search_steps: int = 20,
        max_progress_speed: float | None = None,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if control_horizon is None:
            control_horizon = horizon
        if control_horizon <= 0 or control_horizon > horizon:
            raise ValueError("control_horizon must be in [1, horizon]")
        if ftol <= 0:
            raise ValueError("ftol must be positive")
        if max_line_search_steps <= 0:
            raise ValueError("max_line_search_steps must be positive")
        if not np.isfinite(v_target):
            raise ValueError("v_target must be finite")
        if max_progress_speed is not None and max_progress_speed <= 0:
            raise ValueError("max_progress_speed must be positive")

        self.track = TrackModel.from_config(config.track)
        if not self.track.closed:
            raise ValueError("MPCC currently requires a closed track")
        self.vehicle = config.vehicle
        self.dt = float(config.simulation.dt)
        self.v_target = float(v_target)
        self.direction = -1.0 if self.v_target < 0 else 1.0
        self.horizon = int(horizon)
        self.control_horizon = int(control_horizon)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.max_line_search_steps = int(max_line_search_steps)
        self.weights = (
            weights
            if isinstance(weights, MPCCWeights)
            else MPCCWeights(**(weights or {}))
        )

        # Reuse the project's Fiala dynamics rather than maintaining another
        # controller-specific vehicle model.
        self.dynamics, _, _, _ = gen_util_funs(config, v_target=v_target, s_weight=0)

        arc = np.concatenate([self.track._arc_samples, [self.track.length]])
        centerline = np.vstack([self.track.centerline, self.track.centerline[0]])
        tangents = np.vstack(
            [self.track._segment_tangents, self.track._segment_tangents[0]]
        )
        self._arc = jnp.asarray(arc)
        self._center_x = jnp.asarray(centerline[:, 0])
        self._center_y = jnp.asarray(centerline[:, 1])
        self._tangent_x = jnp.asarray(tangents[:, 0])
        self._tangent_y = jnp.asarray(tangents[:, 1])
        self._track_length = float(self.track.length)

        # Liniger et al. impose 0 <= v_theta <= v_bar. For this target-speed
        # ablation, v_bar defaults to the requested speed magnitude.
        max_progress = (
            abs(self.v_target)
            if max_progress_speed is None
            else float(max_progress_speed)
        )
        self.max_progress_speed = max(max_progress, 1e-6)
        progress_bounds = (0.0, self.max_progress_speed)
        self._bounds = (
            [(-1.0, 1.0), (-1.0, 1.0), progress_bounds]
            * self.control_horizon
        )

        # Input blocking keeps the full state prediction horizon but optimizes
        # fewer piecewise-constant inputs. Near-term commands are still replanned
        # every simulator step. This is the primary real-time speed adaptation.
        block_indices = np.floor(
            np.arange(self.horizon) * self.control_horizon / self.horizon
        ).astype(np.int32)
        self._block_indices_np = block_indices
        self._block_indices = jnp.asarray(block_indices)
        self._block_starts = np.array(
            [np.flatnonzero(block_indices == i)[0] for i in range(self.control_horizon)]
        )

        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))
        self.last_trajectory: np.ndarray | None = None
        self.last_block_trajectory: np.ndarray | None = None
        self.last_result = None
        self.previous_control: np.ndarray | None = None
        self._compiled = False

    @staticmethod
    def _wrap_angle(angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _reference(self, theta):
        theta = jnp.mod(theta, self._track_length)
        x = jnp.interp(theta, self._arc, self._center_x)
        y = jnp.interp(theta, self._arc, self._center_y)
        tx = jnp.interp(theta, self._arc, self._tangent_x)
        ty = jnp.interp(theta, self._arc, self._tangent_y)
        norm = jnp.sqrt(tx * tx + ty * ty + 1e-9)
        return x, y, tx / norm, ty / norm

    def _stage_cost(self, state, theta, control, previous_control):
        """Liniger et al. (13a), plus target-speed and trailer safety costs."""
        w = self.weights
        ref_x, ref_y, tx, ty = self._reference(theta)
        dx = state[0] - ref_x
        dy = state[1] - ref_y
        # Approximate errors from Liniger et al. (11a)-(11b). Their signs do
        # not affect the squared objective, but are retained here verbatim.
        contour_error = ty * dx - tx * dy
        lag_error = -tx * dx - ty * dy

        hitch = self._wrap_angle(state[3] - state[2])
        hitch_rate = state[7] - state[6]
        delta_control = control - previous_control

        track_slack = jax.nn.relu(
            jnp.abs(contour_error) - self.track.width * 0.5
        )
        hitch_slack = jax.nn.relu(jnp.abs(hitch) - self.vehicle.max_hitch)
        directed_speed = self.direction * state[4]
        speed_error = directed_speed - self.max_progress_speed
        speed_violation = (
            jax.nn.relu(directed_speed - self.max_progress_speed) ** 2
            + jax.nn.relu(-directed_speed) ** 2
        )

        return (
            w.contour * contour_error**2
            + w.lag * lag_error**2
            - w.progress * control[2] * self.dt
            + w.steer_rate * delta_control[0] ** 2
            + w.accel_rate * delta_control[1] ** 2
            + w.progress_rate * delta_control[2] ** 2
            + w.hitch * hitch**2
            + w.hitch_rate * hitch_rate**2
            + w.speed * speed_error**2
            + w.speed_violation * speed_violation
            + w.track_slack_linear * track_slack
            + w.track_slack_quadratic * track_slack**2
            + w.hitch_slack_linear * hitch_slack
            + w.hitch_slack_quadratic * hitch_slack**2
        )

    def _objective(self, flat_controls, initial_state, initial_control):
        block_controls = flat_controls.reshape(self.control_horizon, 3)
        controls = block_controls[self._block_indices]
        initial_theta = initial_state[9]

        def rollout(carry, control):
            state, theta, previous_control = carry
            state = state + self.dt * self.dynamics(state, control[:2])
            # The shared Fiala dynamics use arc length as a local projection
            # hint and expect it in [0, track_length). Keep it periodic inside
            # the prediction too, especially when reversing through arc zero.
            state = state.at[9].set(jnp.mod(state[9], self._track_length))
            theta = theta + self.direction * self.dt * control[2]
            cost = self._stage_cost(state, theta, control, previous_control)
            return (state, theta, control), cost

        _, costs = jax.lax.scan(
            rollout,
            (initial_state, initial_theta, initial_control),
            controls,
        )
        return jnp.sum(costs)

    def reset(self, previous_vehicle_control=None):
        """Clear the warm start and optionally set the last applied input."""
        self.last_trajectory = None
        self.last_block_trajectory = None
        self.last_result = None
        if previous_vehicle_control is None:
            self.previous_control = None
            return
        previous_vehicle_control = np.asarray(previous_vehicle_control, dtype=float)
        if previous_vehicle_control.shape != (2,) or not np.all(
            np.isfinite(previous_vehicle_control)
        ):
            raise ValueError("previous_vehicle_control must be a finite 2-vector")
        self.previous_control = np.array(
            [*previous_vehicle_control, self.max_progress_speed], dtype=float
        )

    def _initial_guess(self):
        if self.last_trajectory is not None:
            shifted = np.vstack(
                [self.last_trajectory[1:], self.last_trajectory[-1]]
            )
            return shifted[self._block_starts]
        guess = np.zeros((self.control_horizon, 3), dtype=float)
        guess[:, 1] = 0.35 if self.v_target > 0 else -0.35
        guess[:, 2] = self.max_progress_speed
        return guess

    def compile(self):
        """Compile the objective before the controller enters the timed loop."""
        if self._compiled:
            return
        x, y = self.track.centerline[0]
        tx, ty = self.track._segment_tangents[0]
        heading = np.arctan2(ty, tx)
        state = jnp.asarray(
            [
                x,
                y,
                heading,
                heading,
                self.v_target,
                0.0,
                0.0,
                0.0,
                self.track.mu,
                0.0,
            ]
        )
        previous_control = jnp.asarray(
            [0.0, 0.0, self.max_progress_speed]
        )
        value, _ = self._value_and_grad(
            jnp.asarray(self._initial_guess().reshape(-1)),
            state,
            previous_control,
        )
        value.block_until_ready()
        self._compiled = True

    def run_mpc(self, state):
        """Optimize and return normalized ``[steer, accel]``."""
        state = np.asarray(state, dtype=float)
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite 10-vector")

        initial = self._initial_guess().reshape(-1)
        jax_state = jnp.asarray(state)
        if self.previous_control is None:
            progress_speed = np.clip(
                abs(state[4]), 0.0, self.max_progress_speed
            )
            previous_control = np.array([0.0, 0.0, progress_speed])
        else:
            previous_control = self.previous_control
        jax_previous_control = jnp.asarray(previous_control)

        def objective(controls):
            value, gradient = self._value_and_grad(
                jnp.asarray(controls), jax_state, jax_previous_control
            )
            return float(value), np.asarray(gradient, dtype=float)

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=self._bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": self.ftol,
                "maxls": self.max_line_search_steps,
            },
        )
        block_trajectory = np.asarray(result.x, dtype=float).reshape(
            self.control_horizon, 3
        )
        trajectory = block_trajectory[self._block_indices_np]
        if not np.isfinite(result.fun) or not np.all(np.isfinite(block_trajectory)):
            raise RuntimeError(f"MPCC solve failed: {result.message}")

        # A finite iterate remains useful when L-BFGS-B reaches its iteration
        # cap, matching the IPOPT baseline's treatment of capped solves.
        self.last_result = result
        self.last_trajectory = trajectory
        self.last_block_trajectory = block_trajectory
        self.previous_control = trajectory[0].copy()
        return trajectory[0, :2].copy()
