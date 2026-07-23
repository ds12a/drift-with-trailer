import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp

# from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
# from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.learning.models.trailer_nn import TrailerModel
# from src.learning.models.trailer_spec import KIN_FS, kin
# from src.learning.models.trailer_spec_nores import RAW_FS, kin_zeros
from src.learning.datasets.trailer_data import DataLoader, FeatureSpec
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs, TrackProjection
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from src.utils.track import TrackModel

# spec = KIN_FS
# kin_fn = kin_zeros



def gen_util_funs(
    params: TrailerBicycleEnvConfig,
    spec: FeatureSpec,
    kin_fn,
    model: TrailerModel,
    loader: DataLoader,
    reverse=False,
    v_target=None,
    p_weight=1e4,
    p_slow_weight=1e0,
    c_weight=1e-2,
    a_weight=1e5,
):

    D_STATE_DIM = 7
    D_U_DIM = 2
    D_EXTRA_DIM = 1
    K_STATE_DIM = 7
    M_STATE_DIM = 6
    H = spec.H
    dt = params.simulation.dt

    M_OUT_DIM = 4
    
    reverse = 1 if reverse else -1
    step = params.simulation.dt
    x_mean = jnp.asarray(loader.x_mean)
    x_std  = jnp.asarray(loader.x_std)
    y_mean, y_std = jnp.asarray(loader.y_mean), jnp.asarray(loader.y_std)

    # print(x_mean, x_std, y_mean, y_std)

    track = TrackModel.from_config(params.track)

    def _project_to_track(x, y, guess) -> tuple[TrackProjection, jax.Array]:
        """
        From Uncertain Racecar Gym, adapted
        """
        WINDOW = 10
        if guess is not None:
            window = ((guess - WINDOW / 2).astype(jnp.int32)) + jnp.arange(WINDOW)
        else:
            window = jnp.arange(len(track.centerline))

        segments_window = jnp.take(track._segments, window, mode="wrap", axis=0)
        segments_len_window = jnp.take(track._segment_lengths, window, mode="wrap")
        segments_sq_window = jnp.take(track._segment_length_sq, window, mode="wrap")
        centerline_window = jnp.take(track.centerline, window, mode="wrap", axis=0)
        segments_normal_window = jnp.take(track._segment_normals, window, mode="wrap", axis=0)
        segments_heading_window = jnp.take(track._segment_headings, window, mode="wrap")
        valid_window = jnp.take(track._segment_valid, window, mode="wrap")
        cumulative_window = jnp.take(track._cumulative, window, mode="wrap")

        point = jnp.stack([x, y])
        delta_from_start = point - centerline_window

        denom = jnp.where(valid_window, segments_sq_window, 1.0)
        t = jnp.where(
            valid_window,
            jnp.einsum("ij,ij->i", delta_from_start, segments_window) / denom,
            0.0,
        )
        t = jnp.clip(t, 0.0, 1.0)
        projected = centerline_window + segments_window * t[:, None]
        delta = point - projected
        distance_sq = jnp.einsum("ij,ij->i", delta, delta)
        distance_sq = jnp.where(valid_window, distance_sq, jnp.inf)
        index = jnp.argmin(distance_sq)  # stays traced; dynamic indexing -> gather

        signed_offset = jnp.dot(point - projected[index], segments_normal_window[index])
        arc = cumulative_window[index] + t[index] * segments_len_window[index]
        return (
            TrackProjection(
                progress=track.arc_to_progress(arc),
                arc_length=arc,
                x=projected[index, 0],
                y=projected[index, 1],
                heading=segments_heading_window[index],
                lateral_error=signed_offset,
                curvature=jnp.interp(
                    arc, track._arc_samples, track._curvature_samples, period=track.length
                ),
            ),
            window[index],
        )

    # Dynamics state: [x, y, phi1, phi2, vx, vy, mu, delta, a] + ctrl [delta, a] + caching [track_pos]
    # Needs to keep for u history
    # In model state: [sin(hitch), cos(hitch), vx, vy, mu, delta, a], mu is only for prior tanh
    # pred [ax, ay, phi1dot, phi2dot]
    def dynamics(x, u):  # passed as windows
        x_windows = x.reshape(H, D_STATE_DIM + D_U_DIM + D_EXTRA_DIM)
        old_u = x_windows[-1][-3:-1]
        x_windows = x_windows.at[-1, -3:-1].set(u)
        # x_windows[-1][-3], x_windows[-1][-2] = u[0], u[1]  # Control

        def slice_kin(window):
            hitch = window[2] - window[3]
            
            return jnp.stack([
                jnp.sin(hitch),  # sh
                jnp.cos(hitch),  # ch
                window[4],  # vx
                window[5],  # vy
                window[6],  # mu
                window[7],  # delta
                window[8],  # a
            ])
        def slice_mod_raw(window):
            def row(w):
                hitch = w[2] - w[3]
                return jnp.stack([jnp.sin(hitch), jnp.cos(hitch), w[4], w[5], w[7], w[8]])
            rows = jax.vmap(row)(window)
            return (rows.reshape(-1) - x_mean) / x_std
        
        kin_in = slice_kin(x_windows[-1]).flatten()
        model_in = slice_mod_raw(x_windows).flatten()[None, ...]

        xpos, ypos, phi1, phi2, vx, vy, *_ = x_windows[-1]
        arc_len = x_windows[-1][-1]
        pred = kin_fn(kin_in)
        pred += model(model_in)[0] * y_std + y_mean

        ax, ay, phi1dot, phi2dot = pred
        xdot = vx * jnp.cos(phi1) - vy * jnp.sin(phi1)
        ydot = vx * jnp.sin(phi1) + vy * jnp.cos(phi1)

        index = jnp.searchsorted(track._cumulative, arc_len, side="right") - 1

        # jax.debug.print("In: {}, Kin pred: {}, Model pred: {}", x, kin_fn(kin_in), model(model_in)[0] * y_std + y_mean)

        projection_curr, _ = _project_to_track(xpos, ypos, index)
        projection_next, _ = _project_to_track(xpos + dt * xdot, ypos + dt * ydot, index)

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / dt

        du = (u - old_u) / step  # Goofy

        dx = jnp.array([
            xdot, ydot, phi1dot, phi2dot, ax, ay, 0, du[0], du[1], track_vel
        ])
        # dx_history = (x_windows[1:] - x_windows[:-1]) / dt
        # dx_window = jnp.concatenate([dx_history, dx[None, :]], axis=0)
        # return dx_window.flatten()
        return dx


    @jax.jit
    def cost(x, u, t):

        x_windows = x.reshape(H, D_STATE_DIM + D_U_DIM + D_EXTRA_DIM)
        x = x_windows[-1] # discard others
        xpos, ypos, phi1, phi2, vx, vy, *_ = x
        arc_len = x[-1]
        
        # Tunable values
        gvx = vx * jnp.cos(phi1) - vy * jnp.sin(phi1)
        gvy = vx * jnp.sin(phi1) + vy * jnp.cos(phi1)

        index = jnp.searchsorted(track._cumulative, arc_len, side="right") - 1

        projection_curr, _ = _project_to_track(xpos, ypos, index)
        projection_next, _ = _project_to_track(xpos + step * gvx, ypos + step * gvy, index)

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / step

        violation = jnp.maximum(
            0, jnp.abs(projection_curr.lateral_error) - (params.track.width * 0.5) * 0.9 + 0.1
        )

        def wrap_angle(angle):
            return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        hitch_angle = wrap_angle(phi1 - phi2)

        violation += jnp.maximum(0, jnp.abs(hitch_angle) - params.vehicle.max_hitch)

        if v_target is None:
            v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[4])
        else:
            v_term = p_weight * jnp.abs(v_target + reverse * jnp.abs(track_vel) * jnp.sign(vx))
            # v_baseline = jnp.minimum(max_safe_v, v_target)
            # # If v is above threshold use actual car velocity instead of track velocity to stop cheating
            # v_car = jnp.where(nominal_v > max_safe_v, nominal_v, track_vel)

            # v_term = p_weight * jnp.maximum(
            #     0, v_car - v_baseline
            # ) + p_weight * p_slow_weight * jnp.maximum(0, v_baseline - v_car)

        c = (
            0.99**t * (1e12 * violation)
            + v_term
            + projection_curr.lateral_error**2 * c_weight
            + jnp.abs(hitch_angle) * a_weight
        )

        # jax.debug.print("cost {c}", c=c)
        return c
    

    def bound(u):
        return jnp.clip(u, jnp.array([-1, -1]), jnp.array([1, 1]))

    def bound_der(u):
        return u
    return dynamics, cost, bound, bound_der
