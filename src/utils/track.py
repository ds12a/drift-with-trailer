"""
Taken from Uncertain Racecar Environment, Taekyung Kim
https://github.com/tkkim-robot/uncertain-racecar-gym

With modifications to add variable friction patches
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

from src.simulation.config.bicycle_config import TrackConfig


@dataclass(slots=True)
class TrackProjection:
    progress: float
    arc_length: float
    x: float
    y: float
    heading: float
    lateral_error: float
    curvature: float


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TrackModel:
    def __init__(
        self, centerline: np.ndarray, width: float, closed: bool = True, mu=1.5, friction_map=None
    ):

        if centerline.shape[0] < 4:
            raise ValueError("Track centerline must contain at least four points.")
        if np.allclose(centerline[0], centerline[-1]):
            centerline = centerline[:-1]
        centerline = np.asarray(centerline, dtype=float)
        deduped = [centerline[0]]
        for point in centerline[1:]:
            if np.linalg.norm(point - deduped[-1]) > 1e-4:
                deduped.append(point)
        centerline = np.asarray(deduped, dtype=float)
        if centerline.shape[0] < 4:
            raise ValueError("Track centerline must contain at least four distinct points.")
        self.centerline = centerline
        self.width = float(width)
        self.closed = bool(closed)

        self.friction_map = jnp.asarray(friction_map)
        self.mu = mu

        extended = np.vstack([self.centerline, self.centerline[0]])
        self._segments = np.diff(extended, axis=0)
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._segment_length_sq = self._segment_lengths * self._segment_lengths
        self._longest_segment_length_sq = self._segment_length_sq.max()
        self._segment_valid = self._segment_lengths > 1e-9
        self._segment_tangents = self._segments / np.maximum(self._segment_lengths[:, None], 1e-9)
        self._segment_normals = np.stack(
            [-self._segment_tangents[:, 1], self._segment_tangents[:, 0]], axis=1
        )
        self._segment_headings = np.arctan2(self._segments[:, 1], self._segments[:, 0])
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])
        self.length = float(self._cumulative[-1])

        self._headings = np.append(self._segment_headings, self._segment_headings[0])

        arc = self._cumulative[:-1]
        heading_unwrapped = np.unwrap(self._headings[:-1])
        curvature = np.gradient(heading_unwrapped, arc, edge_order=1)
        self._arc_samples = arc
        self._curvature_samples = curvature

    @classmethod
    def from_config(cls, config: TrackConfig) -> "TrackModel":
        frame = pd.read_csv(Path(config.csv))
        if {"x", "y"}.issubset(frame.columns):
            centerline = frame[["x", "y"]].to_numpy(dtype=float)
        else:
            centerline = frame.iloc[:, :2].to_numpy(dtype=float)

        friction_map = None
        if config.friction_csv is not None:
            friction = pd.read_csv(Path(config.friction_csv))
            friction_map = friction[["x", "y", "r", "mu"]].to_numpy(dtype=float)

        return cls(
            centerline=centerline,
            width=config.width,
            closed=config.closed,
            mu=config.mu,
            friction_map=friction_map,
        )

    def progress_to_arc(self, progress: float) -> float:
        return float(progress % 1.0) * self.length

    def arc_to_progress(self, arc_length: float) -> float:
        return ((arc_length % self.length) / self.length)

    def sample(self, progress: float) -> TrackProjection:
        arc = self.progress_to_arc(progress)
        index = int(np.searchsorted(self._cumulative, arc, side="right") - 1) % len(self._segments)
        segment_start = self.centerline[index]
        segment = self._segments[index]
        segment_length = self._segment_lengths[index]
        local_t = 0.0 if segment_length < 1e-9 else (arc - self._cumulative[index]) / segment_length
        point = segment_start + segment * np.clip(local_t, 0.0, 1.0)
        heading = float(np.arctan2(segment[1], segment[0]))
        tangent = segment / max(segment_length, 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        curvature = float(
            np.interp(arc, self._arc_samples, self._curvature_samples, period=self.length)
        )
        return TrackProjection(
            progress=self.arc_to_progress(arc),
            arc_length=arc,
            x=float(point[0]),
            y=float(point[1]),
            heading=heading,
            lateral_error=0.0,
            curvature=curvature,
        )

    def project(self, x: float, y: float, guess: int | None = None) -> tuple[TrackProjection, int]:
        start = (guess - 5) if guess is not None else 0
        end = (guess + 5) if guess is not None else len(self.centerline)
        if start > end:
            start, end = end, start

        window = np.arange(start, end)
        segments_window = np.take(
            self._segments, window, mode="wrap", axis=0
        )
        segments_len_window = np.take(
            self._segment_lengths, window, mode="wrap"
        )
        segments_sq_window = np.take(
            self._segment_length_sq, window, mode="wrap", axis=0
        )
        centerline_window = np.take(
            self.centerline, window, mode="wrap", axis=0
        )
        segments_normal_window = np.take(
            self._segment_normals, window, mode="wrap", axis=0
        )
        segments_heading_window = np.take(
            self._segment_headings, window, mode="wrap", axis=0
        )
        valid_window = np.take(
            self._segment_valid, window, mode="wrap", axis=0
        )

        point = np.asarray([x, y], dtype=float)
        delta_from_start = point - centerline_window
        print(guess)
        t = np.divide(
            np.einsum("ij,ij->i", delta_from_start, segments_window),
            segments_sq_window,
            out=np.zeros_like(segments_sq_window),
            where=valid_window,
        )
        t = np.clip(t, 0.0, 1.0)
        projected = centerline_window + segments_window * t[:, None]
        delta = point - projected
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        distance_sq = np.where(valid_window, distance_sq, np.inf)
        index = int(np.argmin(distance_sq))

        if not np.isfinite(distance_sq[index]):
            raise RuntimeError(f"Unable to project point onto track: ({x}, {y})")

        if guess and distance_sq[index] > (self._longest_segment_length_sq + self.width**2):
            raise RuntimeError(f"Guess {guess} is suspicious")

        signed_offset = float(np.dot(point - projected[index], segments_normal_window[index]))
        arc = float(self._cumulative[window[index]] + t[index] * segments_len_window[index])
        return (
            TrackProjection(
                progress=self.arc_to_progress(arc),
                arc_length=arc,
                x=float(projected[index, 0]),
                y=float(projected[index, 1]),
                heading=float(segments_heading_window[index]),
                lateral_error=signed_offset,
                curvature=float(
                    np.interp(arc, self._arc_samples, self._curvature_samples, period=self.length)
                ),
            ),
            window[index],
        )

    def spawn_pose(
        self, progress: float, lateral_error: float = 0.0, heading_error: float = 0.0
    ) -> tuple[float, float, float]:
        projection = self.sample(progress)
        tangent = np.array([np.cos(projection.heading), np.sin(projection.heading)])
        normal = np.array([-tangent[1], tangent[0]])
        position = np.array([projection.x, projection.y]) + normal * lateral_error
        yaw = wrap_angle(projection.heading + heading_error)
        return float(position[0]), float(position[1]), float(yaw)

    def lookahead_curvatures(self, progress: float, count: int, spacing_m: float) -> np.ndarray:
        base_arc = self.progress_to_arc(progress)
        arcs = base_arc + np.arange(count, dtype=float) * spacing_m
        return np.interp(
            arcs % self.length, self._arc_samples, self._curvature_samples, period=self.length
        )

    def out_of_bounds(self, lateral_error: float) -> bool:
        return abs(lateral_error) > (self.width * 0.5)

    def find_mu(self, x, y):
        # TODO handle overlapping ice patches? Currently just taking closest valid one

        if self.friction_map is None or self.friction_map.shape[0] == 0:
            return self.mu

        point = jnp.array([x, y], dtype=float)
        delta = point - self.friction_map[:, :2]
        distance_sq = jnp.einsum("ij,ij->i", delta, delta)

        dist_from_circ = jnp.sqrt(distance_sq) - self.friction_map[:, 2]
        i = jnp.argmin(dist_from_circ)

        mu = jnp.where(dist_from_circ[i] < 0, self.friction_map[i, 3], self.mu)
        return mu
