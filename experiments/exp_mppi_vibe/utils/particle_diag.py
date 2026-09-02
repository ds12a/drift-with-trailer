"""
Per-control-step diagnostic logging for the particle controllers.

One CSV row per control step, keyed by track position, so ESS / acceptance /
diversity can be plotted against where on the lap the controller lost it. This
is the rung-0 instrument: if ESS is flat across successful and failing runs,
weight degeneracy is not the binding constraint and the whole direction stops.

Everything is pulled to host once per control step and only as scalars, so the
transfer is a few floats -- not the (K, T, x_d) rollout tensor.
"""

from pathlib import Path

import numpy as np


class ParticleDiag:
    """
    Accumulates scalar diagnostics and writes a CSV on `save()`.

    Handles both controllers: RBR emits feasibility counters, BSMC emits
    per-checkpoint ESS / acceptance, and unknown keys are reduced generically so
    adding a diagnostic to a controller needs no change here.
    """

    def __init__(self, controller: str, path: str | Path):
        self.controller = controller
        self.path = Path(path)
        self.rows: list[dict] = []

    @staticmethod
    def _flatten(key, value) -> dict:
        arr = np.asarray(value)

        if key == "unique_anc0":
            # padded with -1; the count is the number that survived
            return {"unique_anc0": int((arr >= 0).sum())}

        if arr.ndim == 0:
            return {key: float(arr)}

        if arr.size == 0:
            return {}

        if key == "n_unsafe":
            # (T,) over the horizon: the shape matters more than any one entry
            return {
                "unsafe_mean": float(arr.mean()),
                "unsafe_max": float(arr.max()),
                "unsafe_final": float(arr[-1]),
            }

        # per-checkpoint vectors (ess_pre, ess_post, accept, sigma_S, d_cost)
        out = {f"{key}_{i}": float(v) for i, v in enumerate(arr)}
        out[f"{key}_mean"] = float(arr.mean())
        return out

    def add(self, step: int, arc_length: float, vx: float, diag: dict) -> None:
        row = {"step": step, "arc": arc_length, "vx": vx}
        for key, value in diag.items():
            row.update(self._flatten(key, value))
        self.rows.append(row)

    def save(self) -> Path | None:
        if not self.rows:
            return None

        columns: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            f.write(",".join(columns) + "\n")
            for row in self.rows:
                f.write(",".join(_fmt(row.get(c)) for c in columns) + "\n")

        print(f"[particle_diag] {self.controller}: {len(self.rows)} rows -> {self.path}")
        return self.path


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"