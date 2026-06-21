from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optuna

from experiments.exp_004_fiala_trailer_ice.utils.bicycle_driver import run_mpc
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.dynamics.vehicle.bicycle_fiala import gen_util_funs
from src.simulation.bicycle_env import BicycleEnv


@dataclass
class Config:
    K: int = 400  
    T: int = 75 
    step: float = 0.05  
    max_steps: int = 3000
    n_starts: int = 2 
    start_speed: float = 8.0 
    reverse: bool = False
    v_target: float | None = 35.0
    p_weight_anchor: float = 1e2 
    p_slow_weight: float = 1.0
    mppi_alpha: float = 0.05
    report_every: int = 150  
    crash_score: float = -1.0e3         # NaN/inf blow-up sentinel (meters-scale)


def objective(trial):
    lam = trial.suggest_float("lambda", 1e-2, 1e3, log=True)
    cvs = trial.suggest_float("cv_steer", 1e-3, 1.0, log=True)
    cva = trial.suggest_float("cv_accel", 1e-2, 4.0, log=True)
    sw  = trial.suggest_float("s_weight", 1e2, 1e9, log=True)
    cw  = trial.suggest_float("c_weight", 1e-2, 1e3, log=True)
    return run_mpc(
        "MPPI",
        ctl_args=(jnp.diag(jnp.array([cvs, cva])),),
        ctl_kwargs=dict(inverse_temp=lam, alpha=Config.mppi_alpha, K=Config.K, step=Config.step, T=Config.T),
        cost_kwargs=dict(reverse=False, v_target=Config.v_target, p_weight=Config.p_weight_anchor, s_weight=sw, c_weight=cw),
        headless=True, quiet=True, record=False, debug=False, return_metric=True, max_steps=Config.max_steps,
        env_kwargs={"renderer": None, "render_mode": None},
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--study-name", default="mppi_car_task004")
    ap.add_argument("--storage", default="sqlite:///optuna_mppi_car.db")
    ap.add_argument("--K", type=int, default=Config.K)
    ap.add_argument("--T", type=int, default=Config.T)
    ap.add_argument("--max-steps", type=int, default=Config.max_steps)
    ap.add_argument("--n-starts", type=int, default=Config.n_starts)
    ap.add_argument("--v-target", type=float, default=Config.v_target,
                    help="target speed in km/h; pass a negative number for None (max-speed mode)")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 trial, tiny budget, sanity check")
    args = ap.parse_args()

    cfg = Config(
        K=args.K, T=args.T, max_steps=args.max_steps, n_starts=args.n_starts,
        reverse=args.reverse,
        v_target=(None if args.v_target is not None and args.v_target < 0
                      else args.v_target),
    )
    if args.smoke:
        cfg.max_steps, cfg.n_starts, args.trials = 80, 1, 1

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.CmaEsSampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=cfg.max_steps // 3),
    )

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== best trial ===")
    bt = study.best_trial
    print(f"value (progress, m): {bt.value:.2f}")
    print("params:", json.dumps(bt.params, indent=2))
    print("derived gamma:", bt.user_attrs.get("gamma_derived"))
    print("stats:", json.dumps(
        {k: bt.user_attrs.get(k) for k in
         ("mean_speed", "laps", "mean_abs_slip_r", "max_abs_slip_r", "any_crash")},
        indent=2))

    with open("best_mppi_car_params.json", "w") as f:
        json.dump({"value": bt.value, "params": bt.params,
                   "gamma_derived": bt.user_attrs.get("gamma_derived")}, f, indent=2)
    print("\nwrote best_mppi_car_params.json")


if __name__ == "__main__":
    main()