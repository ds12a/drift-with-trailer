"""Replay bundled history/friction candidates on the existing turn_div segments.

Run from the repository root:
    MPLBACKEND=Agg .venv/bin/python -m experiments.exp_008_beamng.replay_history_friction_divergence

The physical initial states, future controls and analytic predictions are taken
from turn_div. Each learned candidate receives H measured history rows recovered
from the original full trajectory, including the extra context required by H=8.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

# Import this before the plotting/runner modules to select the CPU backend on
# machines without CUDA and avoid incompatible installed PJRT plugins.
from experiments.exp_008_beamng import rerun_div_from_waypoints as replay
from experiments.exp_008_beamng import plot_model_divergence as plotting
from experiments.exp_008_beamng import run_model_for_div as div
from experiments.exp_008_beamng.experiment_paths import DATA_ROOT, MODEL_BUNDLE, resolve_saved_path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts


def restore_checked(model, checkpoint: Path) -> None:
    """Read local Orbax arrays without restoring saved CUDA sharding."""
    for index, layer in enumerate(model.model.layers):
        if not hasattr(layer, "kernel"):
            continue
        for field in ("kernel", "bias"):
            value = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "ocdbt",
                        "base": f"file://{checkpoint.resolve()}",
                        "path": f"model.layers.{index}.{field}.value",
                    },
                }, open=True,
            ).result().read().result()
            parameter = getattr(layer, field)
            if value.shape != parameter.value.shape:
                raise ValueError(
                    f"{checkpoint}: layer {index}/{field}: "
                    f"expected {parameter.value.shape}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite weights in {checkpoint}")
            parameter.value = jnp.asarray(value)


def load_npz(path: Path) -> dict:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def measured_segment(run_dir: Path, history: int) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expert = load_npz(run_dir / f"expert_{replay.BASELINE_EXPERT}.npz")
    original = resolve_saved_path(manifest["source_run"])
    source = replay.load_source_expert(original)
    start = manifest["waypoint_selection"]["nearest_state_index"]
    count = len(expert["controls"])
    np.testing.assert_array_equal(
        expert["states"], source["states"][start:start + count + 1]
    )
    np.testing.assert_array_equal(
        expert["controls"], source["controls"][start:start + count]
    )
    expert["initial_history"] = replay.history_at_state(
        source["states"], source["controls"], start, history,
    )
    np.testing.assert_array_equal(
        expert["initial_history"][-1, :10], expert["states"][0, :10]
    )
    return expert, manifest


def write_index(output: Path, cases: list[dict], source: Path = Path('turn_div')) -> None:
    lines = [
        f"# History/friction divergence on {source.name}",
        "",
        f"Each candidate uses the same measured five-second {source.name} segments "
        "(100 transitions at dt=0.05 s), in all four environments and both directions. "
        "The reference trajectory is the original learned_h4-controlled BeamNG run. "
        "Initial history uses the candidate's last H measured rows from the full "
        "source trajectory; no history is padded. Future controls are identical "
        "for all candidates. These are offline open-loop replays.",
        "",
        "Every figure retains spatial rollout, wrapped absolute hitch-angle error "
        "in degrees and velocity-vector error sqrt(delta_vx² + delta_vy²) in m/s. "
        "Analytic and friction-aware analytic predictions are copied unchanged "
        f"from {source.name}. Fixed/mixed describes training friction, not evaluation "
        "friction. Figure RMSE tables exclude the initial state.",
        "",
        "## Main comparisons: best validation checkpoints",
        "",
        "| Candidate | History | Training friction | Forward | Backward |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for selection in ("best", "other"):
        if selection == "other":
            lines += ["", "## Last checkpoints and rollout-refined reference", "",
                      "| Candidate | History | Training friction | Forward | Backward |",
                      "| --- | ---: | --- | --- | --- |"]
        for case in cases:
            if (case["selection"] == "best") != (selection == "best"):
                continue
            name = case["name"]
            links = [
                f"[{direction}]({name}/combined/{direction}/"
                f"divergence_expert-{replay.BASELINE_EXPERT}.png)"
                for direction in ("forward", "backward")
            ]
            lines.append(
                f"| {name} | {case['H']} | {case['training_friction']} | "
                + " | ".join(links) + " |"
            )
    lines += [
        "", "Raw reference/prediction archives and manifests are under "
        "`<candidate>/<environment>/<direction>/`. Per-direction RMSE tables "
        "are beside the figures. `metrics.csv` combines all figures; "
        "`cases.json` records the checkpoints and matching stats.",
        "", "Reproduce from the repository root:", "", "```bash",
        "MPLBACKEND=Agg .venv/bin/python -m "
        "experiments.exp_008_beamng.replay_history_friction_divergence "
        f"--source {source} --output {output}",
        "```", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=MODEL_BUNDLE)
    parser.add_argument("--source", type=Path, default=DATA_ROOT / "turn_div")
    parser.add_argument("--output", type=Path, default=Path("turn_div_history_friction_20261029"))
    parser.add_argument("--case", action="append", help="candidate name; repeat to select a subset")
    args = parser.parse_args()
    cases = json.loads((args.bundle / "bundled_cases.json").read_text())
    if args.case:
        missing = set(args.case) - {case["name"] for case in cases}
        if missing:
            raise ValueError(f"unknown candidates: {sorted(missing)}")
        cases = [case for case in cases if case["name"] in args.case]
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    for case in cases:
        for key in ("checkpoint", "stats"):
            if not (args.bundle / case[key]).exists():
                raise FileNotFoundError(args.bundle / case[key])
    args.output.mkdir(parents=True)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    (args.output / "evaluation_config.json").write_text(json.dumps({
        "source": str(args.source.resolve()),
        "bundle": str(args.bundle.resolve()),
        "environments": list(replay.SOURCE_ENVIRONMENTS),
        "directions": ["forward", "backward"],
        "cases": [case["name"] for case in cases],
    }, indent=2))
    div.restore_model = restore_checked
    terrain_cache = {}
    metrics = []

    for case in cases:
        name, history = case["name"], int(case["H"])
        stats_path = (args.bundle / case["stats"]).resolve()
        checkpoint = (args.bundle / case["checkpoint"]).resolve()
        selection = case["selection"]
        title = f"H={history} | {case['training_friction']}-friction training | {selection} checkpoint"
        plotting.MODEL_LABELS[name] = "Learned"
        plotting.MODEL_COLORS[name] = "tab:blue"
        print(f"\n{name}: restoring {checkpoint}", flush=True)
        for direction in ("forward", "backward"):
            combined_rows = []
            first_run = args.source / replay.SOURCE_ENVIRONMENTS[0] / direction
            first_manifest = json.loads((first_run / "manifest.json").read_text())
            runners, _, learned_config = div.build_runners(
                (name,), replay.scenario_from_manifest(first_manifest),
                float(first_manifest["target_kph"]),
                {name: (history, stats_path, checkpoint)},
            )
            for environment, label in plotting.COMBINED_ENVIRONMENTS:
                source_run = args.source / environment / direction
                expert, manifest = measured_segment(source_run, history)
                dt = float(manifest["dt"])
                if dt != div.DT:
                    raise ValueError(f"unexpected dt={dt} in {source_run}")
                output = args.output / name / environment / direction
                output.mkdir(parents=True)
                manifest.update(
                    models=[name, "fiala_surface", "fiala"],
                    learned=learned_config,
                    ablation_case=case,
                    replay_source=str(source_run.resolve()),
                    initial_history_length=history,
                    stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                )
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
                div.save_expert(output, replay.BASELINE_EXPERT, expert)
                predicted = runners[name].open_loop(
                    expert["states"], expert["controls"], expert["initial_history"], dt,
                )
                if not np.isfinite(predicted).all():
                    print(f"WARNING: nonfinite prediction: {name}/{environment}/{direction}", flush=True)
                div.save_prediction(output, replay.BASELINE_EXPERT, name, expert, predicted)
                for analytic in ("fiala", "fiala_surface"):
                    filename = f"prediction_expert-{replay.BASELINE_EXPERT}_model-{analytic}.npz"
                    shutil.copy2(source_run / filename, output / filename)
                predictions = plotting.load_predictions(output)[replay.BASELINE_EXPERT]
                div.write_summary(output, [
                    {"expert": replay.BASELINE_EXPERT, "predictor": predictor,
                     "steps": len(expert["controls"]),
                     **div.compute_metrics(data["expert_states"], data["predicted_states"])}
                    for predictor, data in predictions.items()
                ])
                combined_rows.append((
                    label, output, predictions, manifest,
                    plotting.terrain_for_environment(
                        manifest["environment_label"], plotting.DEFAULT_LEVELS_DIR, terrain_cache,
                    ), plotting.load_track(manifest, source_run),
                ))
            combined_output = args.output / name / "combined" / direction
            plotting.plot_combined_expert(
                combined_rows, replay.BASELINE_EXPERT, combined_output,
                log_errors=False, max_seconds=None, full_rollout=True,
                figure_title=title,
            )
            plt.close("all")
            with (combined_output / f"rmse_expert-{replay.BASELINE_EXPERT}.csv").open() as file:
                metrics.extend({"candidate": name, "history": history,
                                "training_friction": case["training_friction"],
                                "checkpoint_selection": selection, **row}
                               for row in csv.DictReader(file))
            print(f"saved {name}/combined/{direction}", flush=True)

    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    write_index(args.output, cases, args.source)
    print(f"Complete: {len(cases)} candidates, {2 * len(cases)} combined figures: {args.output}", flush=True)


if __name__ == "__main__":
    main()
