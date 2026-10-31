"""Run the BeamNG sweep against every installed ground environment.

Each source level is staged as ``tech_ground`` before ``run_sweep`` is started.
The sweep output is shown in the terminal and mirrored to ``out_<name>.txt`` in
the repository root.  The original ``tech_ground`` directory is restored when
the batch exits.  A crashed sweep is retried against the same environment until
it succeeds or the batch is stopped with Ctrl-C.

Run from anywhere with::

    python -m experiments.exp_008_beamng.runner
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVELS_DIR = (
    Path.home() / ".local/share/BeamNG/BeamNG.tech/current/levels"
)
SWEEP_MODULE = "experiments.exp_008_beamng.run_sweep"
DOUBLE_TAP_S = 2.0

# These paths match the level directories produced by the four BeamNG maps.
# The combined snow-and-slope archive has one additional ``tech_ground`` layer.
ENVIRONMENTS = {
    "asphalt": Path("tech_ground_asphalt"),
    "slope": Path("tech_ground_slope"),
    "snow": Path("tech_ground_snow"),
    "both": Path("tech_ground_final/tech_ground"),
}


def remove_path(path: Path) -> None:
    """Remove one known staging path, whether it is a directory or a link."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def activate_environment(source: Path, target: Path) -> None:
    """Copy *source* to *target* with the copy completed before target changes."""
    stage_root = Path(
        tempfile.mkdtemp(prefix=".runner-stage-", dir=target.parent)
    )
    staged_target = stage_root / target.name
    try:
        shutil.copytree(source, staged_target, symlinks=True)
        remove_path(target)
        staged_target.rename(target)
    finally:
        remove_path(stage_root)


class InterruptForwarder:
    """Forward Ctrl-C to the sweep and remember a double-tap batch abort."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.last_interrupt = 0.0
        self.abort_batch = False

    def handle_sigint(self, _signum: int, _frame: object) -> None:
        now = time.monotonic()
        if self.process is None or self.process.poll() is not None:
            self.abort_batch = True
            return
        if now - self.last_interrupt <= DOUBLE_TAP_S:
            self.abort_batch = True
        self.last_interrupt = now

        try:
            os.killpg(self.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def run_sweep(
    log_path: Path, interrupts: InterruptForwarder, *, append: bool = False
) -> int:
    """Run one sweep, streaming its combined stdout/stderr to a text log."""
    command = [sys.executable, "-m", SWEEP_MODULE]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("ab" if append else "wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        interrupts.process = process
        assert process.stdout is not None
        try:
            while chunk := process.stdout.read(64 * 1024):
                log_file.write(chunk)
                log_file.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            return process.wait()
        finally:
            interrupts.process = None
            process.stdout.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()


def validate_sources(levels_dir: Path, names: Sequence[str]) -> dict[str, Path]:
    """Resolve all requested environment directories before changing the level."""
    if not levels_dir.is_dir():
        raise FileNotFoundError(f"BeamNG levels directory not found: {levels_dir}")

    sources = {name: levels_dir / ENVIRONMENTS[name] for name in names}
    missing = [f"{name}: {path}" for name, path in sources.items() if not path.is_dir()]
    if missing:
        details = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing BeamNG environment directories:\n  {details}")
    return sources


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "environments",
        nargs="*",
        metavar="{" + ",".join(ENVIRONMENTS) + "}",
        help="environments to run in order (default: all four)",
    )
    parser.add_argument(
        "--levels-dir",
        type=Path,
        default=DEFAULT_LEVELS_DIR,
        help=f"BeamNG levels directory (default: {DEFAULT_LEVELS_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT,
        help=f"directory for out_<environment>.txt (default: {REPO_ROOT})",
    )
    args = parser.parse_args(argv)
    unknown = [name for name in args.environments if name not in ENVIRONMENTS]
    if unknown:
        parser.error(
            f"unknown environment {unknown[0]!r}; choose from "
            f"{', '.join(ENVIRONMENTS)}"
        )
    if not args.environments:
        args.environments = list(ENVIRONMENTS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    levels_dir = args.levels_dir.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    names = args.environments

    try:
        sources = validate_sources(levels_dir, names)
    except (FileNotFoundError, OSError) as exc:
        print(f"runner: {exc}", file=sys.stderr)
        return 2

    target = levels_dir / "tech_ground"
    backup_root = Path(
        tempfile.mkdtemp(prefix=".runner-backup-", dir=levels_dir)
    )
    backup = backup_root / target.name
    had_original = target.exists() or target.is_symlink()
    original_backed_up = False
    interrupts = InterruptForwarder()
    previous_sigint = signal.signal(signal.SIGINT, interrupts.handle_sigint)

    try:
        if had_original:
            target.rename(backup)
            original_backed_up = True

        for index, name in enumerate(names, start=1):
            if interrupts.abort_batch:
                break
            print("\n" + "=" * 88, flush=True)
            print(
                f"ENVIRONMENT {index}/{len(names)}: {name} "
                f"({sources[name]} -> {target})",
                flush=True,
            )
            print("=" * 88, flush=True)

            activate_environment(sources[name], target)
            if interrupts.abort_batch:
                break
            log_path = log_dir / f"out_{name}.txt"
            print(f"Logging sweep to {log_path}", flush=True)
            attempt = 1
            while not interrupts.abort_batch:
                return_code =  run_sweep(
                    log_path, interrupts, append=attempt > 1
                )
                if return_code == 0 or interrupts.abort_batch:
                    break

                attempt += 1
                retry_message = (
                    f"\n\nrunner: {name} sweep crashed with status "
                    f"{return_code}; retrying (attempt {attempt})\n"
                )
                print(retry_message, file=sys.stderr, end="", flush=True)
                with log_path.open("ab") as log_file:
                    log_file.write(retry_message.encode())

            if interrupts.abort_batch:
                print("runner: double Ctrl-C received; stopping the batch", flush=True)
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if original_backed_up:
            remove_path(target)
            try:
                backup.rename(target)
            except OSError:
                print(
                    f"runner: could not restore tech_ground; backup retained at {backup}",
                    file=sys.stderr,
                )
                raise
        elif not had_original:
            remove_path(target)
        remove_path(backup_root)

    return 130 if interrupts.abort_batch else 0


if __name__ == "__main__":
    raise SystemExit(main())
