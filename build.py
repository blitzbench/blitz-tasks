#!/usr/bin/env python3
"""build.py - build BlitzBench tasks as static or shared libraries.

Usage:
    python3 build.py [--mode static|shared] [--task NAME ...] [--out-dir DIR]

Discovers tasks by scanning the repo root for directories containing a
`TASK.json` file. For each selected task:

- Rust tasks (have a `Cargo.toml`): driven via `cargo build --release`,
  producing the requested crate-type artifact in `target/release/`.
- C/C++ tasks (have a `CMakeLists.txt`): driven via `cmake -B build && cmake
  --build build --config Release` with `-DBLITZ_BUILD_MODE={STATIC|SHARED}`.

Resulting library files are copied into `<out-dir>/<mode>/<task>/`.

The framework crates `task-lib/rust`, `task-lib/c`, `task-lib/cpp`, and
`task-runtime/rust` are built automatically when any task depends on them
(cargo / cmake handles transitive deps).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUST_LIB_EXT = {
    "static": [".a"],
    "shared": [".so", ".dylib", ".dll"],
}
CMAKE_LIB_EXT = RUST_LIB_EXT


def log(msg: str) -> None:
    print(f"[build.py] {msg}", flush=True)


def discover_tasks() -> list[Path]:
    """Return absolute paths of every directory under ROOT that contains TASK.json."""
    tasks: list[Path] = []
    for entry in sorted(ROOT.iterdir()):
        if entry.is_dir() and (entry / "TASK.json").is_file():
            tasks.append(entry)
    return tasks


def build_rust_task(task_dir: Path, mode: str, out_dir: Path) -> None:
    log(f"rust:  {task_dir.name} ({mode})")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(task_dir),
        check=True,
    )
    target_dir = task_dir / "target" / "release"
    dest = out_dir / mode / task_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    copied = _copy_artifacts(target_dir, dest, RUST_LIB_EXT[mode])
    log(f"  → {copied} artifact(s) → {dest}")


def build_cmake_task(task_dir: Path, mode: str, out_dir: Path) -> None:
    log(f"cmake: {task_dir.name} ({mode})")
    build_dir = task_dir / "build" / mode
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_mode = "STATIC" if mode == "static" else "SHARED"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(task_dir),
            "-B",
            str(build_dir),
            f"-DBLITZ_BUILD_MODE={cmake_mode}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        check=True,
    )
    dest = out_dir / mode / task_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    copied = _copy_artifacts(build_dir, dest, CMAKE_LIB_EXT[mode])
    log(f"  → {copied} artifact(s) → {dest}")


def _copy_artifacts(src_dir: Path, dest: Path, suffixes: list[str]) -> int:
    count = 0
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.endswith(suf) for suf in suffixes):
            shutil.copy2(path, dest / path.name)
            count += 1
    return count


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["static", "shared"],
        default="static",
        help="library kind to build (default: static)",
    )
    p.add_argument(
        "--task",
        action="append",
        default=[],
        help="task name(s) to build; omit to build all discovered tasks",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "build"),
        help="output directory (default: ./build)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_tasks()
    if not discovered:
        log("no tasks found - nothing to do")
        return 0

    if args.task:
        wanted = set(args.task)
        selected = [t for t in discovered if t.name in wanted]
        missing = wanted - {t.name for t in selected}
        if missing:
            log(f"unknown task(s): {', '.join(sorted(missing))}")
            return 2
    else:
        selected = discovered

    log(f"building {len(selected)} task(s) in {args.mode} mode")
    for task in selected:
        try:
            if (task / "Cargo.toml").is_file():
                build_rust_task(task, args.mode, out_dir)
            elif (task / "CMakeLists.txt").is_file():
                build_cmake_task(task, args.mode, out_dir)
            else:
                log(f"skip {task.name}: no Cargo.toml or CMakeLists.txt")
        except subprocess.CalledProcessError as e:
            log(f"build failed for {task.name}: exit {e.returncode}")
            return e.returncode

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
