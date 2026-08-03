#!/usr/bin/env python3
"""Validate per-task license metadata and generate license reports.

Scans every ``blitz-task_*/TASK.json`` (``license`` + ``libraries``) and
``third_party/infrastructure.json``, validates the metadata, and generates:

- ``THIRD_PARTY_NOTICES.md`` — attribution: every third-party library with
  homepage/source/version(s), the tasks that use it, and its verbatim license
  text.
- ``LICENSING.md`` — the per-task license summary table.

Checks (errors fail the run; warnings are printed but do not fail):

- every task declares ``license`` with a plausible SPDX id and a ``file``
  that exists under ``LICENSES/``;
- every library entry has all required fields, a valid ``role``, a plausible
  SPDX expression, and an existing ``license_file``;
- the library license is compatible with the declaring task's own license
  (e.g. a GPL library inside a source-available task is an error — the task
  itself must be GPL); unknown pairings are flagged for manual review;
- the same library ``name`` used by multiple tasks agrees on ``source``,
  ``license`` and ``license_file`` (versions may differ);
- no orphans: every ``third_party/licenses/*`` dir and every ``LICENSES/*``
  file is referenced from somewhere.

Usage:
    scripts/gen_third_party_notices.py            # validate + (re)generate
    scripts/gen_third_party_notices.py --check    # validate + fail if the
                                                  # generated files are stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LICENSES_DIR = REPO / "LICENSES"
THIRD_PARTY_LICENSES_DIR = REPO / "third_party" / "licenses"
INFRASTRUCTURE_JSON = REPO / "third_party" / "infrastructure.json"
NOTICES_MD = REPO / "THIRD_PARTY_NOTICES.md"
LICENSING_MD = REPO / "LICENSING.md"

DEFAULT_TASK_LICENSE = "LicenseRef-BlitzBench-Source-Available"

LIBRARY_REQUIRED_FIELDS = (
    "name",
    "homepage",
    "source",
    "version",
    "license",
    "license_file",
    "role",
    "usage",
)
LIBRARY_OPTIONAL_FIELDS = ("notes",)
LIBRARY_ROLES = ("workload", "support")

# --- License classification --------------------------------------------------

PERMISSIVE = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Zlib",
    "ISC",
    "BSL-1.0",
    "Unlicense",
    "CC0-1.0",
    "0BSD",
}
# Vendor runtimes redistributed unmodified under their own terms and loaded at
# runtime rather than built into a task. They are not copyleft and impose no
# condition on the task's own license, so the copyleft rule below does not apply
# to them -- but each still needs its licence vendored under third_party/licenses/
# and its redistribution terms honoured (see the library's `notes`).
PROPRIETARY_REDISTRIBUTABLE = {
    "LicenseRef-NVIDIA-CUDA-EULA",
}

WEAK_COPYLEFT_PREFIXES = ("LGPL-", "MPL-")
STRONG_COPYLEFT_PREFIXES = ("GPL-",)
NETWORK_COPYLEFT_PREFIXES = ("AGPL-", "SSPL-")

SPDX_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+\-]*$")

# Rank order: higher = more restrictive.
CLASS_PERMISSIVE = 0
CLASS_WEAK = 1
CLASS_STRONG = 2
CLASS_NETWORK = 3
CLASS_CUSTOM = 4
CLASS_UNKNOWN = 5


def classify_single(spdx_id: str) -> int:
    if spdx_id.startswith("LicenseRef-"):
        return CLASS_CUSTOM
    if spdx_id in PERMISSIVE:
        return CLASS_PERMISSIVE
    if spdx_id.startswith(WEAK_COPYLEFT_PREFIXES):
        return CLASS_WEAK
    if spdx_id.startswith(STRONG_COPYLEFT_PREFIXES):
        return CLASS_STRONG
    if spdx_id.startswith(NETWORK_COPYLEFT_PREFIXES):
        return CLASS_NETWORK
    return CLASS_UNKNOWN


def spdx_ids(expression: str) -> list[str]:
    """Split an SPDX expression into its license ids (ignoring operators)."""
    ids = []
    for token in expression.replace("(", " ").replace(")", " ").split():
        if token in ("AND", "OR", "WITH"):
            continue
        ids.append(token)
    return ids


def plausible_spdx(expression: str) -> bool:
    ids = spdx_ids(expression)
    return bool(ids) and all(SPDX_TOKEN_RE.match(i) for i in ids)


def classify_expression(expression: str) -> int:
    """Classify a full SPDX expression.

    ``OR`` lets the consumer pick, so the effective class is the least
    restrictive alternative; parts joined by ``AND`` all apply, so there the
    most restrictive part wins. We approximate: if the expression contains
    ``OR`` take the minimum class, otherwise the maximum.
    """
    classes = [classify_single(i) for i in spdx_ids(expression)]
    if not classes:
        return CLASS_UNKNOWN
    return min(classes) if " OR " in expression else max(classes)


def check_compatibility(task_class: int, lib_class: int) -> str:
    """Return 'ok', 'review', or 'error' for a task/library license pairing."""
    if lib_class == CLASS_PERMISSIVE:
        return "ok"
    if task_class in (CLASS_CUSTOM, CLASS_PERMISSIVE):
        # Proprietary/source-available (and permissive) tasks must not embed
        # copyleft code; weak copyleft depends on linking mode -> review.
        return "review" if lib_class == CLASS_WEAK else "error"
    if task_class == CLASS_WEAK:
        return "ok" if lib_class == CLASS_WEAK else "error"
    if task_class == CLASS_STRONG:
        if lib_class in (CLASS_WEAK, CLASS_STRONG):
            return "ok"
        return "review" if lib_class == CLASS_NETWORK else "error"
    return "review"


# --- Loading -----------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def load_tasks(report: Report) -> list[dict]:
    tasks = []
    for task_json in sorted(REPO.glob("blitz-task_*/TASK.json")):
        rel = task_json.relative_to(REPO)
        try:
            data = json.loads(task_json.read_text())
        except json.JSONDecodeError as e:
            report.error(f"{rel}: invalid JSON: {e}")
            continue
        data["_path"] = rel
        data["_dir"] = task_json.parent.name
        tasks.append(data)
    return tasks


def validate_task_license(task: dict, report: Report) -> None:
    rel = task["_path"]
    lic = task.get("license")
    if not isinstance(lic, dict):
        report.error(f"{rel}: missing required top-level 'license' object")
        return
    spdx = lic.get("spdx", "")
    file = lic.get("file", "")
    if not plausible_spdx(spdx):
        report.error(f"{rel}: license.spdx {spdx!r} is not a plausible SPDX id")
    if not file.startswith("LICENSES/"):
        report.error(f"{rel}: license.file {file!r} must point into LICENSES/")
    elif not (REPO / file).is_file():
        report.error(f"{rel}: license.file {file!r} does not exist")


def validate_library(task: dict, lib: dict, report: Report) -> None:
    rel = task["_path"]
    name = lib.get("name", "<unnamed>")
    for field in LIBRARY_REQUIRED_FIELDS:
        if not lib.get(field):
            report.error(f"{rel}: library {name!r}: missing required field {field!r}")
    unknown = set(lib) - set(LIBRARY_REQUIRED_FIELDS) - set(LIBRARY_OPTIONAL_FIELDS)
    if unknown:
        report.error(f"{rel}: library {name!r}: unknown fields {sorted(unknown)}")
    if lib.get("role") not in LIBRARY_ROLES:
        report.error(
            f"{rel}: library {name!r}: role {lib.get('role')!r} not in {LIBRARY_ROLES}"
        )
    lib_license = lib.get("license", "")
    if lib_license and not plausible_spdx(lib_license):
        report.error(
            f"{rel}: library {name!r}: license {lib_license!r} is not a plausible SPDX expression"
        )
    license_file = lib.get("license_file", "")
    if license_file:
        if not license_file.startswith("third_party/licenses/"):
            report.error(
                f"{rel}: library {name!r}: license_file {license_file!r} must point into third_party/licenses/"
            )
        elif not (REPO / license_file).is_file():
            report.error(
                f"{rel}: library {name!r}: license_file {license_file!r} does not exist"
            )
    task_spdx = task.get("license", {}).get("spdx", DEFAULT_TASK_LICENSE)
    if lib_license and plausible_spdx(lib_license) and plausible_spdx(task_spdx):
        if all(i in PROPRIETARY_REDISTRIBUTABLE for i in spdx_ids(lib_license)):
            verdict = (
                "ok"
                if classify_expression(task_spdx) in (CLASS_CUSTOM, CLASS_PERMISSIVE)
                else "review"
            )
        else:
            verdict = check_compatibility(
                classify_expression(task_spdx), classify_expression(lib_license)
            )
        if verdict == "error":
            report.error(
                f"{rel}: library {name!r} ({lib_license}) is not license-compatible "
                f"with the task's own license ({task_spdx})"
            )
        elif verdict == "review":
            report.warn(
                f"{rel}: library {name!r} ({lib_license}) with task license "
                f"({task_spdx}) needs manual license review"
            )
    if "Apache-2.0" in spdx_ids(lib_license) and license_file:
        notice = (REPO / license_file).parent / "NOTICE"
        if not notice.is_file():
            report.warn(
                f"{rel}: library {name!r} is Apache-2.0 licensed but no vendored "
                f"NOTICE file exists next to {license_file!r} (required if upstream ships one)"
            )


def validate_cross_task(tasks: list[dict], report: Report) -> None:
    seen: dict[str, tuple] = {}
    for task in tasks:
        for lib in task.get("libraries", []):
            name = lib.get("name")
            if not name:
                continue
            key = (lib.get("source"), lib.get("license"), lib.get("license_file"))
            if name in seen and seen[name][0] != key:
                report.error(
                    f"library {name!r} drifted between tasks: "
                    f"{seen[name][1]} declares {seen[name][0]}, "
                    f"{task['_path']} declares {key} "
                    "(source/license/license_file must agree across tasks)"
                )
            seen.setdefault(name, (key, task["_path"]))


def load_infrastructure(report: Report) -> list[dict]:
    if not INFRASTRUCTURE_JSON.is_file():
        report.error(f"{INFRASTRUCTURE_JSON.relative_to(REPO)} is missing")
        return []
    deps = json.loads(INFRASTRUCTURE_JSON.read_text()).get("dependencies", [])
    for dep in deps:
        name = dep.get("name", "<unnamed>")
        for field in ("name", "homepage", "source", "version", "license", "license_file", "used_for"):
            if not dep.get(field):
                report.error(f"infrastructure.json: {name!r}: missing field {field!r}")
        license_file = dep.get("license_file", "")
        if license_file and not (REPO / license_file).is_file():
            report.error(f"infrastructure.json: {name!r}: license_file {license_file!r} does not exist")
    return deps


def validate_orphans(tasks: list[dict], infra: list[dict], report: Report) -> None:
    referenced_license_files = {
        t.get("license", {}).get("file") for t in tasks if isinstance(t.get("license"), dict)
    }
    for path in sorted(LICENSES_DIR.glob("*")):
        rel = str(path.relative_to(REPO))
        if rel not in referenced_license_files:
            report.error(f"{rel}: orphaned — no TASK.json license.file references it")

    referenced_dirs = set()
    for task in tasks:
        for lib in task.get("libraries", []):
            f = lib.get("license_file", "")
            if f.startswith("third_party/licenses/"):
                referenced_dirs.add(f.split("/")[2])
    for dep in infra:
        f = dep.get("license_file", "")
        if f.startswith("third_party/licenses/"):
            referenced_dirs.add(f.split("/")[2])
    if THIRD_PARTY_LICENSES_DIR.is_dir():
        for d in sorted(THIRD_PARTY_LICENSES_DIR.iterdir()):
            if d.is_dir() and d.name not in referenced_dirs:
                report.error(
                    f"third_party/licenses/{d.name}/: orphaned — nothing references it"
                )


# --- Generation --------------------------------------------------------------

GENERATED_HEADER = (
    "<!-- Generated by scripts/gen_third_party_notices.py — do not edit by hand. -->\n"
)


def gen_notices(tasks: list[dict], infra: list[dict]) -> str:
    out = [GENERATED_HEADER]
    out.append("# Third-Party Notices\n")
    out.append(
        "This file lists the third-party software used by BlitzBench benchmark\n"
        "tasks (declared per task in `TASK.json` under `libraries`) and by the\n"
        "build infrastructure (`third_party/infrastructure.json`), together with\n"
        "their license texts.\n"
    )

    by_name: dict[str, dict] = {}
    for task in tasks:
        for lib in task.get("libraries", []):
            entry = by_name.setdefault(
                lib["name"],
                {"lib": lib, "versions": set(), "used_by": []},
            )
            entry["versions"].add(lib.get("version", "?"))
            entry["used_by"].append((task["_dir"], lib.get("usage", "")))

    out.append("\n## Task libraries\n")
    if not by_name:
        out.append("\n_No task currently declares third-party libraries._\n")
    for name in sorted(by_name):
        entry = by_name[name]
        lib = entry["lib"]
        out.append(f"\n### {name}\n")
        out.append(f"\n- Homepage: {lib['homepage']}")
        out.append(f"\n- Source: {lib['source']}")
        out.append(f"\n- Version(s) in use: {', '.join(sorted(entry['versions']))}")
        out.append(f"\n- License: `{lib['license']}` ([text]({lib['license_file']}))")
        if lib.get("notes"):
            out.append(f"\n- Notes: {lib['notes']}")
        out.append("\n- Used by:")
        for task_dir, usage in sorted(entry["used_by"]):
            out.append(f"\n  - `{task_dir}` — {usage}")
        out.append("\n\n<details><summary>License text</summary>\n\n```")
        out.append("\n" + (REPO / lib["license_file"]).read_text().rstrip("\n"))
        out.append("\n```\n\n</details>\n")

    out.append("\n## Build infrastructure\n")
    for dep in sorted(infra, key=lambda d: d["name"].lower()):
        out.append(f"\n### {dep['name']}\n")
        out.append(f"\n- Homepage: {dep['homepage']}")
        out.append(f"\n- Source: {dep['source']}")
        out.append(f"\n- Version: {dep['version']}")
        out.append(f"\n- License: `{dep['license']}` ([text]({dep['license_file']}))")
        out.append(f"\n- Used for: {dep['used_for']}")
        out.append("\n\n<details><summary>License text</summary>\n\n```")
        out.append("\n" + (REPO / dep["license_file"]).read_text().rstrip("\n"))
        out.append("\n```\n\n</details>\n")

    return "".join(out)


def gen_licensing(tasks: list[dict]) -> str:
    out = [GENERATED_HEADER]
    out.append("# Licensing overview\n")
    out.append(
        "\nEach task declares its own license in its `TASK.json` (`license` key);\n"
        "the repository default is the BlitzBench Source-Available License. See\n"
        "`LICENSE` for the licensing model and `THIRD_PARTY_NOTICES.md` for\n"
        "third-party attributions.\n"
    )
    out.append("\n| Task | License | Third-party libraries |\n|---|---|---|\n")
    for task in sorted(tasks, key=lambda t: t["_dir"]):
        lic = task.get("license", {})
        spdx = lic.get("spdx", "?")
        file = lic.get("file", "")
        libs = ", ".join(
            f"{l['name']} (`{l['license']}`)" for l in task.get("libraries", [])
        ) or "—"
        out.append(f"| `{task['_dir']}` | [`{spdx}`]({file}) | {libs} |\n")
    return "".join(out)


# --- Main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; fail if the generated files are missing or stale",
    )
    args = parser.parse_args()

    report = Report()
    tasks = load_tasks(report)
    for task in tasks:
        validate_task_license(task, report)
        libs = task.get("libraries", [])
        if not isinstance(libs, list):
            report.error(f"{task['_path']}: 'libraries' must be an array")
            continue
        for lib in libs:
            validate_library(task, lib, report)
    validate_cross_task(tasks, report)
    infra = load_infrastructure(report)
    validate_orphans(tasks, infra, report)

    for msg in report.warnings:
        print(f"WARNING: {msg}", file=sys.stderr)
    if report.errors:
        for msg in report.errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        print(f"\n{len(report.errors)} error(s).", file=sys.stderr)
        return 1

    notices = gen_notices(tasks, infra)
    licensing = gen_licensing(tasks)
    if args.check:
        stale = []
        if not NOTICES_MD.is_file() or NOTICES_MD.read_text() != notices:
            stale.append(NOTICES_MD.name)
        if not LICENSING_MD.is_file() or LICENSING_MD.read_text() != licensing:
            stale.append(LICENSING_MD.name)
        if stale:
            print(
                f"ERROR: {', '.join(stale)} stale — rerun scripts/gen_third_party_notices.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {len(tasks)} tasks validated, generated files up to date.")
    else:
        NOTICES_MD.write_text(notices)
        LICENSING_MD.write_text(licensing)
        print(f"OK: {len(tasks)} tasks validated; wrote {NOTICES_MD.name}, {LICENSING_MD.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
