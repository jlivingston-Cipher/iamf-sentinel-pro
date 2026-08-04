"""Shared test support for iamf-sentinel-pro (doc 97, item 24).

Two things the suite needs that are not properties of this repo:

1. **The core's fixture builders.** `fixtures/build.py` and `fixtures/mp4wrap.py`
   ship in the `iamf-sentinel` *source* tree, not in the installed wheel — the
   wheel carries `sentinel/`, not `fixtures/`. Two modules here build real IAMF
   bytes with them. Before doc 97 they hard-coded `../../sentinel-oss`, the
   internal monorepo directory name, so a standalone clone of this repo failed
   at *collection* (`ModuleNotFoundError: fixtures.build`) rather than running.
   Resolution order is now `$IAMF_SENTINEL_SRC`, then a sibling `sentinel-oss/`
   (internal layout), then a sibling `iamf-sentinel/` (public clone layout);
   absent all three, the dependent modules skip cleanly.

2. **Cross-platform tool stubs.** Several unit tests stand in for the kernel /
   decoder binaries with fake executables. Before item 25 those were
   `#!/bin/sh` scripts and the five tests that ran or discovered one were
   skipped on Windows. `write_exec_stub` now emits a `.cmd` batch file there
   instead — executable by `subprocess` and, because it carries an extension
   in `%PATHEXT%`, discoverable by `shutil.which()` — so the same five tests
   run on every CI leg. Callers must treat the RETURNED path as the stub's
   identity: on Windows it is `name + ".cmd"`, not `name`.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_oss_source() -> Path | None:
    """Locate an `iamf-sentinel` SOURCE checkout that carries `fixtures/`."""
    candidates: list[Path] = []
    env = os.environ.get("IAMF_SENTINEL_SRC")
    if env:
        candidates.append(Path(env))
    candidates += [
        _REPO_ROOT.parent / "sentinel-oss",     # internal tree layout
        _REPO_ROOT.parent / "iamf-sentinel",    # public side-by-side clones
    ]
    for c in candidates:
        if (c / "fixtures" / "build.py").is_file():
            return c.resolve()
    return None


OSS_SRC: Path | None = _find_oss_source()

NO_OSS_SRC_REASON = (
    "iamf-sentinel source checkout not found — clone it beside this repo or "
    "set $IAMF_SENTINEL_SRC (the fixture builders live in the core's source "
    "tree, not in the installed wheel)"
)

for _p in ([str(_REPO_ROOT)] + ([str(OSS_SRC)] if OSS_SRC else [])):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------- platform

IS_WINDOWS = os.name == "nt"


def write_exec_stub(directory: Path, name: str, *, stdout: str | None = None,
                    stderr: str | None = None, exit_code: int = 0) -> Path:
    """Create a fake tool binary that runs AND discovers on POSIX and Windows.

    POSIX gets an extensionless ``#!/bin/sh`` script (chmod +x); Windows gets
    ``name + ".cmd"`` — batch files execute directly under ``subprocess`` and
    the extension is what ``shutil.which()`` needs there (``%PATHEXT%``
    resolution; an extensionless file is invisible to it). Item 25's whole
    point: the platform difference lives HERE, not in per-test skip gates.

    The ``.cmd`` body is written with explicit CRLF line endings (cmd.exe's
    native form) and payloads must stay within characters cmd.exe echoes
    literally — no parens, carets, percent signs, angle brackets or pipes.
    ``>&2 echo`` (redirect first) avoids cmd's trailing-space-before-redirect
    trap. Callers compare paths with ``os.path.samefile`` rather than string
    equality: ``which()`` may report a different extension case than we wrote.
    """
    if IS_WINDOWS:
        stub = directory / (name + ".cmd")
        lines = ["@echo off"]
        if stdout is not None:
            lines.append(f"echo {stdout}")
        if stderr is not None:
            lines.append(f">&2 echo {stderr}")
        lines += [f"exit /b {exit_code}", ""]
        stub.write_text("\r\n".join(lines), encoding="ascii", newline="")
    else:
        stub = directory / name
        lines = ["#!/bin/sh"]
        if stdout is not None:
            lines.append(f"echo '{stdout}'")
        if stderr is not None:
            lines.append(f"echo '{stderr}' >&2")
        lines += [f"exit {exit_code}", ""]
        stub.write_text("\n".join(lines), encoding="utf-8", newline="")
        stub.chmod(stub.stat().st_mode
                   | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub
