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

3. **A guard against silently collecting less.** Point 1's clean skip is right
   for a contributor who cloned this repo alone, and wrong for CI. When
   `OSS_SRC` is None the two dependent modules skip at *module* level, so 17
   tests collapse into 2 skip lines and the suite still exits 0 — 62 passed /
   20 skipped instead of 79 / 18, all green (doc 127 §4, measured doc 128).
   `ci.yml` supplies the sibling checkout, but nothing enforced that it had:
   remove, rename or silently fail that step and every leg goes on reporting
   success while a third of these units never run. Setting
   `IAMF_SENTINEL_REQUIRE_FULL_COLLECTION=1` — which `ci.yml` now does — turns
   the absent checkout into a hard error. The check lives in
   `pytest_configure`, which runs *before* any test module is imported, so it
   covers every future module that depends on `OSS_SRC` and not merely the two
   that exist today.

   Do not replace this with a count assertion. Counts are not a reliable
   statement about the environment here: `test_packaging.py` resolves the
   distribution through `importlib.metadata`, which finds a stale
   `iamf_sentinel_pro.egg-info/` left in the working tree by any earlier build
   and reports 80/17 with nothing installed at all (measured doc 128).
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

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


# ------------------------------------------------- collection-completeness

REQUIRE_FULL_COLLECTION_ENV = "IAMF_SENTINEL_REQUIRE_FULL_COLLECTION"

_FALSEY = {"", "0", "false", "no", "off"}


def require_full_collection(environ=None) -> bool:
    """True when the caller has demanded a fully-collected run (see §3 above).

    Split out from the hook so the policy is testable without a subprocess.
    """
    env = os.environ if environ is None else environ
    return env.get(REQUIRE_FULL_COLLECTION_ENV, "").strip().lower() not in _FALSEY


INCOMPLETE_COLLECTION_ERROR = (
    "{var}=1 demands a fully-collected run, but the iamf-sentinel SOURCE "
    "checkout was not found.\n"
    "  Effect: tests/test_l3_layout_units.py and tests/test_oracle_units.py "
    "skip at MODULE level, so 17 tests are never collected and the suite "
    "would still exit 0.\n"
    "  Fix: clone iamf-sentinel beside this repo (CI does this with a second "
    "actions/checkout step), or set $IAMF_SENTINEL_SRC to a source checkout "
    "that carries fixtures/build.py.\n"
    "  If you are a contributor running this repo standalone, unset {var} "
    "and the two modules will skip cleanly as designed."
).format(var=REQUIRE_FULL_COLLECTION_ENV)


def pytest_configure(config):                     # noqa: ARG001
    """Refuse the collapsed-collection run when the caller asked for a full one.

    Deliberately a `pytest_configure` hook and not a module-level raise: this
    runs before any test module is imported, so the two `allow_module_level`
    skips below are unreachable in the guarded state and a NEW module that
    forgets the guard is covered by construction.
    """
    if require_full_collection() and OSS_SRC is None:
        raise pytest.UsageError(INCOMPLETE_COLLECTION_ERROR)


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
