"""The collection-completeness guard (conftest §3, doc 127 §4 / doc 128).

The defect this pins: when the iamf-sentinel SOURCE checkout is absent,
`test_l3_layout_units.py` and `test_oracle_units.py` skip at MODULE level, 17
tests are never collected, and the suite exits 0 — 62 passed / 20 skipped
instead of 79 / 18, indistinguishable from a healthy run to anything that reads
the exit code. `ci.yml` supplies the sibling checkout; the guard is what makes
that supply *load-bearing* rather than assumed.

Both directions are pinned here, because a guard that cannot be shown to stay
out of the way is as untrustworthy as one that cannot be shown to fire.

Method note. These two tests run pytest in a subprocess against a synthetic
repo root — `tmp/pro/tests/conftest.py` (a copy of the real one) plus a trivial
probe module — rather than against this repo. That is the only way to reach the
"no core source" state deterministically on every machine: `conftest._REPO_ROOT`
is derived from the conftest's own location, so a synthetic root has synthetic
siblings, and CI's real sibling checkout cannot leak in and turn the test
vacuous. `$IAMF_SENTINEL_SRC` is cleared in the child for the same reason.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (INCOMPLETE_COLLECTION_ERROR,  # noqa: E402
                      REQUIRE_FULL_COLLECTION_ENV, require_full_collection)

_REAL_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

_PROBE = "def test_probe():\n    assert True\n"


def _synthetic_repo(tmp_path: Path) -> Path:
    """A repo root whose parent provably has no core checkout beside it."""
    root = tmp_path / "pro"
    (root / "tests").mkdir(parents=True)
    shutil.copy2(_REAL_CONFTEST, root / "tests" / "conftest.py")
    (root / "tests" / "test_probe.py").write_text(_PROBE, encoding="utf-8")
    # Belt and braces: assert the precondition instead of assuming it.
    for sibling in ("sentinel-oss", "iamf-sentinel"):
        assert not (tmp_path / sibling).exists()
    return root


def _run(root: Path, *, guard: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["IAMF_SENTINEL_SRC"] = ""          # never inherit a real one
    env.pop(REQUIRE_FULL_COLLECTION_ENV, None)
    if guard is not None:
        env[REQUIRE_FULL_COLLECTION_ENV] = guard
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
    )


def test_guard_refuses_a_collapsed_collection(tmp_path):
    """With the guard set and no core source, the run FAILS and says why."""
    proc = _run(_synthetic_repo(tmp_path), guard="1")
    out = proc.stdout + proc.stderr

    assert proc.returncode != 0, (
        "the collapsed-collection run exited 0 under "
        f"{REQUIRE_FULL_COLLECTION_ENV}=1 — the guard did not fire:\n{out}"
    )
    # The whole point is that this can never read as a green run.
    assert " passed" not in out, f"a 'passed' summary survived the guard:\n{out}"
    # The operator must be told what to do, not merely that something is wrong.
    assert REQUIRE_FULL_COLLECTION_ENV in out
    assert "IAMF_SENTINEL_SRC" in out
    assert "actions/checkout" in out


@pytest.mark.parametrize("guard", [None, "0"])
def test_guard_is_inert_unless_asked(tmp_path, guard):
    """Unset or falsey, the standalone-clone experience is unchanged: green."""
    proc = _run(_synthetic_repo(tmp_path), guard=guard)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"guard={guard!r} broke an unguarded run:\n{out}"
    assert "1 passed" in out, out
    assert INCOMPLETE_COLLECTION_ERROR.splitlines()[0] not in out


def test_require_full_collection_reads_the_environment():
    """The policy itself, without a subprocess (conftest §3)."""
    for falsey in ("", "0", "false", "FALSE", "no", "off", "  0  "):
        assert require_full_collection({REQUIRE_FULL_COLLECTION_ENV: falsey}) is False
    for truthy in ("1", "true", "yes", "on", "anything"):
        assert require_full_collection({REQUIRE_FULL_COLLECTION_ENV: truthy}) is True
    assert require_full_collection({}) is False
