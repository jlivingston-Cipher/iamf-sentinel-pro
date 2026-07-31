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

2. **A POSIX-shell gate.** Several unit tests stand in for the kernel / decoder
   binaries with `#!/bin/sh` stubs. Windows cannot execute those, so the tests
   that *run* a stub (as opposed to merely hashing one) are gated. See
   `needs_posix_shell`.
"""

from __future__ import annotations

import os
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


# --------------------------------------------------------------- platform

#: Tests that *execute* a `#!/bin/sh` stub binary. Windows has no shebang
#: handling, so such a stub raises WinError 193 rather than standing in for
#: the real tool. Tests that only hash a stub, or only check discovery
#: precedence, are not gated by this.
needs_posix_shell = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX '#!/bin/sh' stub binaries are not executable on Windows",
)

#: Discovery tests that rely on `shutil.which()` finding an EXTENSIONLESS
#: file. On Windows `which()` searches `name + ext` for each `%PATHEXT%`
#: entry, so an extensionless stub is invisible to it — a difference in the
#: platform's resolution rules, not in ours. The env-var branch of
#: `dsp.kernel_path()` is not gated: it is `os.path.isfile` + `os.access`,
#: both of which behave on Windows.
needs_windows_pathext = pytest.mark.skipif(
    os.name == "nt",
    reason="shutil.which() resolves via %PATHEXT% on Windows; an "
           "extensionless stub is not discoverable there",
)
