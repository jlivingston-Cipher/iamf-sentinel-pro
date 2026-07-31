"""Packaging invariants.

Named regression (doc 101). `iamf-sentinel-pro 0.3.1` shipped to PyPI with
`pyproject.toml` bumped and `sentinel_pro/__init__.py` left at the previous
value, so the distribution and the module disagreed about their own version on
a published release — and PyPI never allows a version to be re-uploaded. The
bump was checked against the README (doc 96 §10.5) and not against the module
beside it. This test makes the pair a gate rather than a habit.
"""

from __future__ import annotations

import pytest

import sentinel_pro


def test_declared_version_matches_module_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        declared = version("iamf-sentinel-pro")
    except PackageNotFoundError:            # pragma: no cover
        pytest.skip("iamf-sentinel-pro is not installed; nothing to compare against")
    assert sentinel_pro.__version__ == declared, (
        "sentinel_pro.__version__ is %r but the installed distribution is %r — "
        "bump both, or neither" % (sentinel_pro.__version__, declared)
    )
