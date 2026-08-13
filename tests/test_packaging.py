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


# --- d177, items 27 + 28 ----------------------------------------------------
# PEP 639: setuptools REMOVES the `license = { text = ... }` table form on
# 2027-02-18. These two assert the shipped shape against the BUILT metadata,
# not against pyproject.toml, because the metadata is what reaches PyPI --
# which is the same lesson the 0.3.1 regression above already paid for.

def _d177_metadata():
    from importlib.metadata import PackageNotFoundError, metadata
    try:
        return metadata("iamf-sentinel-pro")
    except PackageNotFoundError:            # pragma: no cover
        pytest.skip("iamf-sentinel-pro is not installed; nothing to inspect")


def test_license_is_a_pep639_spdx_expression():
    md = _d177_metadata()
    assert md.get("License-Expression") == "Apache-2.0", (
        "License-Expression is %r -- setuptools>=77 emits the SPDX string, and "
        "the deprecated table form must not come back" % md.get("License-Expression")
    )
    assert not md.get("License"), (
        "the deprecated free-text License field is populated (%r) -- that is the "
        "table form returning" % md.get("License")
    )


def test_project_urls_name_this_repository():
    md = _d177_metadata()
    urls = {}
    for entry in md.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip()] = url.strip()
    assert urls.get("Repository") == "https://github.com/jlivingston-Cipher/iamf-sentinel-pro", (
        "Project-URL Repository is %r, expected %r -- repo = module = index"
        % (urls.get("Repository"), "https://github.com/jlivingston-Cipher/iamf-sentinel-pro")
    )
