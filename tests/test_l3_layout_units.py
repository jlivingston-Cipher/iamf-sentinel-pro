"""_native_analysis_layout units (toolchain-free) — including the doc-76 fix:
stereo-bed programs now analyse as "2.0" (S-320/321/322 were silently skipped
for them before), while mono stays excluded by design (a 2.0 render of a mono
bed has identical L/R by construction — duplication is no misroute signature).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentinel_pro.l3_rendered import _native_analysis_layout   # noqa: E402

_OSS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "sentinel-oss"))
sys.path.insert(0, _OSS)
from sentinel.parser import parse_bytes                        # noqa: E402
from fixtures.build import build, channel_spec, scene_spec     # noqa: E402


def _model(spec):
    return parse_bytes(build(spec))


def _ae_ids(mod):
    return list(mod.audio_elements)


def test_stereo_bed_analyses_as_2_0():
    mod = _model(channel_spec("stereo"))
    assert _native_analysis_layout(mod, _ae_ids(mod)) == ("2.0", True)


def test_mono_bed_is_excluded_by_design():
    mod = _model(channel_spec("mono"))
    assert _native_analysis_layout(mod, _ae_ids(mod)) == (None, True)


def test_5_1_bed_native_layout():
    mod = _model(channel_spec("5.1"))
    assert _native_analysis_layout(mod, _ae_ids(mod)) == ("5.1", True)


def test_7_1_4_bed_native_layout():
    mod = _model(channel_spec("7.1.4"))
    assert _native_analysis_layout(mod, _ae_ids(mod)) == ("7.1.4", True)


def test_scene_gets_widest_view_and_not_beds_only():
    mod = _model(scene_spec(1))
    assert _native_analysis_layout(mod, _ae_ids(mod)) == ("7.1.4", False)


def test_unknown_element_ids_are_ignored():
    mod = _model(channel_spec("stereo"))
    layout, beds_only = _native_analysis_layout(mod, [999999])
    assert layout is None and beds_only is True
