"""intent_compare render-gated checks (S-343/S-345) — needs ear-render.

Skips cleanly when EAR is absent (CI legs carry no renderer; the with-EAR
figures are part of the doc-123 gate evidence). The fixture builder lives in
test_intent_compare_units; objects sit exactly ON 7.1.4 speakers so clean
dominance is unambiguous and the tampered prediction is decisively wrong.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))

from sentinel_pro import intent_compare as ic                     # noqa: E402
from test_intent_compare_units import (build_adm, sidecar,        # noqa: E402
                                       static_traj, DUR)

pytestmark = pytest.mark.skipif(
    ic.find_ear() is None,
    reason="ear-render not found (set $SENTINEL_EAR) — S-343/S-345 need it")


def _two_static_objects(tmp_path):
    return build_adm(tmp_path / "objs.wav",
                     objects=[{"name": "Obj1",
                               "blocks": [(0.0, DUR, 90.0, 0.0)]},
                              {"name": "Obj2",
                               "blocks": [(0.0, DUR, -90.0, 0.0)]}])


def _decode(az_label, n_slices=20):
    return {"layout": "7.1.4", "dominant": [az_label] * n_slices,
            "coverage": {az_label: 1.0}}


def test_clean_isolation_renders_pass(tmp_path):
    f = _two_static_objects(tmp_path)
    objs = [{"id": 1, "track": 1, "trajectory": static_traj(90.0),
             "expectActive": 0.98, "decode": _decode("M+090")},
            {"id": 2, "track": 2, "trajectory": static_traj(-90.0),
             "expectActive": 0.98, "decode": _decode("M-090")}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    r = ic.compare(sc, f)
    assert r.execution_error is None
    assert r.findings == []
    om = r.metrics["objects"]["object 1"]
    assert om["dominant_agreement"] == 1.0
    assert om["coverage_measured"].get("M+090", 0) > 0.9


def test_flipped_prediction_fires_s343_and_s345_exactly(tmp_path):
    """Sidecar trajectory matches the file (no S-342) but the decode
    prediction claims the mirror speaker — isolates the render checks."""
    f = _two_static_objects(tmp_path)
    objs = [{"id": 1, "track": 1, "trajectory": static_traj(90.0),
             "expectActive": 0.98, "decode": _decode("M-090")},
            {"id": 2, "track": 2, "trajectory": static_traj(-90.0),
             "expectActive": 0.98, "decode": _decode("M-090")}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    r = ic.compare(sc, f)
    assert r.execution_error is None
    got = sorted((x.check_id, x.where.split()[1]) for x in r.findings)
    assert got == [("S-343", "1"), ("S-345", "1")]
    j1 = json.dumps(r.to_dict())
    j2 = json.dumps(ic.compare(sc, f).to_dict())
    assert j1 == j2                                   # determinism with EAR
