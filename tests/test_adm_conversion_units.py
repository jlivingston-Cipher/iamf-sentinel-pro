"""S-333 / S-334 (IA-0076): per-track ADM conversion fidelity, toolchain-free.

The fixtures are made here: a BW64 ADM master written track by track, and an
LPCM IAMF built from it under two rules -- the standard's (every track at its
gain, a Cartesian position through BS.2127-0 §10.1.2) and iamf-tools' importer
rule (gain / max(|p|, 0.5), azimuth atan2 of the cube coordinates, the LFE
panned into the scene at its position). The first must pass, the second must
fire both checks, with the values the converter produces measured to a
hundredth of a dB (s318 ia0057 §5.3: -3.0103 dB at the corners, -4.7712 dB for
the LFE).
"""

from __future__ import annotations

import math
import os
import struct
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.build import (ChannelLayerSpec, ElementSpec, Writer, build,  # noqa: E402
                            obu, scene_spec)
from sentinel_pro.adm_compare import compare                          # noqa: E402
from sentinel.findings import Severity                                # noqa: E402
from sentinel_pro.adm_conversion import (angle_between,              # noqa: E402
                                         conversion_fidelity,
                                         inside_angle_range,
                                         point_cart_to_polar,
                                         relative_angle)

SR, N = 48000, 48000


# ------------------------------------------------------------ §10.1.2 values

# Expected values from EAR 2.1.0 (ear.core.objectbased.conversion.
# point_cart_to_polar), the BS.2127 reference implementation, run 2026-11-11
# as an oracle only (it agreed with the implementation under test to 1e-6 on
# these and four more points; -0.0 written as 0.0). The implementation under
# test is written from the Recommendation's text, not from EAR's source.
EAR_CART_TO_POLAR = [
    ((0, 1, 0), (0.0, 0.0, 1.0)),
    ((1, 1, 0), (-30.0, 0.0, 1.0)),
    ((-1, 1, 0), (30.0, 0.0, 1.0)),
    ((1, -1, 0), (-110.0, 0.0, 1.0)),
    ((-1, -1, 0), (110.0, 0.0, 1.0)),
    ((1, 0, 0), (-70.0, 0.0, 1.0)),
    ((0.5, 1, 0), (-15.0, 0.0, 1.0)),
    ((0.3, 0.8, 0.2), (-11.949113, 9.357496, 0.8)),
    ((-0.7, -0.2, 0.6), (80.841944, 27.06753, 0.7)),
    ((-1, 1, 1), (30.0, 30.0, 1.0)),
    ((-1, 1, -1), (30.0, -30.0, 1.0)),
    ((0, 0, 1), (0.0, 90.0, 1.0)),
    ((0.2, -0.9, -0.4), (-154.151928, -15.974993, 0.9)),
    ((0, 0.5, 0), (0.0, 0.0, 0.5)),
]


@pytest.mark.parametrize("xyz,expected", EAR_CART_TO_POLAR)
def test_point_cart_to_polar_matches_the_reference(xyz, expected):
    az, el, d = point_cart_to_polar(*xyz)
    assert az == pytest.approx(expected[0], abs=1e-5)
    assert el == pytest.approx(expected[1], abs=1e-5)
    assert d == pytest.approx(expected[2], abs=1e-6)


def test_the_rear_is_plus_or_minus_180():
    az, el, d = point_cart_to_polar(0, -1, 0)
    assert abs(abs(az) - 180.0) < 1e-9 and el == 0 and d == pytest.approx(1)


def test_angle_helpers_follow_section_6():
    assert relative_angle(-30, 0) == 0 and relative_angle(0, -30) == 330
    assert relative_angle(-180, 180) == -180      # the smallest y + 360n >= x
    # BS.2127-0 Table 2
    assert inside_angle_range(0, -90, 90) and not inside_angle_range(120, -90, 90)
    assert inside_angle_range(95, -90, 90, 5) and not inside_angle_range(96, -90, 90, 5)
    assert inside_angle_range(-120, 90, -90) and not inside_angle_range(0, 90, -90)
    assert inside_angle_range(-180, 180, 180) and inside_angle_range(33, -180, 180)
    assert angle_between(45, 0, 30, 0) == pytest.approx(15.0)


# --------------------------------------------------------------- fixtures

def _cart_polar_az(x, y, z):        # iamf-tools' panner (panner.cc): plain atan2
    return (-math.degrees(math.atan2(x, y)),
            math.degrees(math.atan2(z, math.hypot(x, y))))


TRACKS = [
    # name, type, position (("cart", x, y, z) | ("polar", az, el, d)), block gain, lfe, blocks
    ("Obj_Front",  "Objects", ("cart", 0.0, 1.0, 0.0), 1.0, False, 1),
    ("Obj_Corner", "Objects", ("cart", 1.0, 1.0, 0.0), 1.0, False, 1),
    ("Obj_Polar",  "Objects", ("polar", 110.0, 0.0, 1.0), 0.5, False, 1),
    ("RoomCentricLFE", "DirectSpeakers", ("cart", -1.0, 1.0, -1.0), 1.0, True, 1),
    ("Obj_Moving", "Objects", ("cart", -1.0, -1.0, 0.0), 1.0, False, 2),
]


def _sources() -> np.ndarray:
    t = np.arange(N) / SR
    rng = np.random.Generator(np.random.PCG64(76))
    cols = [0.2 * np.sin(2 * np.pi * (220 + 110 * k) * t) + 0.05 * rng.standard_normal(N)
            for k in range(len(TRACKS))]
    return np.stack(cols, axis=1)


def _axml() -> bytes:
    objs, packs, chans, uids = [], [], [], []
    for k, (name, td, pos, gain, lfe, nblocks) in enumerate(TRACKS, start=1):
        ac, ap, ao, uid = f"AC_{0x1000 + k:04x}0001", f"AP_{0x1000 + k:04x}0001", f"AO_{0x1000 + k:04x}", f"ATU_{k:08x}"
        tlabel = "0001" if td == "DirectSpeakers" else "0003"
        blocks = []
        for b in range(nblocks):
            if pos[0] == "cart":
                p = (f'<position coordinate="X">{pos[1] + 0.1 * b}</position>'
                     f'<position coordinate="Y">{pos[2]}</position>'
                     f'<position coordinate="Z">{pos[3]}</position><cartesian>1</cartesian>')
            else:
                p = (f'<position coordinate="azimuth">{pos[1]}</position>'
                     f'<position coordinate="elevation">{pos[2]}</position>'
                     f'<position coordinate="distance">{pos[3]}</position>')
            blocks.append(f'<audioBlockFormat audioBlockFormatID="AB_{0x1000 + k:04x}0001_{b + 1:08x}">'
                          f'{p}<gain>{gain}</gain></audioBlockFormat>')
        freq = '<frequency typeDefinition="lowPass">120</frequency>' if lfe else ""
        chans.append(f'<audioChannelFormat audioChannelFormatID="{ac}" audioChannelFormatName="{name}" '
                     f'typeLabel="{tlabel}" typeDefinition="{td}">{freq}{"".join(blocks)}</audioChannelFormat>')
        packs.append(f'<audioPackFormat audioPackFormatID="{ap}" typeDefinition="{td}">'
                     f'<audioChannelFormatIDRef>{ac}</audioChannelFormatIDRef></audioPackFormat>')
        objs.append(f'<audioObject audioObjectID="{ao}" audioObjectName="{name}">'
                    f'<audioPackFormatIDRef>{ap}</audioPackFormatIDRef>'
                    f'<audioTrackUIDRef>{uid}</audioTrackUIDRef></audioObject>')
        uids.append(f'<audioTrackUID UID="{uid}"><audioChannelFormatIDRef>{ac}</audioChannelFormatIDRef>'
                    f'<audioPackFormatIDRef>{ap}</audioPackFormatIDRef></audioTrackUID>')
    return ('<?xml version="1.0" encoding="UTF-8"?><ebuCoreMain xmlns="urn:ebu:metadata-schema:ebucore">'
            '<coreMetadata><format><audioFormatExtended version="ITU-R_BS.2076-2">'
            + "".join(objs + packs + chans + uids)
            + "</audioFormatExtended></format></coreMetadata></ebuCoreMain>").encode()


def _pcm24(x: np.ndarray) -> bytes:
    v = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int64).reshape(-1)
    v &= 0xFFFFFF
    b = np.empty((v.size, 3), dtype=np.uint8)
    b[:, 0], b[:, 1], b[:, 2] = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
    return b.tobytes()


def write_adm(path, src: np.ndarray) -> None:
    ch = src.shape[1]
    chna = struct.pack("<HH", ch, ch)
    for k in range(1, ch + 1):
        chna += struct.pack("<H", k) + f"ATU_{k:08x}".encode() + \
            f"AC_{0x1000 + k:04x}0001".encode().ljust(14, b"\0") + \
            f"AP_{0x1000 + k:04x}0001".encode().ljust(11, b"\0") + b"\0"
    axml = _axml()
    data = _pcm24(src)
    fmt = struct.pack("<HHIIHH", 1, ch, SR, SR * ch * 3, ch * 3, 24)
    body = b"WAVE"
    for cid, b in ((b"fmt ", fmt), (b"chna", chna), (b"axml", axml), (b"data", data)):
        body += cid + struct.pack("<I", len(b)) + b + (b"\0" if len(b) & 1 else b"")
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def _sh1(az: float, el: float) -> np.ndarray:      # SN3D ACN 0..3: W, Y, Z, X
    a, e = math.radians(az), math.radians(el)
    return np.array([1.0, math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])


def write_iamf(path, src: np.ndarray, rule: str, codec: str = "ipcm", noise: float = 0.0,
               fault_db: dict | None = None, fault_az: dict | None = None) -> None:
    """One first-order scene element (substreams 0-3) and one mono channel
    element (substream 4) for the LFE, LPCM 24-bit, from `src` by `rule`."""
    scene = np.zeros((N, 4))
    lfe = np.zeros(N)
    for k, (name, td, pos, gain, is_lfe, nblocks) in enumerate(TRACKS):
        s = src[:, k] * 10 ** ((fault_db or {}).get(name, 0.0) / 20)
        if pos[0] == "cart":
            x, y, z = pos[1:]
            if rule == "standard":
                az, el, _ = point_cart_to_polar(x, y, z)
                g = gain
            else:
                az, el = _cart_polar_az(x, y, z)
                g = gain / max(math.sqrt(x * x + y * y + z * z), 0.5)
        else:
            az, el, g = pos[1], pos[2], gain / (max(pos[3], 0.5) if rule == "converter" else 1)
        if is_lfe and rule == "standard":
            lfe += gain * s
            continue
        az = (fault_az or {}).get(name, lambda a: a)(az)
        scene += np.outer(g * s, _sh1(az, el))
    if noise:      # independent of every source: what a codec leaves behind
        rng = np.random.Generator(np.random.PCG64(4108))
        scene = scene + noise * rng.standard_normal(scene.shape)
        lfe = lfe + noise * rng.standard_normal(lfe.shape)
    assert np.max(np.abs(scene)) < 1.0 and np.max(np.abs(lfe)) < 1.0      # no clipping
    spec = scene_spec(1)
    spec.codec_id = codec
    spec.num_samples_per_frame = 480
    spec.elements.append(ElementSpec(301, 0, "channel", [4], layers=[ChannelLayerSpec(0, 1, 0)]))
    spec.mixes[0].ae_ids = [300, 301]
    spec.temporal_units = 0
    out = bytearray(build(spec))
    cols = [scene[:, i] for i in range(4)] + [lfe]
    for f in range(N // 480):
        for sid, c in enumerate(cols):
            payload = Writer().leb128(sid).raw(_pcm24(c[f * 480:(f + 1) * 480])).bytes()
            out += obu(5, payload)
    with open(path, "wb") as fh:
        fh.write(bytes(out))


@pytest.fixture()
def pair(tmp_path):
    src = _sources()
    adm = tmp_path / "master.wav"
    write_adm(str(adm), src)
    def make(rule, codec="ipcm", noise=0.0, fault_db=None, fault_az=None):
        p = tmp_path / (f"{rule}-{codec}-{noise}-{sorted((fault_db or {}).items())}-"
                        f"{sorted(fault_az or {})}.iamf")
        write_iamf(str(p), src, rule, codec, noise, fault_db, fault_az)
        return str(adm), str(p)
    return make


def _by_name(metrics):
    return {t["name"]: t for t in metrics["tracks"]}


# ------------------------------------------------------------------- checks

def test_a_conversion_to_the_standard_passes(pair):
    findings, m = conversion_fidelity(*pair("standard"))
    assert findings == [], [f.message for f in findings]
    t = _by_name(m)
    for name in ("Obj_Front", "Obj_Corner", "Obj_Polar", "RoomCentricLFE"):
        assert abs(t[name]["level_error_db"]) < 0.01, t[name]
    for name in ("Obj_Front", "Obj_Corner", "Obj_Polar"):
        assert t[name]["direction_error_deg"] < 0.1, t[name]
    assert t["Obj_Polar"]["expected_gain_db"] == pytest.approx(-6.02, abs=0.01)
    assert t["RoomCentricLFE"]["lands_in"] == "channels 301"
    assert t["Obj_Moving"]["not_judged"].startswith("2 blocks")


def test_the_converters_rule_fires_the_level_check_and_its_direction_is_the_plain_reading(pair):
    # IA-0113 (ruled Either): the converter's plain-trig corner (-45) is 15 deg from
    # 10.1.2's -30 and passes on the plain reading; its level fault still fails.
    findings, m = conversion_fidelity(*pair("converter"))
    ids = sorted(f.check_id for f in findings)
    assert ids == ["S-333"], ids
    t = _by_name(m)
    assert t["Obj_Corner"]["level_error_db"] == pytest.approx(-3.01, abs=0.01)
    assert t["RoomCentricLFE"]["level_error_db"] == pytest.approx(-4.77, abs=0.01)
    assert t["RoomCentricLFE"]["lands_in"] == "scene 300"
    assert t["Obj_Front"]["level_error_db"] == pytest.approx(0.0, abs=0.01)
    assert t["Obj_Polar"]["level_error_db"] == pytest.approx(0.0, abs=0.01)
    assert t["Obj_Corner"]["direction_error_deg"] < 0.1
    assert t["Obj_Corner"]["direction_reading"] == "plain-trig"
    assert t["Obj_Corner"]["direction_error_other_deg"] == pytest.approx(15.0, abs=0.1)
    assert t["Obj_Corner"]["direction_call"] == "pass"
    assert t["Obj_Corner"]["realised_az_el"][0] == pytest.approx(-45.0, abs=0.1)
    assert t["Obj_Front"]["direction_error_deg"] < 0.1
    assert t["Obj_Polar"]["direction_reading"] == "authored"
    assert m["cartesian_readings"]["plain-trig"] >= 1
    s333 = next(f for f in findings if f.check_id == "S-333")
    assert "RoomCentricLFE" in s333.message and "-4.77" in s333.message


def test_a_moving_track_is_counted_and_not_judged(pair):
    findings, m = conversion_fidelity(*pair("converter"))
    mv = _by_name(m)["Obj_Moving"]
    assert mv["not_judged"] and mv["level_error_db"] is not None
    assert all("Obj_Moving" not in (f.where or "") for f in findings)


def test_a_coded_iamf_is_not_judged(pair):
    findings, m = conversion_fidelity(*pair("standard", codec="Opus"))
    assert findings == [] and "not LPCM" in m["not_judged"]


def test_compare_conversion_only_needs_no_renderer(pair, monkeypatch):
    monkeypatch.setenv("SENTINEL_EAR", "/nonexistent/ear-render")
    monkeypatch.setenv("PATH", "")
    res = compare(*pair("converter"), conversion_only=True)
    assert res.execution_error is None
    assert {f.check_id for f in res.findings} == {"S-333"}
    assert res.exit_code() == 1
    res2 = compare(*pair("standard"), conversion_only=True)
    assert res2.findings == [] and res2.exit_code() == 0


# ------------------------------------------- a Cartesian block passes on either reading (IA-0113)

def test_the_standard_reading_passes_and_says_so(pair):
    findings, m = conversion_fidelity(*pair("standard"))
    t = _by_name(m)["Obj_Corner"]
    assert findings == [] and t["direction_reading"] == "bs2127-10.1.2"
    assert t["direction_error_other_deg"] == pytest.approx(15.0, abs=0.1)
    assert m["cartesian_readings"] == {"bs2127-10.1.2": 2, "plain-trig": 0}


@pytest.mark.parametrize("rule", ["standard", "converter"])
@pytest.mark.parametrize("fault,label", [
    (lambda a: -a, "mirrored"),          # left for right: -30/-45 lands at +30/+45
    (lambda a: a + 25.0, "rotated 25"),  # a wrong channel order's kind of error
])
def test_a_direction_on_neither_reading_fails(pair, rule, fault, label):
    findings, m = conversion_fidelity(*pair(rule, fault_az={"Obj_Corner": fault}))
    t = _by_name(m)["Obj_Corner"]
    assert t["direction_call"] == "fail", (label, t)
    assert t["direction_error_deg"] >= 5.0 and t["direction_error_other_deg"] >= 5.0
    s334 = [f for f in findings if f.check_id == "S-334"]
    assert len(s334) == 1 and s334[0].severity == Severity.FAIL
    assert "Obj_Corner" in s334[0].message and "either" in s334[0].expected


def test_the_plain_reading_is_iamf_tools_panner():
    from sentinel_pro.adm_conversion import point_cart_to_polar_plain
    for xyz in ((1, 1, 0), (-1, 0, 0), (0, -1, 0), (0.3, 0.8, 0.5), (-1, 1, 1)):
        assert point_cart_to_polar_plain(*xyz) == pytest.approx(_cart_polar_az(*xyz))
    assert point_cart_to_polar_plain(-1, 0, 0) == pytest.approx((90.0, 0.0))
    assert point_cart_to_polar(-1, 0, 0)[0] == pytest.approx(70.0)


# ------------------------------------------------- each reading's margin of error (IA-0108)

def test_exact_content_has_no_margin_and_the_calls_are_unchanged(pair):
    findings, m = conversion_fidelity(*pair("standard"))
    assert m["se_blocks"] == 16 and findings == []
    for t in m["tracks"]:
        if t["level_call"] is not None:
            assert t["level_se_db"] < 1e-3 and t["level_call"] == "pass", t
    findings, m = conversion_fidelity(*pair("converter"))
    assert _by_name(m)["Obj_Corner"]["level_call"] == "fail"
    assert all(f.severity == Severity.FAIL for f in findings)


def test_noise_gives_a_margin_and_a_reading_on_the_line_is_undetermined(pair):
    # The same 1.0 dB fault: exact content fails it (on the line, |e| >= tol);
    # with codec-like noise the reading is 1.0 +/- a margin and is not called.
    findings, m = conversion_fidelity(*pair("standard", noise=0.08, fault_db={"Obj_Front": -1.0}))
    t = _by_name(m)["Obj_Front"]
    assert 0.005 < t["level_se_db"] < 0.1, t
    assert t["level_call"] == "undetermined", t
    s333 = [f for f in findings if f.check_id == "S-333"]
    assert len(s333) == 1 and s333[0].severity == Severity.INFO
    assert "undetermined" in s333[0].message and "Obj_Front" in s333[0].message
    # well clear of the line either way, the noise does not change the call
    findings, m = conversion_fidelity(*pair("standard", noise=0.08, fault_db={"Obj_Front": -1.6}))
    assert _by_name(m)["Obj_Front"]["level_call"] == "fail"
    assert any(f.check_id == "S-333" and f.severity == Severity.FAIL for f in findings)
    findings, m = conversion_fidelity(*pair("standard", noise=0.08))
    assert _by_name(m)["Obj_Front"]["level_call"] == "pass"
    assert [f for f in findings if f.severity != Severity.INFO] == []


def test_noise_gives_directions_a_margin_too(pair):
    findings, m = conversion_fidelity(*pair("standard", noise=0.08))
    for name in ("Obj_Front", "Obj_Corner", "Obj_Polar"):
        t = _by_name(m)[name]
        assert t["direction_se_deg"] > 0 and t["direction_call"] == "pass", t


def test_an_undetermined_reading_does_not_fail_adm_compare(pair):
    res = compare(*pair("standard", noise=0.08, fault_db={"Obj_Front": -1.0}), conversion_only=True)
    assert [f.severity for f in res.findings] == [Severity.INFO]
    assert res.exit_code() == 0


def test_call_is_symmetric_and_k_wide():
    from sentinel_pro.adm_conversion import CALL_K, call
    assert call(1.5, 0.0, 1.0) == "fail" and call(0.99, 0.0, 1.0) == "pass"
    assert call(1.0, 0.0, 1.0) == "fail"                      # on the line, exact: >= tol
    assert call(-1.0 - 2 * CALL_K * 0.1, 0.1, 1.0) == "fail"
    assert call(1.0 - CALL_K * 0.1 + 1e-9, 0.1, 1.0) == "undetermined"
    assert call(1.0 - CALL_K * 0.1 - 1e-9, 0.1, 1.0) == "pass"
