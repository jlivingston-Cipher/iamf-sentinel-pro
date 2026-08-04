"""intent_compare units — toolchain-free (no EAR, no oracle, no samples).

Fixture files are built by the compact ADM BW64 writer below (original,
self-contained: RIFF + fmt + chna + axml + 16-bit data), so the suite runs
from a bare clone. The builder writes its own spherical->cartesian conversion
(ADM convention: az CCW-from-front, y front, x right) rather than importing
the module under test, keeping fixture authoring independent of the code it
exercises. EAR-dependent checks (S-343/S-345) live in test_intent_compare.py.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))

from sentinel.findings import REGISTRY, Severity                  # noqa: E402
from sentinel_pro import intent_compare as ic                     # noqa: E402

SR = 48000
DUR = 2.0
N = int(SR * DUR)


# ------------------------------------------------------------ ADM builder --

def _t(seconds: float) -> str:
    return f"00:00:{seconds:08.5f}"


def _sph_cart(az, el, dist=1.0):
    a, e = math.radians(az), math.radians(el)
    return (-math.sin(a) * math.cos(e) * dist,
            math.cos(a) * math.cos(e) * dist,
            math.sin(e) * dist)


def _chunk(ckid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return ckid + struct.pack("<I", len(payload)) + payload + pad


def _chna(entries) -> bytes:
    body = struct.pack("<HH", len(entries), len(entries))
    for num, uid, tref, pref in entries:
        body += (struct.pack("<H", num) + uid.encode().ljust(12)[:12]
                 + tref.encode().ljust(14)[:14] + pref.encode().ljust(11)[:11]
                 + b"\x00")
    return _chunk(b"chna", body)


XML_HEAD = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebuCore_2016">'
            '<coreMetadata><format>'
            '<audioFormatExtended version="ITU-R_BS.2076-2">')
XML_TAIL = '</audioFormatExtended></format></coreMetadata></ebuCoreMain>'

BED_LABELS_51 = ["M+030", "M-030", "M+000", "LFE1", "M+110", "M-110"]


def build_adm(path, beds=(), objects=(), amp_db=-18.0, sr=SR, dur=DUR):
    """beds: [(label, level_db|None)]  objects: [{name, blocks, gains, jump}]
    where blocks = [(rtime, duration, az, el)]. One track per entry, sines at
    313+139k Hz (track k), 16-bit PCM."""
    n = int(sr * dur)
    tracks, ents, uid_no = [], [], 0
    xml_o, xml_p, xml_c, xml_s, xml_u, contents = [], [], [], [], [], []

    def sine(k, level_db):
        t = np.arange(n) / sr
        x = math.pow(10.0, level_db / 20.0) * np.sin(
            2 * math.pi * (313.0 + 139.0 * k) * t)
        f = int(0.02 * sr)
        env = np.ones(n)
        env[:f] = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
        return x * env

    if beds:
        pack = "AP_00011100"
        chan_ids = []
        for i, (label, lvl) in enumerate(beds):
            idx = 0x1100 + i
            cid, tid, sid = (f"AC_0001{idx:04x}", f"AT_0001{idx:04x}_01",
                             f"AS_0001{idx:04x}")
            blk = (f'<audioBlockFormat audioBlockFormatID="AB_0001{idx:04x}'
                   f'_00000001" rtime="{_t(0)}" duration="{_t(dur)}">'
                   f"<speakerLabel>{label}</speakerLabel></audioBlockFormat>")
            xml_c.append(f'<audioChannelFormat audioChannelFormatID="{cid}" '
                         f'audioChannelFormatName="{label}" typeLabel="0001" '
                         f'typeDefinition="DirectSpeakers">{blk}'
                         f"</audioChannelFormat>")
            xml_s.append(
                f'<audioStreamFormat audioStreamFormatID="{sid}" '
                f'audioStreamFormatName="PCM_{cid}" formatLabel="0001" '
                f'formatDefinition="PCM">'
                f"<audioChannelFormatIDRef>{cid}</audioChannelFormatIDRef>"
                f"</audioStreamFormat>"
                f'<audioTrackFormat audioTrackFormatID="{tid}" '
                f'audioTrackFormatName="PCM_{cid}" formatLabel="0001" '
                f'formatDefinition="PCM">'
                f"<audioStreamFormatIDRef>{sid}</audioStreamFormatIDRef>"
                f"</audioTrackFormat>")
            chan_ids.append(cid)
            uid_no += 1
            uid = f"ATU_{uid_no:08d}"
            ents.append((len(tracks) + 1, uid, tid, pack))
            xml_u.append(f'<audioTrackUID UID="{uid}" sampleRate="{sr}" '
                         f'bitDepth="16">'
                         f"<audioTrackFormatIDRef>{tid}</audioTrackFormatIDRef>"
                         f"<audioPackFormatIDRef>{pack}</audioPackFormatIDRef>"
                         f"</audioTrackUID>")
            tracks.append(sine(len(tracks), amp_db if lvl is None else lvl))
        xml_p.append(f'<audioPackFormat audioPackFormatID="{pack}" '
                     f'audioPackFormatName="Bed" typeLabel="0001" '
                     f'typeDefinition="DirectSpeakers">'
                     + "".join(f"<audioChannelFormatIDRef>{c}"
                               f"</audioChannelFormatIDRef>" for c in chan_ids)
                     + "</audioPackFormat>")
        uids = [e[1] for e in ents]
        xml_o.append(f'<audioObject audioObjectID="AO_1001" '
                     f'audioObjectName="Bed" start="{_t(0)}" '
                     f'duration="{_t(dur)}">'
                     f"<audioPackFormatIDRef>{pack}</audioPackFormatIDRef>"
                     + "".join(f"<audioTrackUIDRef>{u}</audioTrackUIDRef>"
                               for u in uids) + "</audioObject>")
        contents.append("AO_1001")

    for o_i, obj in enumerate(objects):
        idx = 0x1001 + o_i
        pack, cid = f"AP_0003{idx:04x}", f"AC_0003{idx:04x}"
        tid, sid = f"AT_0003{idx:04x}_01", f"AS_0003{idx:04x}"
        blocks = []
        gains = obj.get("gains") or [None] * len(obj["blocks"])
        for b_i, (rt, du, az, el) in enumerate(obj["blocks"]):
            x, y, z = _sph_cart(az, el)
            g = "" if gains[b_i] is None else f"<gain>{gains[b_i]:.6f}</gain>"
            j = ('<jumpPosition switch="1" interpolationLength="0.0"/>'
                 if obj.get("jump") else "")
            blocks.append(
                f'<audioBlockFormat audioBlockFormatID="AB_0003{idx:04x}_'
                f'{b_i + 1:08x}" rtime="{_t(rt)}" duration="{_t(du)}">'
                f"<cartesian>1</cartesian>"
                f'<position coordinate="X">{x:.6f}</position>'
                f'<position coordinate="Y">{y:.6f}</position>'
                f'<position coordinate="Z">{z:.6f}</position>{g}{j}'
                f"</audioBlockFormat>")
        xml_c.append(f'<audioChannelFormat audioChannelFormatID="{cid}" '
                     f'audioChannelFormatName="{obj["name"]}" '
                     f'typeLabel="0003" typeDefinition="Objects">'
                     + "".join(blocks) + "</audioChannelFormat>")
        xml_s.append(
            f'<audioStreamFormat audioStreamFormatID="{sid}" '
            f'audioStreamFormatName="PCM_{cid}" formatLabel="0001" '
            f'formatDefinition="PCM">'
            f"<audioChannelFormatIDRef>{cid}</audioChannelFormatIDRef>"
            f"</audioStreamFormat>"
            f'<audioTrackFormat audioTrackFormatID="{tid}" '
            f'audioTrackFormatName="PCM_{cid}" formatLabel="0001" '
            f'formatDefinition="PCM">'
            f"<audioStreamFormatIDRef>{sid}</audioStreamFormatIDRef>"
            f"</audioTrackFormat>")
        xml_p.append(f'<audioPackFormat audioPackFormatID="{pack}" '
                     f'audioPackFormatName="{obj["name"]}" typeLabel="0003" '
                     f'typeDefinition="Objects">'
                     f"<audioChannelFormatIDRef>{cid}</audioChannelFormatIDRef>"
                     f"</audioPackFormat>")
        uid_no += 1
        uid = f"ATU_{uid_no:08d}"
        ents.append((len(tracks) + 1, uid, tid, pack))
        xml_u.append(f'<audioTrackUID UID="{uid}" sampleRate="{sr}" '
                     f'bitDepth="16">'
                     f"<audioTrackFormatIDRef>{tid}</audioTrackFormatIDRef>"
                     f"<audioPackFormatIDRef>{pack}</audioPackFormatIDRef>"
                     f"</audioTrackUID>")
        oid = f"AO_{2001 + o_i}"
        xml_o.append(f'<audioObject audioObjectID="{oid}" '
                     f'audioObjectName="{obj["name"]}" start="{_t(0)}" '
                     f'duration="{_t(dur)}">'
                     f"<audioPackFormatIDRef>{pack}</audioPackFormatIDRef>"
                     f"<audioTrackUIDRef>{uid}</audioTrackUIDRef>"
                     f"</audioObject>")
        contents.append(oid)
        lvl = obj.get("level_db", amp_db)
        if obj.get("mute"):
            tracks.append(np.zeros(n))
        else:
            tracks.append(sine(len(tracks), lvl))

    xml = (XML_HEAD
           + f'<audioProgramme audioProgrammeID="APR_1001" '
             f'audioProgrammeName="P" start="{_t(0)}" end="{_t(dur)}">'
           + "".join(f"<audioContentIDRef>ACO_1001</audioContentIDRef>"
                     for _ in [0])
           + "</audioProgramme>"
           + '<audioContent audioContentID="ACO_1001" audioContentName="C">'
           + "".join(f"<audioObjectIDRef>{o}</audioObjectIDRef>"
                     for o in contents)
           + "</audioContent>"
           + "".join(xml_o) + "".join(xml_p) + "".join(xml_c)
           + "".join(xml_s) + "".join(xml_u) + XML_TAIL)

    data = np.stack(tracks, axis=1) if tracks else np.zeros((n, 1))
    ints = np.clip(np.round(data * 32767.0), -32768, 32767).astype("<i2")
    nch = ints.shape[1]
    fmt = struct.pack("<HHIIHH", 1, nch, sr, sr * nch * 2, nch * 2, 16)
    body = (_chunk(b"fmt ", fmt) + _chna(ents)
            + _chunk(b"axml", xml.encode("utf-8"))
            + _chunk(b"data", ints.tobytes()))
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    return str(path)


def sidecar(path, *, bed=None, objects=None, scene=None, tolerances=None,
            end=DUR, sr=SR, producer_export="spatial.export_adm"):
    doc = {"intent": 0,
           "producer": {"tool": "hand", "version": "test",
                        "export": producer_export},
           "program": {"sampleRate": sr, "start": 0.0, "end": end,
                       "sliceSec": 0.1}}
    if bed is not None:
        doc["bed"] = bed
    if objects is not None:
        doc["objects"] = objects
    if scene is not None:
        doc["scene"] = scene
    if tolerances is not None:
        doc["tolerances"] = tolerances
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return str(path)


def sweep_blocks(n_blocks=4, dur=DUR, az0=0.0, rate_dps=45.0):
    bl, step = [], dur / n_blocks
    return [(i * step, step, az0 + rate_dps * (i + 1) * step, 0.0)
            for i in range(n_blocks)]


def sweep_traj(n=41, dur=DUR, az0=0.0, rate_dps=45.0):
    """Dense sidecar trajectory under DELIVERED semantics: the first block
    holds its end value across its span, later blocks interpolate to end."""
    pts, step = [], dur / 4
    for i in range(n):
        t = dur * i / (n - 1)
        if t <= step:                       # first block holds
            az = az0 + rate_dps * step
        else:
            az = az0 + rate_dps * t
        pts.append({"t": round(t, 4), "az": round(az, 4), "el": 0.0,
                    "dist": 1.0})
    return pts


def static_traj(az, el=0.0, end=DUR):
    return [{"t": 0.0, "az": az, "el": el, "dist": 1.0},
            {"t": end, "az": az, "el": el, "dist": 1.0}]


CLEAN_51_LEVELS = [-18.0, -21.0, -24.0, -30.0, -20.0, -23.0]


def clean_bed_file(tmp_path, name="bed.wav"):
    return build_adm(tmp_path / name,
                     beds=list(zip(BED_LABELS_51, CLEAN_51_LEVELS)))


def clean_bed_roster(measured=None):
    lv = measured or CLEAN_51_LEVELS
    return {"layout": "5.1",
            "roster": [{"channel": lab, "track": i + 1,
                        "expectRmsDb": lv[i] - 3.0103,
                        "expectActive": 0.98}
                       for i, lab in enumerate(BED_LABELS_51)]}


# --------------------------------------------------------------- registry --

def test_s34x_registered_fail_l3_no_frefs():
    for cid in ("S-340", "S-341", "S-342", "S-343", "S-344", "S-345",
                "S-346"):
        assert cid in REGISTRY
        assert REGISTRY[cid].default_severity == Severity.FAIL
        assert REGISTRY[cid].layer == "L3"
        assert REGISTRY[cid].f_refs == ()
        assert REGISTRY[cid].description


# --------------------------------------------------------------- geometry --

def test_canonical_labels_cover_friendly_and_roomcentric():
    assert ic.canonical_label("L") == "M+030"
    assert ic.canonical_label("Lrs") == "M+135"
    assert ic.canonical_label("RoomCentricLeftSideSurround") == "M+090"
    assert ic.canonical_label("M+030") == "M+030"


def test_angular_error_basics():
    a = ic.sph_to_unit(90, 0)
    b = ic.sph_to_unit(-90, 0)
    assert abs(ic.angular_error_deg(a, b) - 180.0) < 1e-9
    assert ic.angular_error_deg(a, a) < 1e-9


def test_trajectory_hold_before_first_and_jump():
    pts = [ic.TrajPoint(1.0, *ic.sph_to_unit(30, 0), gain=1.0, hold=True),
           ic.TrajPoint(2.0, *ic.sph_to_unit(60, 0), gain=1.0)]
    x, y, z, _ = ic.sample_trajectory(pts, 0.0)      # hold: first value
    az, _el, _d = ic.cart_to_sph(x, y, z)
    assert abs(az - 30.0) < 1e-6
    x, y, z, _ = ic.sample_trajectory(pts, 1.5)      # chord midpoint
    az, _el, _d = ic.cart_to_sph(x, y, z)
    assert abs(az - 45.0) < 1.0


def test_tolerance_floors_clamp_hostile_values():
    eff = ic.effective_tolerances({"tolerances": {
        "posDeg": 0.01, "levelLu": 0.01, "dominantFrac": 1.0,
        "covSilent": 0.5, "silentDb": -5.0}})
    assert eff["posDeg"] == 1.0
    assert eff["levelLu"] == 0.25
    assert eff["dominantFrac"] == 0.95
    assert eff["covSilent"] == 0.5          # looser than the floor: kept
    assert eff["silentDb"] == -30.0
    assert set(eff["floors_applied"]) == {"posDeg", "levelLu",
                                          "dominantFrac", "silentDb"}


# ------------------------------------------------------------ parsing ------

def test_parse_delivered_custom_bed_and_objects(tmp_path):
    p = build_adm(tmp_path / "mix.wav",
                  beds=list(zip(BED_LABELS_51, CLEAN_51_LEVELS)),
                  objects=[{"name": "Obj1", "blocks": sweep_blocks()}])
    m = ic.parse_delivered(p)
    assert m.sample_rate == SR
    assert m.n_tracks == 7
    assert m.bed_layout == "5.1"
    assert set(m.bed_tracks) == set(BED_LABELS_51)
    assert list(m.objects) == [7]
    assert len(m.objects[7].points) == 4


def test_parse_delivered_cd_stereo(tmp_path):
    """Common-definitions refs (no in-file channel formats) map to labels."""
    n = int(SR * DUR)
    t = np.arange(n) / SR
    data = np.stack([0.1 * np.sin(2 * math.pi * 313 * t),
                     0.1 * np.sin(2 * math.pi * 452 * t)], axis=1)
    ints = (data * 32767).astype("<i2")
    fmt = struct.pack("<HHIIHH", 1, 2, SR, SR * 4, 4, 16)
    xml = (XML_HEAD
           + f'<audioProgramme audioProgrammeID="APR_1001" '
             f'audioProgrammeName="P" start="{_t(0)}" end="{_t(DUR)}">'
             f"</audioProgramme>"
           + '<audioObject audioObjectID="AO_1001" audioObjectName="Bed">'
             "<audioPackFormatIDRef>AP_00010002</audioPackFormatIDRef>"
             "<audioTrackUIDRef>ATU_00000001</audioTrackUIDRef>"
             "<audioTrackUIDRef>ATU_00000002</audioTrackUIDRef>"
             "</audioObject>"
           + '<audioTrackUID UID="ATU_00000001" sampleRate="48000" '
             'bitDepth="16">'
             "<audioTrackFormatIDRef>AT_00010001_01</audioTrackFormatIDRef>"
             "<audioPackFormatIDRef>AP_00010002</audioPackFormatIDRef>"
             "</audioTrackUID>"
             '<audioTrackUID UID="ATU_00000002" sampleRate="48000" '
             'bitDepth="16">'
             "<audioTrackFormatIDRef>AT_00010002_01</audioTrackFormatIDRef>"
             "<audioPackFormatIDRef>AP_00010002</audioPackFormatIDRef>"
             "</audioTrackUID>"
           + XML_TAIL)
    body = (_chunk(b"fmt ", fmt)
            + _chna([(1, "ATU_00000001", "AT_00010001_01", "AP_00010002"),
                     (2, "ATU_00000002", "AT_00010002_01", "AP_00010002")])
            + _chunk(b"axml", xml.encode())
            + _chunk(b"data", ints.tobytes()))
    p = tmp_path / "cd.wav"
    p.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    m = ic.parse_delivered(str(p))
    assert m.bed_layout == "2.0"
    assert m.bed_tracks == {"M+030": 1, "M-030": 2}


# ------------------------------------------------------------- checks ------

def test_clean_bed_zero_findings_and_deterministic(tmp_path):
    f = clean_bed_file(tmp_path)
    sc = sidecar(tmp_path / "s.json", bed=clean_bed_roster())
    r1 = ic.compare(sc, f)
    assert r1.execution_error is None
    assert r1.findings == []
    assert r1.exit_code() == 0
    r2 = ic.compare(sc, f)
    assert json.dumps(r1.to_dict()) == json.dumps(r2.to_dict())


def test_roster_gap_fires_s340_once(tmp_path):
    f = build_adm(tmp_path / "bed4.wav",
                  beds=list(zip(BED_LABELS_51, CLEAN_51_LEVELS))[:5])
    sc = sidecar(tmp_path / "s.json", bed=clean_bed_roster())
    r = ic.compare(sc, f)
    assert [x.check_id for x in r.findings] == ["S-340"]
    assert "M-110" in r.findings[0].message


def test_bed_level_shift_fires_s344(tmp_path):
    lv = list(CLEAN_51_LEVELS)
    lv[0] -= 6.0                                     # deliver L 6 dB down
    f = build_adm(tmp_path / "bed.wav", beds=list(zip(BED_LABELS_51, lv)))
    sc = sidecar(tmp_path / "s.json", bed=clean_bed_roster())
    r = ic.compare(sc, f)
    assert [x.check_id for x in r.findings] == ["S-344"]
    assert r.findings[0].where == "bed channel M+030"


def test_object_mute_fires_s341_and_suppresses_the_rest(tmp_path):
    f = build_adm(tmp_path / "objs.wav",
                  objects=[{"name": "Obj1", "blocks": sweep_blocks()},
                           {"name": "Obj2", "blocks": sweep_blocks(az0=90),
                            "mute": True}])
    objs = [{"id": 1, "label": "Obj1", "track": 1,
             "trajectory": sweep_traj(),
             "expectRmsDb": -21.01, "expectActive": 0.98},
            {"id": 2, "label": "Obj2", "track": 2,
             "trajectory": sweep_traj(az0=90),
             "expectRmsDb": -21.01, "expectActive": 0.98}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    r = ic.compare(sc, f)
    assert [x.check_id for x in r.findings] == ["S-341"]
    assert "object 2" in r.findings[0].where
    om = r.metrics["objects"]["object 2"]
    assert "trajectory_p95_deg" not in om            # suppressed


def test_clean_sweep_then_zeroed_positions_fire_s342(tmp_path):
    f_clean = build_adm(tmp_path / "sweep.wav",
                        objects=[{"name": "Obj1", "blocks": sweep_blocks()}])
    objs = [{"id": 1, "track": 1, "trajectory": sweep_traj(),
             "expectRmsDb": -21.01, "expectActive": 0.98}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    r = ic.compare(sc, f_clean)
    assert r.findings == []                          # convention agreement
    f_zero = build_adm(tmp_path / "zeroed.wav",
                       objects=[{"name": "Obj1",
                                 "blocks": [(0.0, DUR, 0.0, 0.0)]}])
    r = ic.compare(sc, f_zero)
    assert [x.check_id for x in r.findings] == ["S-342"]
    assert r.metrics["objects"]["object 1"]["trajectory_p95_deg"] > 5.0


def test_gain_strip_fires_s344_gain_half(tmp_path):
    gains = [1.0, 0.7, 0.4, 0.2]
    blocks = sweep_blocks()
    f_clean = build_adm(tmp_path / "ga.wav",
                        objects=[{"name": "Obj1", "blocks": blocks,
                                  "gains": gains}])
    traj = sweep_traj(n=5)                           # points AT block ends
    for i, p in enumerate(traj):
        # delivered semantics: block i's gain is reached at its END
        # (t = (i+1)*step); the first block holds its gain from t=0.
        p["gain"] = gains[max(0, i - 1)]
    objs = [{"id": 1, "track": 1, "trajectory": traj,
             "expectActive": 0.98}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    r = ic.compare(sc, f_clean)
    assert r.findings == []
    f_flat = build_adm(tmp_path / "gb.wav",
                       objects=[{"name": "Obj1", "blocks": blocks}])
    r = ic.compare(sc, f_flat)
    assert [x.check_id for x in r.findings] == ["S-344"]
    assert "gain trajectory" in r.findings[0].where


def test_wrong_file_fires_s346_and_gates(tmp_path):
    f = build_adm(tmp_path / "st.wav",
                  beds=[("M+030", -18.0), ("M-030", -18.0)])
    sc = sidecar(tmp_path / "s.json", bed=clean_bed_roster())
    r = ic.compare(sc, f)
    assert [x.check_id for x in r.findings] == ["S-346"]
    assert r.metrics["skipped"] == ["S-340", "S-341", "S-342", "S-343",
                                    "S-344", "S-345"]


def test_hostile_tolerances_floored_clean_still_passes(tmp_path):
    f = clean_bed_file(tmp_path)
    sc = sidecar(tmp_path / "s.json", bed=clean_bed_roster(),
                 tolerances={"posDeg": 0.01, "levelLu": 0.01,
                             "dominantFrac": 1.0, "covSilent": 0.5,
                             "silentDb": -5.0})
    r = ic.compare(sc, f)
    assert r.findings == []
    assert r.exit_code() == 0
    fl = r.metrics["tolerances_effective"]["floors_applied"]
    assert set(fl) == {"posDeg", "levelLu", "dominantFrac", "silentDb"}


def test_decode_prediction_without_ear_is_execution_error(tmp_path,
                                                          monkeypatch):
    f = build_adm(tmp_path / "o.wav",
                  objects=[{"name": "Obj1",
                            "blocks": [(0.0, DUR, 90.0, 0.0)]}])
    objs = [{"id": 1, "track": 1, "trajectory": static_traj(90.0),
             "expectActive": 0.98,
             "decode": {"layout": "7.1.4", "dominant": ["M+090"] * 20,
                        "coverage": {"M+090": 1.0}}}]
    sc = sidecar(tmp_path / "s.json", objects=objs)
    monkeypatch.setattr(ic, "find_ear", lambda: None)
    r = ic.compare(sc, f)
    assert r.execution_error is not None
    assert "ear-render" in r.execution_error
    assert r.exit_code() == 2


def test_unknown_schema_version_is_execution_error(tmp_path):
    f = clean_bed_file(tmp_path)
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"intent": 1}), encoding="utf-8")
    r = ic.compare(str(p), f)
    assert r.exit_code() == 2
    assert "intent" in r.execution_error
