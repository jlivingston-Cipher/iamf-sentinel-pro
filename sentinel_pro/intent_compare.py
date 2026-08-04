"""Intent-conformance QC (B1-R3): session intent sidecar vs delivered ADM BWF.

The sidecar is the authoring session's own prediction of what it exported —
geometry, levels, activity — emitted at export time (schema v0, envelope
``"intent": 0``). This module is the CONSUMER half: decode the delivered ADM
BW64, measure the same quantities, and answer "does the file render as the
session intended?" with S-34x findings. No DAW logic lives here; no IAMF
decode in v1 (the schema is container-agnostic — IAMF input is v1.1 via the
L3 render path).

Checks (S-34x family, registered in sentinel.findings):

    S-340  roster mismatch (bed channel absent/extra vs roster)        FAIL
    S-341  object missing / predicted-active object silent             FAIL
    S-342  trajectory divergence (angular, per-slice, beyond posDeg)   FAIL
    S-343  dominant-speaker mismatch (isolated render vs prediction)   FAIL
    S-344  level/gain delta beyond levelLu (audio and metadata halves) FAIL
    S-345  energy at predicted-silent channels (wrong-fold class)      FAIL
    S-346  sidecar/file program shape mismatch (gates everything)      FAIL

Measurement conventions (pre-registered, doc 123):

* Trajectories interpolate CARTESIAN-CHORD between points (wrap-safe; both
  sides sampled by the same function). A sidecar point's value is reached at
  its ``t`` and holds constant before the first point. Delivered
  audioBlockFormat values apply at rtime+duration interpolating from the
  previous block's value (BS.2076); ``jumpPosition switch="1"`` blocks are
  piecewise-constant; a first block holds its value across its span. S-342
  judges ANGULAR error only (great-circle); distance is reported in metrics.
* Dominant-speaker measurement uses a PER-OBJECT ISOLATION render: every
  other track silenced (adm_compare.silence_tracks), EAR renders the result,
  per-slice argmax channel energy (LFE excluded). Slices whose top-2 channels
  are within 6 dB carry no clear dominant and leave the agreement
  denominator; if every slice leaves, S-343 is skipped for that object.
* Per-channel/per-object expectLufs is BS.1770-4 K-weighted gated with
  channel weight 1.0; bed.expectLufs uses the layout's conformant
  positional weights (dsp.CHANNEL_WEIGHTS). expectRmsDb is plain dBFS RMS.
* Consumer-side tolerance floors guard against a hostile sidecar demanding
  impossible precision: posDeg >= 1.0, levelLu >= 0.25, dominantFrac <= 0.95,
  covSilent >= 0.005, silentDb <= -30, sliceSec >= 0.05. Effective values are
  recorded in ``metrics.tolerances_effective``.

Fail-closed: a sidecar that carries ``decode`` blocks needs ear-render; when
it is absent the run is an execution error (exit 2), never a silently
narrower report. Exit-code contract identical to adm_compare (0/1/2).

IP boundary: original work. BW64/chna/axml reading from ITU-R BS.2088 /
EBU Tech 3285 / BS.2076; EAR (BSD-3-Clause) is a subprocess oracle only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from sentinel.findings import Finding, Severity, spec
from .adm_compare import (_read_wav, _find_chunks, silence_tracks,
                          ear_render, find_ear, EAR_LAYOUTS)
from . import dsp

SCHEMA_VERSION = 0
ACTIVITY_WIN_S = 0.100          # activity window (100 ms)
ACTIVITY_FLOOR_DB = -60.0       # window RMS above this counts as active
DOMINANCE_GAP_DB = 6.0          # top-2 within this -> no clear dominant
SLICE_SILENCE_DB = -70.0        # slices below this total energy are skipped
DURATION_TOL_S = 0.05

# Consumer-side floors (hostile-sidecar guard). Direction matters: a floor
# CLAMPS the demand to the loosest of (sidecar value, floor limit).
TOLERANCE_FLOORS = {"posDeg": 1.0, "levelLu": 0.25, "dominantFrac": 0.95,
                    "covSilent": 0.005, "silentDb": -30.0, "sliceSec": 0.05}
TOLERANCE_DEFAULTS = {"posDeg": 5.0, "levelLu": 0.5, "dominantFrac": 0.9,
                      "covSilent": 0.01, "silentDb": -40.0}

# ---------------------------------------------------------------- layouts --

# Canonical BS.2051-style labels, in the delivered/rendered channel order the
# rest of the stack uses (WP1 decode-verify convention; EAR system order).
LAYOUT_LABELS: dict[str, list[str]] = {
    "2.0":   ["M+030", "M-030"],
    "5.1":   ["M+030", "M-030", "M+000", "LFE1", "M+110", "M-110"],
    "5.1.2": ["M+030", "M-030", "M+000", "LFE1", "M+110", "M-110",
              "U+030", "U-030"],
    "5.1.4": ["M+030", "M-030", "M+000", "LFE1", "M+110", "M-110",
              "U+030", "U-030", "U+110", "U-110"],
    "7.1":   ["M+030", "M-030", "M+000", "LFE1", "M+090", "M-090",
              "M+135", "M-135"],
    "7.1.4": ["M+030", "M-030", "M+000", "LFE1", "M+090", "M-090",
              "M+135", "M-135", "U+045", "U-045", "U+135", "U-135"],
}

# Label -> (az CCW-from-front, el), matching the EAR system geometry the
# isolation renders come back in (fidelity-measure convention). LFE has none.
LABEL_POSITIONS: dict[str, tuple[float, float] | None] = {
    "M+030": (30, 0), "M-030": (-30, 0), "M+000": (0, 0), "LFE1": None,
    "M+110": (110, 0), "M-110": (-110, 0), "M+090": (90, 0), "M-090": (-90, 0),
    "M+135": (135, 0), "M-135": (-135, 0), "U+030": (30, 30), "U-030": (-30, 30),
    "U+110": (110, 30), "U-110": (-110, 30), "U+045": (45, 45), "U-045": (-45, 45),
    "U+135": (135, 45), "U-135": (-135, 45),
}

# Friendly + Dolby room-centric aliases -> canonical labels. The friendly set
# follows the 7.1.4 naming (Ltf/Rtf = U+045 there); sidecars SHOULD use
# canonical labels, aliases are a courtesy.
LABEL_ALIASES: dict[str, str] = {
    "L": "M+030", "R": "M-030", "C": "M+000", "LFE": "LFE1",
    "Ls": "M+110", "Rs": "M-110", "Lss": "M+090", "Rss": "M-090",
    "Lrs": "M+135", "Rrs": "M-135", "Ltf": "U+045", "Rtf": "U-045",
    "Ltb": "U+135", "Rtb": "U-135",
    "RoomCentricLeft": "M+030", "RoomCentricRight": "M-030",
    "RoomCentricCenter": "M+000", "RoomCentricLFE": "LFE1",
    "RoomCentricLeftSurround": "M+110", "RoomCentricRightSurround": "M-110",
    "RoomCentricLeftSideSurround": "M+090",
    "RoomCentricRightSideSurround": "M-090",
    "RoomCentricLeftRearSurround": "M+135",
    "RoomCentricRightRearSurround": "M-135",
    "RoomCentricLeftTopSurround": "U+090", "RoomCentricRightTopSurround": "U-090",
    "RoomCentricLeftTopFront": "U+045", "RoomCentricRightTopFront": "U-045",
    "RoomCentricLeftTopRear": "U+135", "RoomCentricRightTopRear": "U-135",
}

# BS.2094 common-definitions packs the delivered side may reference.
CD_PACK_LAYOUT = {"AP_00010002": "2.0", "AP_00010003": "5.1",
                  "AP_00010004": "5.1.2", "AP_00010005": "5.1.4",
                  "AP_0001000f": "7.1", "AP_00010017": "7.1.4"}
CD_PACK_HOA = {"AP_00040001": 1, "AP_00040003": 3}      # pack -> order
CD_PACK_BINAURAL = {"AP_00050001"}
_CD_TRACK_LISTS = {
    "2.0":   ["AT_00010001_01", "AT_00010002_01"],
    "5.1":   ["AT_00010001_01", "AT_00010002_01", "AT_00010003_01",
              "AT_00010004_01", "AT_00010005_01", "AT_00010006_01"],
    "5.1.2": ["AT_00010001_01", "AT_00010002_01", "AT_00010003_01",
              "AT_00010004_01", "AT_00010005_01", "AT_00010006_01",
              "AT_0001000d_01", "AT_0001000f_01"],
    "5.1.4": ["AT_00010001_01", "AT_00010002_01", "AT_00010003_01",
              "AT_00010004_01", "AT_00010005_01", "AT_00010006_01",
              "AT_0001000d_01", "AT_0001000f_01", "AT_00010010_01",
              "AT_00010012_01"],
    "7.1":   ["AT_00010001_01", "AT_00010002_01", "AT_00010003_01",
              "AT_00010004_01", "AT_0001000a_01", "AT_0001000b_01",
              "AT_0001001c_01", "AT_0001001d_01"],
    "7.1.4": ["AT_00010001_01", "AT_00010002_01", "AT_00010003_01",
              "AT_00010004_01", "AT_0001000a_01", "AT_0001000b_01",
              "AT_0001001c_01", "AT_0001001d_01", "AT_00010022_01",
              "AT_00010023_01", "AT_0001001e_01", "AT_0001001f_01"],
}
CD_TRACK_LABEL: dict[str, str] = {}
for _lay, _ids in _CD_TRACK_LISTS.items():
    for _tid, _lab in zip(_ids, LAYOUT_LABELS[_lay]):
        CD_TRACK_LABEL.setdefault(_tid, _lab)


def canonical_label(name: str) -> str:
    return LABEL_ALIASES.get(name, name)


# ------------------------------------------------------------- geometry ----

def sph_to_unit(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """ADM convention: az CCW-from-front (+left), el up; y front, x right."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return (-math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def cart_to_sph(x: float, y: float, z: float) -> tuple[float, float, float]:
    dist = math.sqrt(x * x + y * y + z * z)
    if dist < 1e-12:
        return 0.0, 0.0, 0.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, z / dist))))
    az = math.degrees(math.atan2(-x, y))
    return az, el, dist


def angular_error_deg(u: tuple[float, float, float],
                      v: tuple[float, float, float]) -> float:
    nu = math.sqrt(sum(c * c for c in u)) or 1e-12
    nv = math.sqrt(sum(c * c for c in v)) or 1e-12
    dot = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


# --------------------------------------------------------- trajectories ----

@dataclass
class TrajPoint:
    t: float                    # time the value is reached / applies
    x: float
    y: float
    z: float
    gain: float = 1.0
    hold: bool = False          # True: constant back to the previous point


def _lerp_point(p0: TrajPoint, p1: TrajPoint, t: float) -> tuple:
    """Cartesian-chord interpolation between p0 (at t0) and p1 (at t1)."""
    if p1.hold or p1.t <= p0.t:
        return (p1.x, p1.y, p1.z, p1.gain) if t >= p1.t - 1e-9 else \
               (p0.x, p0.y, p0.z, p0.gain)
    f = max(0.0, min(1.0, (t - p0.t) / (p1.t - p0.t)))
    return (p0.x + f * (p1.x - p0.x), p0.y + f * (p1.y - p0.y),
            p0.z + f * (p1.z - p0.z), p0.gain + f * (p1.gain - p0.gain))


def sample_trajectory(points: list[TrajPoint], t: float) -> tuple:
    """(x, y, z, gain) at time t under the module's shared conventions."""
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    if t <= points[0].t:
        p = points[0]
        return p.x, p.y, p.z, p.gain
    for i in range(1, len(points)):
        if t <= points[i].t:
            return _lerp_point(points[i - 1], points[i], t)
    p = points[-1]
    return p.x, p.y, p.z, p.gain


def sidecar_trajectory(obj: dict) -> list[TrajPoint]:
    pts = []
    for p in obj.get("trajectory", []):
        x, y, z = sph_to_unit(float(p.get("az", 0.0)), float(p.get("el", 0.0)))
        d = float(p.get("dist", 1.0))
        pts.append(TrajPoint(float(p["t"]), x * d, y * d, z * d,
                             float(p.get("gain", 1.0))))
    pts.sort(key=lambda p: p.t)
    return pts


# ----------------------------------------------------------- ADM parsing ---

def _parse_hms(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass
class DeliveredObject:
    track: int                  # 1-based BW64 track number
    name: str
    points: list[TrajPoint] = field(default_factory=list)
    has_gain_meta: bool = False


@dataclass
class DeliveredModel:
    sample_rate: int = 0
    frames: int = 0
    n_tracks: int = 0
    bed_layout: str | None = None
    bed_tracks: dict[str, int] = field(default_factory=dict)   # label -> track
    objects: dict[int, DeliveredObject] = field(default_factory=dict)
    scene_order: int | None = None
    scene_tracks: list[int] = field(default_factory=list)      # ACN order
    binaural: bool = False
    notes: list[str] = field(default_factory=list)


def _parse_chna(body: bytes) -> list[tuple[int, str, str, str]]:
    """[(track_num, uid, trackFormatRef, packFormatRef)] from a chna chunk."""
    num_uids = struct.unpack("<H", body[2:4])[0]
    out = []
    for i in range(num_uids):
        e = body[4 + 40 * i: 4 + 40 * (i + 1)]
        if len(e) < 40:
            break
        num = struct.unpack("<H", e[0:2])[0]
        uid = e[2:14].decode("ascii", "replace").strip("\x00 ")
        tref = e[14:28].decode("ascii", "replace").strip("\x00 ")
        pref = e[28:39].decode("ascii", "replace").strip("\x00 ")
        out.append((num, uid, tref, pref))
    return out


def _block_points(cf_el) -> tuple[list[TrajPoint], bool]:
    """TrajPoints (delivered semantics) + has_gain from an audioChannelFormat."""
    pts: list[TrajPoint] = []
    has_gain = False
    blocks = [b for b in cf_el if _local(b.tag) == "audioBlockFormat"]
    for b in blocks:
        rt = _parse_hms(b.get("rtime", "00:00:00.00000"))
        du = _parse_hms(b.get("duration", "00:00:00.00000"))
        cart = None
        coords: dict[str, float] = {}
        gain = None
        jump = False
        for ch in b:
            nm = _local(ch.tag)
            if nm == "cartesian":
                cart = (ch.text or "").strip() == "1"
            elif nm == "position":
                coords[ch.get("coordinate", "")] = float(ch.text or 0.0)
            elif nm == "gain":
                gain = float(ch.text or 1.0)
            elif nm == "jumpPosition":
                # both authored shapes seen in the wild: the switch attribute
                # (corpus generator) and the element-text bool (EAR's reading)
                jump = (ch.get("switch", "").strip() == "1"
                        or (ch.text or "").strip() in ("1", "true"))
        if gain is not None:
            has_gain = True
        if cart or ("X" in coords or "Y" in coords):
            x = coords.get("X", 0.0)
            y = coords.get("Y", 1.0)
            z = coords.get("Z", 0.0)
        else:
            az = coords.get("azimuth", 0.0)
            el = coords.get("elevation", 0.0)
            dist = coords.get("distance", 1.0)
            ux, uy, uz = sph_to_unit(az, el)
            x, y, z = ux * dist, uy * dist, uz * dist
        g = gain if gain is not None else 1.0
        if jump:
            # value applies from rtime, constant across the block
            pts.append(TrajPoint(rt, x, y, z, g, hold=True))
        else:
            # BS.2076: interpolate from the previous value, reached at
            # rtime+duration; a first block holds its value across its span.
            pts.append(TrajPoint(rt + du, x, y, z, g,
                                 hold=not pts))
    pts.sort(key=lambda p: p.t)
    return pts, has_gain


def parse_delivered(path: str) -> DeliveredModel:
    """Walk the delivered ADM BW64 into the model intent-compare judges."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] not in (b"RIFF", b"BW64", b"RF64"):
        raise ValueError(f"{path}: not a RIFF/BW64 file")
    chunks = _find_chunks(data)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise ValueError(f"{path}: no fmt /data chunk")
    off, size = chunks[b"fmt "]
    _tag, nch, rate, _bps, block_align, _bits = struct.unpack(
        "<HHIIHH", data[off:off + 16])
    m = DeliveredModel(sample_rate=rate, n_tracks=nch)
    _doff, dsize = chunks[b"data"]
    m.frames = dsize // block_align if block_align else 0
    if b"axml" not in chunks or b"chna" not in chunks:
        raise ValueError(f"{path}: no axml/chna chunk — not an ADM BW64")
    off, size = chunks[b"chna"]
    chna = _parse_chna(data[off:off + size])
    off, size = chunks[b"axml"]
    root = ET.fromstring(data[off:off + size].decode("utf-8", "replace"))

    uid_to_refs: dict[str, tuple[str, str]] = {}
    objects_by_uid: dict[str, str] = {}
    packs: dict[str, tuple[str, str, list[str]]] = {}   # id -> (name, type, chans)
    chans: dict[str, ET.Element] = {}
    track_to_chan: dict[str, str] = {}                  # trackFormat -> chan (via stream)
    stream_to_chan: dict[str, str] = {}
    for el in root.iter():
        nm = _local(el.tag)
        if nm == "audioTrackUID":
            uid = el.get("UID", "")
            tref = pref = ""
            for ch in el:
                if _local(ch.tag) == "audioTrackFormatIDRef":
                    tref = (ch.text or "").strip()
                elif _local(ch.tag) == "audioPackFormatIDRef":
                    pref = (ch.text or "").strip()
            uid_to_refs[uid] = (tref, pref)
        elif nm == "audioObject":
            oname = el.get("audioObjectName", el.get("audioObjectID", "?"))
            for ch in el:
                if _local(ch.tag) == "audioTrackUIDRef":
                    objects_by_uid[(ch.text or "").strip()] = oname
        elif nm == "audioPackFormat":
            pid = el.get("audioPackFormatID", "")
            ptype = el.get("typeDefinition", el.get("typeLabel", ""))
            crefs = [(c.text or "").strip() for c in el
                     if _local(c.tag) == "audioChannelFormatIDRef"]
            packs[pid] = (el.get("audioPackFormatName", pid), ptype, crefs)
        elif nm == "audioChannelFormat":
            chans[el.get("audioChannelFormatID", "")] = el
        elif nm == "audioStreamFormat":
            sid = el.get("audioStreamFormatID", "")
            for ch in el:
                if _local(ch.tag) == "audioChannelFormatIDRef":
                    stream_to_chan[sid] = (ch.text or "").strip()
        elif nm == "audioTrackFormat":
            tid = el.get("audioTrackFormatID", "")
            for ch in el:
                if _local(ch.tag) == "audioStreamFormatIDRef":
                    track_to_chan.setdefault(tid, (ch.text or "").strip())

    bed_labels: list[tuple[str, int, str | None]] = []  # (label, track, layout)
    hoa: list[tuple[int, int]] = []                     # (track, order guess)
    for num, uid, tref_chna, pref_chna in chna:
        tref, pref = uid_to_refs.get(uid, ("", ""))
        tref = tref or tref_chna
        pref = pref or pref_chna
        oname = objects_by_uid.get(uid, "")
        if pref in CD_PACK_LAYOUT:
            lab = CD_TRACK_LABEL.get(tref)
            if lab:
                bed_labels.append((lab, num, CD_PACK_LAYOUT[pref]))
            else:
                m.notes.append(f"track {num}: unknown CD track {tref}")
            continue
        if pref in CD_PACK_HOA:
            hoa.append((num, CD_PACK_HOA[pref]))
            continue
        if pref in CD_PACK_BINAURAL:
            m.binaural = True
            continue
        # custom pack: resolve type via the pack, channel via stream/track
        ptype = packs.get(pref, ("", "", []))[1]
        sid = track_to_chan.get(tref, "")
        cid = stream_to_chan.get(sid, "")
        cf = chans.get(cid)
        if ptype == "DirectSpeakers" or (
                cf is not None and cf.get("typeDefinition") == "DirectSpeakers"):
            label = None
            if cf is not None:
                for b in cf.iter():
                    if _local(b.tag) == "speakerLabel" and (b.text or "").strip():
                        label = canonical_label((b.text or "").strip())
                        break
                if label is None:
                    label = canonical_label(
                        cf.get("audioChannelFormatName", ""))
            if label and label in LABEL_POSITIONS:
                bed_labels.append((label, num, None))
            else:
                m.notes.append(f"track {num}: unmapped bed channel "
                               f"{cf.get('audioChannelFormatName', '?') if cf is not None else '?'}")
        elif ptype == "Objects" or (
                cf is not None and cf.get("typeDefinition") == "Objects"):
            name = oname or (cf.get("audioChannelFormatName", f"track{num}")
                             if cf is not None else f"track{num}")
            pts, has_gain = _block_points(cf) if cf is not None else ([], False)
            m.objects[num] = DeliveredObject(track=num, name=name,
                                             points=pts,
                                             has_gain_meta=has_gain)
        elif ptype == "HOA":
            hoa.append((num, 0))
        else:
            m.notes.append(f"track {num}: unclassified (pack {pref!r} "
                           f"type {ptype!r})")

    if bed_labels:
        m.bed_tracks = {lab: num for lab, num, _lay in bed_labels}
        declared = {lay for _l, _n, lay in bed_labels if lay}
        if len(declared) == 1:
            m.bed_layout = declared.pop()
        else:
            got = {lab for lab, _n, _l in bed_labels}
            for lay, labs in LAYOUT_LABELS.items():
                if got == set(labs):
                    m.bed_layout = lay
                    break
    if hoa:
        m.scene_tracks = [n for n, _o in sorted(hoa)]
        declared_orders = {o for _n, o in hoa if o}
        if len(declared_orders) == 1:
            m.scene_order = declared_orders.pop()
        else:
            n = len(m.scene_tracks)
            r = int(round(math.sqrt(n))) - 1
            m.scene_order = r if (r + 1) * (r + 1) == n else None
    return m


# ------------------------------------------------------------ measuring ----

def _rms_db(x: np.ndarray) -> float:
    e = float((x.astype(np.float64) ** 2).mean()) if x.size else 0.0
    return 10.0 * math.log10(e) if e > 0 else -120.0


def _active_fraction(x: np.ndarray, rate: int) -> float:
    w = max(1, int(rate * ACTIVITY_WIN_S))
    n = (len(x) // w) * w
    if n == 0:
        return 0.0
    wins = (x[:n].astype(np.float64) ** 2).reshape(-1, w).mean(axis=1)
    db = 10.0 * np.log10(np.maximum(wins, 1e-12))
    return float((db > ACTIVITY_FLOOR_DB).mean())


# --------------------------------------------------------------- result ----

@dataclass
class IntentCompareResult:
    sidecar: str
    delivered: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    execution_error: str | None = None

    def exit_code(self) -> int:
        if self.execution_error:
            return 2
        return 1 if any(f.severity >= Severity.FAIL for f in self.findings) else 0

    def to_dict(self) -> dict:
        return {
            "sentinel_intent_compare": "0.1.0",
            "sidecar": self.sidecar, "delivered": self.delivered,
            "result": "ERROR" if self.execution_error else
                      ("FAIL" if self.exit_code() else "PASS"),
            "execution_error": self.execution_error,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics, "exit_code": self.exit_code(),
        }


def _mk(check_id: str, message: str, **kw) -> Finding:
    s = spec(check_id)
    return Finding(check_id=check_id, severity=s.default_severity,
                   message=message, f_refs=s.f_refs, **kw)


def effective_tolerances(sidecar: dict) -> dict:
    t = dict(TOLERANCE_DEFAULTS)
    t["sliceSec"] = float(sidecar.get("program", {}).get("sliceSec", 0.1))
    for k, v in (sidecar.get("tolerances") or {}).items():
        if k in TOLERANCE_DEFAULTS:
            t[k] = float(v)
    eff = dict(t)
    eff["posDeg"] = max(t["posDeg"], TOLERANCE_FLOORS["posDeg"])
    eff["levelLu"] = max(t["levelLu"], TOLERANCE_FLOORS["levelLu"])
    eff["dominantFrac"] = min(t["dominantFrac"],
                              TOLERANCE_FLOORS["dominantFrac"])
    eff["covSilent"] = max(t["covSilent"], TOLERANCE_FLOORS["covSilent"])
    eff["silentDb"] = min(t["silentDb"], TOLERANCE_FLOORS["silentDb"])
    eff["sliceSec"] = max(t["sliceSec"], TOLERANCE_FLOORS["sliceSec"])
    eff["floors_applied"] = sorted(k for k in TOLERANCE_DEFAULTS
                                   if eff[k] != t[k])
    return eff


# ---------------------------------------------------------------- checks ---

def _slice_grid(start: float, end: float, slice_s: float) -> np.ndarray:
    n = max(1, int(round((end - start) / slice_s)))
    return start + (np.arange(n) + 0.5) * slice_s


def _check_shape(sc: dict, m: DeliveredModel, res, eff) -> bool:
    """S-346. Returns True when downstream checks may run."""
    prog = sc.get("program", {})
    problems = []
    sr = int(prog.get("sampleRate", 0))
    if sr and sr != m.sample_rate:
        problems.append(f"sampleRate {m.sample_rate} != predicted {sr}")
    start = float(prog.get("start", 0.0))
    end = float(prog.get("end", 0.0))
    if end > start and m.sample_rate:
        dur = m.frames / m.sample_rate
        if abs(dur - (end - start)) > DURATION_TOL_S:
            problems.append(f"duration {dur:.3f}s != predicted "
                            f"{end - start:.3f}s")
    bed = sc.get("bed")
    if bed and m.bed_layout:
        want = bed.get("layout", "")
        if want and want in LAYOUT_LABELS and want != m.bed_layout:
            problems.append(f"bed layout {m.bed_layout} != predicted {want}")
    scene = sc.get("scene")
    if scene and m.scene_order is not None:
        want_o = int(scene.get("order", 0))
        if want_o and want_o != m.scene_order:
            problems.append(f"scene order {m.scene_order} != predicted {want_o}")
    if problems:
        res.findings.append(_mk(
            "S-346",
            "delivered program shape contradicts the sidecar: "
            + "; ".join(problems),
            where="program",
            expected="sidecar and file describe the same program",
            found="; ".join(problems)))
        res.metrics["skipped"] = ["S-340", "S-341", "S-342", "S-343",
                                  "S-344", "S-345"]
        return False
    return True


def _check_roster(sc: dict, m: DeliveredModel, pcm, rate, res, eff) -> None:
    bed = sc.get("bed")
    if not bed:
        return
    roster = bed.get("roster") or []
    want = {canonical_label(str(r.get("channel", ""))) for r in roster}
    got = set(m.bed_tracks)
    missing = sorted(want - got)
    extra = sorted(got - want)
    res.metrics["bed"] = {"layout": m.bed_layout,
                          "roster_predicted": sorted(want),
                          "roster_delivered": sorted(got)}
    if roster and (missing or extra):
        res.findings.append(_mk(
            "S-340",
            f"bed roster mismatch: missing {missing or 'none'}, "
            f"unpredicted {extra or 'none'}",
            where=f"bed {bed.get('layout', '?')}",
            expected=f"roster {sorted(want)}",
            found=f"delivered {sorted(got)}"))
        return
    lu = eff["levelLu"]
    ch_metrics = {}
    for r in roster:
        lab = canonical_label(str(r.get("channel", "")))
        tr = m.bed_tracks.get(lab)
        if tr is None or tr > pcm.shape[1]:
            continue
        x = pcm[:, tr - 1]
        e = {"rms_db": round(_rms_db(x), 3),
             "active": round(_active_fraction(x, rate), 3)}
        devs = []           # one S-344 per entity, whichever measures deviate
        if r.get("expectRmsDb") is not None:
            d = e["rms_db"] - float(r["expectRmsDb"])
            e["rms_delta_db"] = round(d, 3)
            if abs(d) > lu:
                devs.append(f"RMS {e['rms_db']:+.2f} dBFS is {d:+.2f} dB "
                            f"from the predicted "
                            f"{float(r['expectRmsDb']):+.2f}")
        if r.get("expectLufs") is not None:
            ml = dsp.integrated_loudness(x[:, None], rate, weights=[1.0])
            d = ml - float(r["expectLufs"])
            e["lufs_measured"] = round(ml, 3)
            e["lufs_delta"] = round(d, 3)
            if abs(d) > lu:
                devs.append(f"{ml:+.2f} LKFS is {d:+.2f} LU from the "
                            f"predicted {float(r['expectLufs']):+.2f}")
        if devs:
            res.findings.append(_mk(
                "S-344",
                f"bed channel {lab}: " + "; ".join(devs)
                + f" (tolerance {lu})",
                where=f"bed channel {lab}",
                expected="measured level within tolerance of the sidecar",
                found="; ".join(devs)))
        ch_metrics[lab] = e
    res.metrics["bed"]["channels"] = ch_metrics
    if bed.get("expectLufs") is not None and m.bed_layout:
        order = [lab for lab in LAYOUT_LABELS[m.bed_layout]
                 if lab in m.bed_tracks]
        idx = [m.bed_tracks[lab] - 1 for lab in order]
        if idx and max(idx) < pcm.shape[1]:
            w = dsp.CHANNEL_WEIGHTS.get(m.bed_layout)
            ml = dsp.integrated_loudness(pcm[:, idx], rate, weights=w)
            d = ml - float(bed["expectLufs"])
            res.metrics["bed"]["lufs_measured"] = round(ml, 3)
            res.metrics["bed"]["lufs_delta"] = round(d, 3)
            if abs(d) > eff["levelLu"]:
                res.findings.append(_mk(
                    "S-344",
                    f"bed measures {ml:+.2f} LKFS, {d:+.2f} LU from the "
                    f"sidecar's {float(bed['expectLufs']):+.2f} "
                    f"(conformant weights, tolerance {eff['levelLu']} LU)",
                    where=f"bed {m.bed_layout}",
                    expected=f"{float(bed['expectLufs']):+.2f} LKFS",
                    found=f"{ml:+.2f} LKFS"))


def _check_scene(sc: dict, m: DeliveredModel, pcm, rate, res, eff) -> None:
    scene = sc.get("scene")
    if not scene:
        return
    exp = scene.get("expectRmsDb")
    sm = {"order": m.scene_order, "tracks": m.scene_tracks}
    res.metrics["scene"] = sm
    if not exp or not m.scene_tracks:
        return
    lu = eff["levelLu"]
    measured = []
    for i, tr in enumerate(m.scene_tracks):
        if tr > pcm.shape[1]:
            break
        r = _rms_db(pcm[:, tr - 1])
        measured.append(round(r, 3))
        if i < len(exp) and exp[i] is not None:
            want = float(exp[i])
            # both silent: fine regardless of dB arithmetic
            if want <= SLICE_SILENCE_DB and r <= SLICE_SILENCE_DB:
                continue
            if abs(r - want) > lu:
                res.findings.append(_mk(
                    "S-344",
                    f"scene ACN {i} measured RMS {r:+.2f} dBFS deviates "
                    f"{r - want:+.2f} dB from the sidecar's {want:+.2f} "
                    f"(tolerance {lu} dB)",
                    where=f"scene ACN {i}",
                    expected=f"{want:+.2f} dBFS",
                    found=f"{r:+.2f} dBFS"))
    sm["rms_db"] = measured


def _object_free_checks(sobj: dict, dobj, pcm, rate, prog, res, eff,
                        om: dict) -> bool:
    """S-341 + S-342 + S-344 halves that need no renderer.
    Returns False when the object is missing/silent (suppresses the rest)."""
    oid = sobj.get("id", "?")
    label = sobj.get("label", "")
    where = f"object {oid}" + (f" ({label})" if label else "")
    tr = int(sobj.get("track", 0))
    if dobj is None:
        res.findings.append(_mk(
            "S-341",
            f"predicted object is not present in the delivered file "
            f"(no object essence at track {tr})",
            where=where,
            expected=f"object audio + metadata at track {tr}",
            found="no such object track"))
        return False
    x = pcm[:, dobj.track - 1]
    act = _active_fraction(x, rate)
    rms = _rms_db(x)
    om["active_fraction"] = round(act, 3)
    om["rms_db"] = round(rms, 3)
    exp_act = float(sobj.get("expectActive", 0.0) or 0.0)
    if exp_act >= 0.5 and act < 0.1 * exp_act:
        lufs = dsp.integrated_loudness(x[:, None], rate, weights=[1.0])
        if lufs <= -69.99:
            res.findings.append(_mk(
                "S-341",
                f"predicted-active object is silent (active fraction "
                f"{act:.2f} vs predicted {exp_act:.2f}; {lufs:.1f} LKFS)",
                where=where,
                expected=f"active fraction ~{exp_act:.2f}",
                found=f"{act:.2f}, gated loudness {lufs:.1f} LKFS"))
            return False
    lu = eff["levelLu"]
    devs = []               # one S-344 per entity, whichever measures deviate
    if sobj.get("expectRmsDb") is not None:
        d = rms - float(sobj["expectRmsDb"])
        om["rms_delta_db"] = round(d, 3)
        if abs(d) > lu:
            devs.append(f"RMS {rms:+.2f} dBFS is {d:+.2f} dB from the "
                        f"predicted {float(sobj['expectRmsDb']):+.2f}")
    if sobj.get("expectLufs") is not None:
        ml = dsp.integrated_loudness(x[:, None], rate, weights=[1.0])
        d = ml - float(sobj["expectLufs"])
        om["lufs_measured"] = round(ml, 3)
        om["lufs_delta"] = round(d, 3)
        if abs(d) > lu:
            devs.append(f"{ml:+.2f} LKFS is {d:+.2f} LU from the "
                        f"predicted {float(sobj['expectLufs']):+.2f}")
    if devs:
        res.findings.append(_mk(
            "S-344",
            f"object stem: " + "; ".join(devs) + f" (tolerance {lu})",
            where=where,
            expected="measured stem level within tolerance of the sidecar",
            found="; ".join(devs)))

    spts = sidecar_trajectory(sobj)
    if spts and dobj.points:
        grid = _slice_grid(float(prog.get("start", 0.0)),
                           float(prog.get("end", 0.0)) or
                           (pcm.shape[0] / rate), eff["sliceSec"])
        errs = np.empty(len(grid))
        gains_s = np.empty(len(grid))
        gains_d = np.empty(len(grid))
        for i, t in enumerate(grid):
            sx, sy, sz, sg = sample_trajectory(spts, float(t))
            dx, dy, dz, dg = sample_trajectory(dobj.points, float(t))
            errs[i] = angular_error_deg((sx, sy, sz), (dx, dy, dz))
            gains_s[i] = max(sg, 1e-6)
            gains_d[i] = max(dg, 1e-6)
        p95 = float(np.percentile(errs, 95))
        om["trajectory_p95_deg"] = round(p95, 3)
        om["trajectory_max_deg"] = round(float(errs.max()), 3)
        if p95 > eff["posDeg"]:
            bad = np.nonzero(errs > eff["posDeg"])[0]
            t0 = float(grid[bad[0]]) - eff["sliceSec"] / 2
            t1 = float(grid[bad[-1]]) + eff["sliceSec"] / 2
            res.findings.append(_mk(
                "S-342",
                f"delivered trajectory diverges from the sidecar by "
                f"{p95:.1f}° (p95; max {errs.max():.1f}°) over "
                f"{t0:.2f}–{t1:.2f}s (tolerance {eff['posDeg']}°)",
                where=where,
                expected=f"angular error <= {eff['posDeg']}°",
                found=f"p95 {p95:.1f}°, max {errs.max():.1f}°"))
        gd = 20.0 * np.log10(gains_d / gains_s)
        gp95 = float(np.percentile(np.abs(gd), 95))
        om["gain_p95_db"] = round(gp95, 3)
        if (any(abs(p.gain - 1.0) > 1e-9 for p in spts)
                or dobj.has_gain_meta) and gp95 > lu:
            bad = np.nonzero(np.abs(gd) > lu)[0]
            t0 = float(grid[bad[0]]) - eff["sliceSec"] / 2
            t1 = float(grid[bad[-1]]) + eff["sliceSec"] / 2
            res.findings.append(_mk(
                "S-344",
                f"delivered gain trajectory deviates from the sidecar by "
                f"{gp95:.2f} dB (p95) over {t0:.2f}–{t1:.2f}s "
                f"(tolerance {lu} dB) — authored gain automation not carried",
                where=f"{where} gain trajectory",
                expected=f"|gain delta| <= {lu} dB",
                found=f"p95 {gp95:.2f} dB"))
    return True


def _object_render_checks(sobj: dict, dobj, delivered_path: str, all_tracks,
                          ear_bin: str, wd: str, prog, res, eff,
                          om: dict) -> None:
    """S-343 + S-345 via the per-object isolation render."""
    dec = sobj.get("decode") or {}
    layout = dec.get("layout", "")
    if layout not in EAR_LAYOUTS or layout not in LAYOUT_LABELS:
        res.metrics.setdefault("render_skipped", []).append(
            f"object {sobj.get('id', '?')}: unsupported decode layout "
            f"{layout!r}")
        return
    oid = sobj.get("id", "?")
    label = sobj.get("label", "")
    where = f"object {oid}" + (f" ({label})" if label else "")
    others = sorted(set(all_tracks) - {dobj.track})
    iso = os.path.join(wd, f"iso_{dobj.track}.wav")
    out = os.path.join(wd, f"iso_{dobj.track}_r.wav")
    silence_tracks(delivered_path, others, iso)
    ok, note = ear_render(ear_bin, iso, layout, out)
    if not ok:
        raise RuntimeError(f"EAR isolation render failed for track "
                           f"{dobj.track}: {note}")
    r, rrate = _read_wav(out)
    labels = LAYOUT_LABELS[layout]
    lfe = labels.index("LFE1") if "LFE1" in labels else None
    nch = min(r.shape[1], len(labels))
    w = max(1, int(rrate * eff["sliceSec"]))
    n = (r.shape[0] // w) * w
    if n == 0:
        return
    e = (r[:n] ** 2).reshape(-1, w, r.shape[1]).sum(axis=1)   # (slices, ch)
    e = e[:, :nch]
    if lfe is not None and lfe < nch:
        e[:, lfe] = 0.0
    tot = e.sum(axis=1)
    tot_db = 10.0 * np.log10(np.maximum(tot, 1e-12))
    live = tot_db > SLICE_SILENCE_DB

    # measured program coverage
    esum = e.sum(axis=0)
    etot = float(esum.sum())
    cov_meas = {labels[i]: float(esum[i] / etot) if etot > 0 else 0.0
                for i in range(nch)}
    om["coverage_measured"] = {k: round(v, 4) for k, v in
                               sorted(cov_meas.items()) if v >= 1e-4}

    # S-343 — dominant agreement on clear-dominance slices
    pred = dec.get("dominant") or []
    if pred:
        srt = np.sort(e, axis=1)
        with np.errstate(divide="ignore"):
            gap_db = 10.0 * np.log10(
                np.maximum(srt[:, -1], 1e-18) / np.maximum(srt[:, -2], 1e-18))
        clear = live & (gap_db >= DOMINANCE_GAP_DB)
        dom_idx = e.argmax(axis=1)
        agree = used = 0
        for i in range(min(len(pred), e.shape[0])):
            if not clear[i]:
                continue
            p = canonical_label(str(pred[i]))
            if p not in labels:
                continue
            used += 1
            if labels[dom_idx[i]] == p:
                agree += 1
        if used == 0:
            res.metrics.setdefault("render_skipped", []).append(
                f"object {oid}: no clear-dominance slices (S-343 skipped)")
        else:
            frac = agree / used
            om["dominant_agreement"] = round(frac, 3)
            om["dominant_slices_used"] = used
            if frac < eff["dominantFrac"]:
                res.findings.append(_mk(
                    "S-343",
                    f"rendered dominant speaker agrees with the sidecar on "
                    f"only {frac:.0%} of clear slices ({agree}/{used}; "
                    f"required {eff['dominantFrac']:.0%})",
                    where=where,
                    expected=f"agreement >= {eff['dominantFrac']:.0%}",
                    found=f"{frac:.0%} over {used} slices"))

    # S-345 — energy at predicted-silent channels
    cov_pred = dec.get("coverage") or {}
    if cov_pred:
        cs = eff["covSilent"]
        trigger = max(10.0 * cs, 0.05)
        pred_cov = {canonical_label(k): float(v) for k, v in cov_pred.items()}
        for i in range(nch):
            lab = labels[i]
            if lab == "LFE1":
                continue
            if pred_cov.get(lab, 0.0) < cs and cov_meas.get(lab, 0.0) > trigger:
                res.findings.append(_mk(
                    "S-345",
                    f"channel {lab} carries {cov_meas[lab]:.0%} of the "
                    f"object's rendered energy but the sidecar predicts "
                    f"{pred_cov.get(lab, 0.0):.1%} (< {cs:.1%}) — audio does "
                    f"not land where the session intended",
                    where=where,
                    expected=f"~no energy at {lab}",
                    found=f"{cov_meas[lab]:.0%} of rendered energy"))


# ------------------------------------------------------------------ main ---

def compare(sidecar_path: str, delivered_path: str, *,
            ear_bin: str | None = None,
            workdir: str | None = None) -> IntentCompareResult:
    res = IntentCompareResult(sidecar=sidecar_path, delivered=delivered_path)
    wd = workdir or tempfile.mkdtemp(prefix="sentinel_intc_")
    try:
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            sc = json.load(fh)
        if sc.get("intent") != SCHEMA_VERSION:
            res.execution_error = (f"unsupported sidecar schema "
                                   f"'intent': {sc.get('intent')!r} "
                                   f"(this consumer speaks {SCHEMA_VERSION})")
            return res
        eff = effective_tolerances(sc)
        res.metrics["tolerances_effective"] = {
            k: eff[k] for k in sorted(TOLERANCE_DEFAULTS)} | {
            "sliceSec": eff["sliceSec"],
            "floors_applied": eff["floors_applied"]}

        objects = sc.get("objects") or []
        needs_ear = any(o.get("decode") for o in objects)
        if needs_ear:
            ear_bin = ear_bin or find_ear()
            if ear_bin is None:
                res.execution_error = (
                    "the sidecar carries decode predictions and ear-render "
                    "was not found (set $SENTINEL_EAR) — refusing to report "
                    "a silently narrower comparison")
                return res

        m = parse_delivered(delivered_path)
        res.metrics["delivered"] = {
            "sample_rate": m.sample_rate, "tracks": m.n_tracks,
            "frames": m.frames, "bed_layout": m.bed_layout,
            "objects": sorted(m.objects), "scene_tracks": m.scene_tracks}
        if m.notes:
            res.metrics["delivered"]["notes"] = m.notes

        if not _check_shape(sc, m, res, eff):
            return res

        pcm, rate = _read_wav(delivered_path)
        prog = sc.get("program", {})

        _check_roster(sc, m, pcm, rate, res, eff)
        _check_scene(sc, m, pcm, rate, res, eff)

        all_tracks = list(range(1, m.n_tracks + 1))
        obj_metrics = {}
        for sobj in objects:
            key = f"object {sobj.get('id', '?')}"
            om: dict = {}
            obj_metrics[key] = om
            dobj = m.objects.get(int(sobj.get("track", 0)))
            alive = _object_free_checks(sobj, dobj, pcm, rate, prog, res,
                                        eff, om)
            if alive and sobj.get("decode"):
                _object_render_checks(sobj, dobj, delivered_path, all_tracks,
                                      ear_bin, wd, prog, res, eff, om)
        if obj_metrics:
            res.metrics["objects"] = obj_metrics
        return res
    except Exception as e:               # CI contract: never crash
        res.execution_error = f"{type(e).__name__}: {e}"
        return res
    finally:
        if workdir is None:
            shutil.rmtree(wd, ignore_errors=True)


def render_json(res: IntentCompareResult, indent: int = 2) -> str:
    return json.dumps(res.to_dict(), indent=indent)


def render_text(res: IntentCompareResult) -> str:
    d = res.to_dict()
    lines = [f"sentinel intent-compare — {d['result']}",
             f"  sidecar   : {res.sidecar}",
             f"  delivered : {res.delivered}"]
    if res.execution_error:
        lines.append(f"  error     : {res.execution_error}")
    for f in res.findings:
        lines.append(f"  [{f.severity.label}] {f.check_id}: {f.message}")
    if not res.findings and not res.execution_error:
        lines.append("  no findings — the file renders as the session "
                     "intended (within tolerances)")
    lines.append(f"  metrics: {json.dumps(res.metrics)}")
    return "\n".join(lines)
