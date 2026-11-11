"""ADM-to-IAMF conversion fidelity (IA-0076): S-333 level, S-334 direction.

adm_compare's S-330/S-332 judge the produced IAMF's level envelope summed over
every channel of a rendered layout, so a bed whose corners arrive 3 dB down
against its centre passes them: the sum barely moves, and a render confounds
level with direction. These two checks judge every source track on its own.

Reference-free -- no renderer, no decoder. iamf-tools' ADM import writes LPCM
(its user_metadata_generator asks for an LPCM codec config), so the converted
signals are read straight out of the IAMF, and each source track's
contribution to every coded channel is solved by least squares against the
source's own tracks. An IAMF that is not LPCM is reported as not judged.

The reference is the standard, not the converter (s318 ia0057 §6.2): a check
built to the converter's own rule would certify its faults.
  level      a static block arrives at block gain x audioObject gain,
             wherever it sits. BS.2127 has no distance-to-level term, and
             OAR 1.0.0 §4.1.3/§4.2.3 renders an object at any distance alike.
  direction  the authored azimuth/elevation for a polar block; for a
             Cartesian block, BS.2127-0 §10.1.2 point_cart_to_polar (the
             cube corner (1, 1, 0) is azimuth -30, not -45).
Readout. In a scene-based element (SN3D, ACN) a point source at unit gain
puts exactly its signal on ACN 0 from any direction, so the coefficient on
W is the track's gain, and ACN 1/2/3 (Y, Z, X) give its direction. In a
channel-based element (an LFE split off, a Default-ADM passthrough) the
coefficient norm over the element's channels is the gain; no direction is
judged there. Not judged: a channel format with more than one block
(movement), HOA and Binaural tracks, a channel format the axml does not
define (a common definition), a silent track.

IP boundary: original work, written from BS.2076-2 (axml), EBU Tech 3285 /
BS.2088 (chna), BS.2127-0 §6.2, §6.7, §10.1 (read 2026-11-11 from the ITU
PDF) and the IAMF specification (OBU and LPCM framing).
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from sentinel.findings import Finding, Severity, spec
from sentinel.parser import parse_bytes

from .adm_compare import _find_chunks, _read_wav

# Provisional thresholds -- the rulings are James's (D128); the readout on
# conforming LPCM content is exact to 24-bit quantisation (0.00 dB, 0.0 deg).
LEVEL_TOL_DB = 1.0        # S-333: |realised - expected| level of any judged track
DIRECTION_TOL_DEG = 5.0   # S-334: great-circle error of any judged track
MAX_SECONDS = 20.0        # the solve uses at most this much audio
SILENT_RMS = 1e-5         # a source track quieter than this is not judged
COND_LIMIT = 1e6          # source tracks this collinear cannot be separated

THETA_TOP, THETA_TOP_C, EPS = 30.0, 45.0, 1e-10


# --------------------------------------------------- BS.2127-0 §6.2, §6.7, §10.1

def relative_angle(x: float, y: float) -> float:
    """§6.7: y + 360n for the smallest n with the result >= x."""
    while y < x:
        y += 360.0
    while y - 360.0 >= x:
        y -= 360.0
    return y


def inside_angle_range(x: float, start: float, end: float, tol: float = 0.0) -> bool:
    """§6.2: whether x lies on the arc from start anticlockwise to end, widened by tol."""
    if end - start >= 360.0 - 2 * tol:      # the whole circle (e.g. -180..180)
        return True
    end = relative_angle(start, end)
    x = relative_angle(start - tol, x)
    return x <= end + tol


# (phi_l, phi_r, (x_l, y_l), (x_r, y_r)) and the azimuth range each serves, §10.1
_SECTORS = [
    (30.0, 0.0, (-1.0, 1.0), (0.0, 1.0)),
    (0.0, -30.0, (0.0, 1.0), (1.0, 1.0)),
    (-30.0, -110.0, (1.0, 1.0), (1.0, -1.0)),
    (-110.0, 110.0, (1.0, -1.0), (-1.0, -1.0)),
    (110.0, 30.0, (-1.0, -1.0), (-1.0, 1.0)),
]
_CART_RANGES = [(0.0, 45.0), (-45.0, 0.0), (-135.0, -45.0), (135.0, -135.0), (45.0, 135.0)]


def _find_cart_sector(phi: float):
    for sec, (a, b) in zip(_SECTORS, _CART_RANGES):
        if inside_angle_range(phi, a, b):
            return sec
    raise ValueError(phi)


def _map_linear_to_az(phi_l: float, phi_r: float, x: float) -> float:
    mid = (phi_l + phi_r) / 2.0
    rng = phi_r - mid
    gl, gr = math.cos(x * math.pi / 2), math.sin(x * math.pi / 2)
    g_r = gr / (gl + gr)
    rel = math.degrees(math.atan(2 * (g_r - 0.5) * math.tan(math.radians(rng))))
    return mid + rel


def point_cart_to_polar(x: float, y: float, z: float) -> tuple[float, float, float]:
    """§10.1.2: (azimuth, elevation, distance) of a BS.2076 Cartesian position."""
    if abs(x) < EPS and abs(y) < EPS:
        if abs(z) < EPS:
            return 0.0, 0.0, 0.0
        return 0.0, math.copysign(90.0, z), abs(z)
    phi_c = -math.degrees(math.atan2(x, y))
    phi_l, phi_r, (xl, yl), (xr, yr) = _find_cart_sector(phi_c)
    det = xl * yr - yl * xr
    g_l = (x * yr - y * xr) / det          # [x y] . inv([[xl yl], [xr yr]])
    g_r = (-x * yl + y * xl) / det
    r_xy = g_l + g_r
    phi_l2 = relative_angle(phi_r, phi_l)
    phi = relative_angle(-180.0, _map_linear_to_az(phi_l2, phi_r, g_r / r_xy))
    if phi > 180.0:
        phi -= 360.0
    th_c = math.degrees(math.atan(z / r_xy))
    if abs(th_c) > THETA_TOP_C:
        th = THETA_TOP + (90.0 - THETA_TOP) * (abs(th_c) - THETA_TOP_C) / (90.0 - THETA_TOP_C)
        return phi, math.copysign(th, th_c), abs(z)
    return phi, th_c * THETA_TOP / THETA_TOP_C, r_xy


def _unit(az: float, el: float) -> np.ndarray:
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])


def angle_between(az1: float, el1: float, az2: float, el2: float) -> float:
    c = float(np.clip(_unit(az1, el1) @ _unit(az2, el2), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ------------------------------------------------------------- the source ADM

@dataclass
class SourceTrack:
    track: int                      # 1-based BW64 track
    uid: str
    channel_format: str | None
    name: str = ""
    type_def: str = ""              # DirectSpeakers / Objects / HOA / Binaural / ...
    blocks: int = 0
    gain: float = 1.0               # block gain x object gain, linear
    polar: tuple[float, float] | None = None       # expected (az, el)
    cart: tuple[float, float, float] | None = None
    is_lfe: bool = False
    not_judged: str | None = None


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _gain_of(el) -> float:
    g = None
    for c in el:
        if _ln(c.tag) == "gain":
            g = c
            break
    if g is None or not (g.text or "").strip():
        return 1.0
    v = float(g.text)
    return 10 ** (v / 20) if (g.get("gainUnit") or "").lower() == "db" else v


_TYPE_LABELS = {"0001": "DirectSpeakers", "0002": "Matrix", "0003": "Objects",
                "0004": "HOA", "0005": "Binaural"}


def read_source(path: str) -> list[SourceTrack]:
    with open(path, "rb") as fh:
        data = fh.read()
    ch = _find_chunks(data)
    if b"chna" not in ch or b"axml" not in ch:
        raise ValueError("source has no chna/axml chunk -- not an ADM BW64")
    off, size = ch[b"chna"]
    body = data[off:off + size]
    n_uids = struct.unpack("<H", body[2:4])[0]
    entries = []
    for i in range(n_uids):
        e = body[4 + 40 * i: 4 + 40 * (i + 1)]
        if len(e) < 40:
            break
        tn = struct.unpack("<H", e[0:2])[0]
        if tn == 0:
            continue
        uid = e[2:14].decode("ascii", "replace").strip("\x00 ")
        tref = e[14:28].decode("ascii", "replace").strip("\x00 ")
        entries.append((tn, uid, tref))
    off, size = ch[b"axml"]
    root = ET.fromstring(data[off:off + size].decode("utf-8", "replace"))
    by_tag: dict[str, dict[str, object]] = {}
    for el in root.iter():
        t = _ln(el.tag)
        idattr = el.get(f"{t}ID")
        if idattr:
            by_tag.setdefault(t, {})[idattr] = el

    def ref(el, name):
        for c in el:
            if _ln(c.tag) == name:
                return (c.text or "").strip()
        return None

    obj_gain: dict[str, float] = {}
    for obj in by_tag.get("audioObject", {}).values():
        g = _gain_of(obj)
        for c in obj:
            if _ln(c.tag) == "audioTrackUIDRef":
                obj_gain[(c.text or "").strip()] = g
    tracks = []
    for tn, uid, tref in entries:
        uel = by_tag.get("audioTrackUID", {}).get(uid)
        acf = None
        if tref.startswith("AC_"):
            acf = tref
        elif uel is not None and ref(uel, "audioChannelFormatIDRef"):
            acf = ref(uel, "audioChannelFormatIDRef")
        else:
            atf = tref if tref.startswith("AT_") else (ref(uel, "audioTrackFormatIDRef") if uel is not None else None)
            at = by_tag.get("audioTrackFormat", {}).get(atf or "")
            asf = by_tag.get("audioStreamFormat", {}).get(ref(at, "audioStreamFormatIDRef") or "") if at is not None else None
            acf = ref(asf, "audioChannelFormatIDRef") if asf is not None else None
        st = SourceTrack(tn, uid, acf)
        cf = by_tag.get("audioChannelFormat", {}).get(acf or "")
        if cf is None:
            st.not_judged = "channel format not defined in the axml (a common definition?)"
            tracks.append(st)
            continue
        st.name = cf.get("audioChannelFormatName", "")
        st.type_def = cf.get("typeDefinition") or _TYPE_LABELS.get(cf.get("typeLabel", ""), "")
        blocks = [b for b in cf if _ln(b.tag) == "audioBlockFormat"]
        st.blocks = len(blocks)
        for c in cf:
            if _ln(c.tag) == "frequency" and c.get("typeDefinition") == "lowPass":
                try:
                    st.is_lfe = float(c.text) <= 200.0
                except (TypeError, ValueError):
                    pass
        if "LFE" in st.name.upper():
            st.is_lfe = True
        if st.type_def not in ("DirectSpeakers", "Objects"):
            st.not_judged = f"{st.type_def or 'unknown'} track"
        elif st.blocks != 1:
            st.not_judged = f"{st.blocks} blocks (movement is not judged)"
        if st.blocks:
            b = blocks[0]
            st.gain = _gain_of(b) * obj_gain.get(uid, 1.0)
            pos = {}
            cart_flag = False
            for c in b:
                t = _ln(c.tag)
                if t == "position":
                    try:
                        pos[c.get("coordinate")] = float(c.text)
                    except (TypeError, ValueError):
                        pass
                elif t == "cartesian":
                    cart_flag = (c.text or "").strip() == "1"
                elif t == "speakerLabel" and "LFE" in (c.text or "").upper():
                    st.is_lfe = True
            if cart_flag or ({"X", "Y"} <= set(pos) and "azimuth" not in pos):
                st.cart = (pos.get("X", 0.0), pos.get("Y", 0.0), pos.get("Z", 0.0))
                az, el, _d = point_cart_to_polar(*st.cart)
                st.polar = (az, el)
            elif "azimuth" in pos:
                st.polar = (pos["azimuth"], pos.get("elevation", 0.0))
            elif st.not_judged is None and not st.is_lfe:
                st.not_judged = "block has no position"
        tracks.append(st)
    return tracks


# ------------------------------------------------------- the IAMF's LPCM audio

def _leb(b: bytes, p: int) -> tuple[int, int]:
    v = s = 0
    while True:
        x = b[p]
        p += 1
        v |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return v, p


def _frame_payload(data: bytes, fr) -> bytes:
    size, p = _leb(data, fr.file_offset + 1)   # obu_size follows the 1-byte header
    end = p + size
    return data[end - fr.frame_bytes:end]


def _lpcm_format(data: bytes, cc) -> tuple[bool, int]:
    p = cc.header.payload_offset
    _, p = _leb(data, p)
    p += 4                                  # codec_id
    _, p = _leb(data, p)                    # num_samples_per_frame
    p += 2                                  # audio_roll_distance
    flags, bits = data[p], data[p + 1]
    return bool(flags & 1), bits


def _decode_pcm(raw: bytes, bits: int, little: bool, ch: int) -> np.ndarray:
    nb = bits // 8
    n = len(raw) // (nb * ch)
    a = np.frombuffer(raw[:n * nb * ch], dtype=np.uint8).reshape(-1, nb).astype(np.int64)
    order = range(nb) if little else range(nb - 1, -1, -1)
    v = np.zeros(a.shape[0], dtype=np.int64)
    for k, idx in enumerate(order):
        v |= a[:, idx] << (8 * k)
    v -= (v >> (bits - 1)) << bits
    return (v.astype(np.float64) / float(1 << (bits - 1))).reshape(n, ch)


@dataclass
class CodedElement:
    element_id: int
    kind: str                           # "scene" | "channel"
    channels: np.ndarray                # [n, c]
    acn: list[int] = field(default_factory=list)   # scene: ACN of each column


def read_iamf_lpcm(path: str) -> tuple[list[CodedElement], int, str | None]:
    """-> (elements, sample_rate, reason-not-judged). Scene elements in mono
    mode and single-layer channel elements; anything else is skipped."""
    with open(path, "rb") as fh:
        data = fh.read()
    m = parse_bytes(data, source=path)
    if not m.codec_configs or any(c.codec_id != "ipcm" for c in m.codec_configs.values()):
        return [], 0, "the IAMF is not LPCM; the conversion readout needs uncompressed audio"
    per_sub: dict[int, list[np.ndarray]] = {}
    sub_meta: dict[int, tuple[int, bool, int]] = {}      # sid -> (channels, little, bits)
    out: list[CodedElement] = []
    rate = 0
    for ae in m.audio_elements.values():
        cc = m.codec_configs[ae.codec_config_id]
        little, bits = _lpcm_format(data, cc)
        rate = cc.sample_rate or rate
        if ae.ambisonics is not None:
            if ae.ambisonics.mode != 0:
                continue
            for sid in ae.audio_substream_ids:
                sub_meta[sid] = (1, little, bits)
        elif ae.channel_layers and len(ae.channel_layers) == 1:
            lay = ae.channel_layers[0]
            for i, sid in enumerate(ae.audio_substream_ids):
                sub_meta[sid] = (2 if i < lay.coupled_substream_count else 1, little, bits)
    for fr in m.audio_frames:
        meta = sub_meta.get(fr.substream_id)
        if meta is None:
            continue
        x = _decode_pcm(_frame_payload(data, fr), meta[2], meta[1], meta[0])
        if fr.trim_start or fr.trim_end:
            x = x[fr.trim_start: x.shape[0] - fr.trim_end]
        per_sub.setdefault(fr.substream_id, []).append(x)
    for ae in m.audio_elements.values():
        cols, acns = [], []
        if ae.ambisonics is not None and ae.ambisonics.mode == 0:
            amap = ae.ambisonics.channel_mapping
            for acn, idx in enumerate(amap):
                if idx == 255 or idx >= len(ae.audio_substream_ids):
                    continue
                sid = ae.audio_substream_ids[idx]
                if sid in per_sub:
                    cols.append(np.concatenate(per_sub[sid])[:, 0])
                    acns.append(acn)
            kind = "scene"
        elif ae.channel_layers and len(ae.channel_layers) == 1:
            for sid in ae.audio_substream_ids:
                if sid in per_sub:
                    s = np.concatenate(per_sub[sid])
                    cols.extend(s[:, j] for j in range(s.shape[1]))
            kind = "channel"
        else:
            continue
        if cols:
            n = min(len(c) for c in cols)
            out.append(CodedElement(ae.audio_element_id, kind,
                                    np.stack([c[:n] for c in cols], axis=1), acns))
    if not out:
        return [], rate, "no scene-based (mono mode) or single-layer channel-based element"
    return out, rate, None


# ----------------------------------------------------------------- the checks

@dataclass
class TrackResult:
    track: int
    name: str
    expected_gain_db: float | None = None
    realised_gain_db: float | None = None
    level_error_db: float | None = None
    expected_dir: tuple[float, float] | None = None
    realised_dir: tuple[float, float] | None = None
    direction_error_deg: float | None = None
    lands_in: str = ""
    not_judged: str | None = None

    def to_dict(self) -> dict:
        r = lambda v, k=2: None if v is None else round(v, k)   # noqa: E731
        return {"track": self.track, "name": self.name,
                "expected_gain_db": r(self.expected_gain_db),
                "realised_gain_db": r(self.realised_gain_db),
                "level_error_db": r(self.level_error_db),
                "expected_az_el": None if self.expected_dir is None else [r(v, 1) for v in self.expected_dir],
                "realised_az_el": None if self.realised_dir is None else [r(v, 1) for v in self.realised_dir],
                "direction_error_deg": r(self.direction_error_deg, 1),
                "lands_in": self.lands_in, "not_judged": self.not_judged}


def _db(v: float) -> float:
    return 20.0 * math.log10(max(abs(v), 1e-12))


def _lag(src: np.ndarray, y: np.ndarray, rate: int) -> int:
    """Samples by which y lags src (cross-correlation of the strongest coded
    channel against the sum of the sources, over the first 4 s)."""
    from scipy.signal import correlate
    m = min(len(src), len(y), 4 * rate)
    a = src[:m].sum(axis=1)
    b = y[:m, int(np.argmax((y[:m] ** 2).sum(axis=0)))]
    if not a.any() or not b.any():
        return 0
    c = correlate(b, a, mode="full", method="fft")
    return int(np.argmax(np.abs(c))) - (m - 1)


def conversion_fidelity(source_adm: str, output_iamf: str) -> tuple[list[Finding], dict]:
    """-> (findings, metrics). Raises nothing for content it cannot judge: the
    reason is in metrics["not_judged"]."""
    tracks = read_source(source_adm)
    elements, rate, why = read_iamf_lpcm(output_iamf)
    metrics: dict = {"tracks": [], "level_tolerance_db": LEVEL_TOL_DB,
                     "direction_tolerance_deg": DIRECTION_TOL_DEG}
    if why:
        metrics["not_judged"] = why
        return [], metrics
    src, srate = _read_wav(source_adm)
    if srate != rate:
        metrics["not_judged"] = f"source rate {srate} vs IAMF rate {rate}"
        return [], metrics
    src = src.astype(np.float64)
    y = np.concatenate([e.channels for e in elements], axis=1)
    lag = _lag(src, y, rate)
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        src = src[-lag:]
    n = min(len(src), len(y), int(MAX_SECONDS * rate))
    src, y = src[:n], y[:n]
    live = [t for t in tracks if 1 <= t.track <= src.shape[1]
            and float(np.sqrt(np.mean(src[:, t.track - 1] ** 2))) > SILENT_RMS]
    cols = sorted({t.track - 1 for t in live})
    metrics.update({"lag_samples": lag, "seconds": round(n / rate, 3),
                    "elements": [{"id": e.element_id, "kind": e.kind,
                                  "channels": int(e.channels.shape[1])} for e in elements]})
    if not cols:
        metrics["not_judged"] = "every source track is silent"
        return [], metrics
    S = src[:, cols]
    cond = float(np.linalg.cond(S))
    metrics["source_condition"] = round(cond, 1)
    if cond > COND_LIMIT:
        metrics["not_judged"] = "the source tracks are too alike to separate"
        return [], metrics
    A, *_ = np.linalg.lstsq(S, y, rcond=None)              # [k, coded channel]
    resid = y - S @ A
    metrics["unexplained_energy_db"] = round(
        10 * math.log10(max(float((resid ** 2).sum()), 1e-30) / max(float((y ** 2).sum()), 1e-30)), 1)
    # column bookkeeping
    spans, c0 = [], 0
    for e in elements:
        spans.append((e, c0, c0 + e.channels.shape[1]))
        c0 += e.channels.shape[1]
    row = {c: i for i, c in enumerate(cols)}
    level_bad, dir_bad = [], []
    for t in tracks:
        r = TrackResult(t.track, t.name or t.uid)
        if t.track - 1 not in row:
            r.not_judged = t.not_judged or "silent or outside the data chunk"
            metrics["tracks"].append(r.to_dict())
            continue
        a = A[row[t.track - 1]]
        energy = 0.0
        w = None
        where = []
        for e, lo, hi in spans:
            if e.kind == "scene":
                if 0 in e.acn:
                    w_e = a[lo + e.acn.index(0)]
                    if abs(w_e) > 1e-4:
                        w = (e, lo, w_e)
                        where.append(f"scene {e.element_id}")
                    energy += w_e ** 2
            else:
                eg = float((a[lo:hi] ** 2).sum())
                if eg > 1e-8:
                    where.append(f"channels {e.element_id}")
                energy += eg
        r.lands_in = ", ".join(where) or "nowhere"
        r.not_judged = t.not_judged
        r.expected_gain_db = _db(t.gain)
        r.realised_gain_db = _db(math.sqrt(energy))
        r.level_error_db = r.realised_gain_db - r.expected_gain_db
        if t.not_judged is None and abs(r.level_error_db) >= LEVEL_TOL_DB:
            level_bad.append(r)
        if w is not None and not t.is_lfe and t.polar is not None:
            e, lo, w_e = w
            comp = {acn: a[lo + i] for i, acn in enumerate(e.acn)}
            if all(k in comp for k in (1, 2, 3)):
                yv, zv, xv = comp[1] / w_e, comp[2] / w_e, comp[3] / w_e
                r.expected_dir = t.polar
                r.realised_dir = (math.degrees(math.atan2(yv, xv)),
                                  math.degrees(math.atan2(zv, math.hypot(xv, yv))))
                r.direction_error_deg = angle_between(*r.expected_dir, *r.realised_dir)
                if t.not_judged is None and r.direction_error_deg >= DIRECTION_TOL_DEG:
                    dir_bad.append(r)
        metrics["tracks"].append(r.to_dict())
    findings: list[Finding] = []
    if level_bad:
        worst = max(level_bad, key=lambda r: abs(r.level_error_db))
        findings.append(_mk(
            "S-333",
            f"{len(level_bad)} source track(s) arrive at a level other than the master "
            f"gives them (worst {worst.name}: {worst.level_error_db:+.2f} dB; tolerance "
            f"{LEVEL_TOL_DB} dB) -- the conversion changed their balance",
            where=", ".join(f"{r.name} {r.level_error_db:+.2f} dB" for r in level_bad[:8]),
            expected="each track at block gain x object gain, wherever it sits (BS.2127)",
            found=f"{worst.name} realised {worst.realised_gain_db:+.2f} dB against "
                  f"{worst.expected_gain_db:+.2f} dB"))
    if dir_bad:
        worst = max(dir_bad, key=lambda r: r.direction_error_deg)
        findings.append(_mk(
            "S-334",
            f"{len(dir_bad)} source track(s) arrive away from their authored direction "
            f"(worst {worst.name}: {worst.direction_error_deg:.1f} deg; tolerance "
            f"{DIRECTION_TOL_DEG} deg)",
            where=", ".join(f"{r.name} {r.direction_error_deg:.1f} deg" for r in dir_bad[:8]),
            expected=f"{worst.name} at az {worst.expected_dir[0]:.1f}, el "
                     f"{worst.expected_dir[1]:.1f} (BS.2127-0 §10.1.2 for Cartesian blocks)",
            found=f"az {worst.realised_dir[0]:.1f}, el {worst.realised_dir[1]:.1f}"))
    return findings, metrics


def _mk(check_id: str, message: str, **kw) -> Finding:
    s = spec(check_id)
    return Finding(check_id=check_id, severity=s.default_severity,
                   message=message, f_refs=s.f_refs, **kw)
