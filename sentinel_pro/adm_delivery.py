"""Measured delivery checks for Atmos ADM masters, for `sentinel adm-check`.

iamf-sentinel's `adm-check` judges every rule its file's metadata settles and
reports the rules that need the audio as UNDETERMINED. When this plugin is
installed it calls `measure_master()` here, which reads the PCM once, and
fills those rows:

  S-522  the LFE channel is low-passed. Delivery Specification v2022.07: a
         SHOULD, "cut-off between 100 and 150 Hz", no slope given. Judged
         against a stated filter shape (`LFE_REF_*`), not a single number.
  S-524  every channel carries signal: none digitally silent (every sample
         zero) and none near-silent (peak below `NEAR_SILENT_DBFS`). Label
         delivery QC practice; not a rule of either Dolby document.
  S-525  a 96 kHz master holds content a 48 kHz recording could not: energy
         above 24 kHz without a brick-wall edge near it, which is how an
         upsampled file gives itself away. Label QC practice; not Dolby's.
  S-523  in sync with the stereo master (Delivery Specification, a MUST): the
         offset is found by cross-correlating a mono fold-down of the master
         (every channel but the LFE) against the stereo's L+R; then no extra
         lead silence (beyond `LEAD_TOL_MS`) and a tail under one frame.

`gapless_boundaries()` turns the per-track S-523 measurements of an album
into its S-533 row (boundaries within half a frame of the stereo's).

Every threshold named in capitals below that is not a number of the
specification itself is provisional until ruled (decision D128: thresholds
are the owner's), and each row names the threshold it was judged with.
The specification's own numbers are not thresholds of ours: the 100-150 Hz
range, one frame at 24 fps (2,000 samples at 48 kHz) and half a frame.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterator

import numpy as np

from .adm_conversion import parse_tracks

API = 1                     # what iamf-sentinel's adm-check checks for

PASS, FAIL, WARN, UNDET = "PASS", "FAIL", "WARN", "UNDETERMINED"

# -- S-522: the LFE low-pass (provisional) ------------------------------------
LFE_REF_FC_HZ = 150.0       # the top of the specification's 100-150 Hz range
LFE_REF_ORDER = 2           # Butterworth order: 2 = 12 dB/octave, the gentlest common slope
LFE_TOL_DB = 3.0            # allowed excess over that shape, per 1/6-octave band
LFE_FLOOR_DB = -60.0        # the shape is not followed below this (relative to the passband)
LFE_PASS_BAND = (20.0, 80.0)    # where the LFE's own level is read
LFE_JUDGE_BAND = (200.0, 4000.0)
LFE_MIN_DBFS = -80.0        # passband level below this: too little bass to judge a filter

# -- S-524: silent channels (provisional) -------------------------------------
NEAR_SILENT_DBFS = -80.0    # a channel whose peak stays below this is near-silent

# -- S-525: a false 96 kHz (provisional) --------------------------------------
HI_BAND = (26000.0, 32000.0)    # above anything a 48 kHz recording can hold
EDGE_BAND = (18000.0, 21000.0)  # below a 48 kHz recording's band edge
CONTENT_OVER_FLOOR_DB = 20.0    # HI_BAND this far above the format's quantization floor = content
JUDGEABLE_OVER_FLOOR_DB = 40.0  # EDGE_BAND this far above the floor = enough top end to tell
BRICKWALL_DB = 50.0             # EDGE_BAND over HI_BAND by this much = a band-edge cliff

# -- S-523: the stereo master (lead tolerance provisional) --------------------
LEAD_TOL_MS = 1.0           # extra lead (or lag) tolerated, as the correlation's resolution
MIN_CORRELATION = 0.5       # weaker than this: the two do not align well enough to judge
FRAMES_PER_SECOND = 24      # the specification's frame (it asks for 24 fps timecode)

_BLOCK = 1 << 16            # frames read per block


class NotJudged(Exception):
    """The audio cannot be read for measurement; the message says why."""


# ------------------------------------------------------------------ reading
def _data_chunk(info: dict[str, Any]) -> tuple[int, int]:
    for c in info.get("chunks", []):
        if c["id"] == "data":
            start = c["offset"] + 8
            return start, max(0, min(c["size"], info.get("fileSize", 0) - start))
    raise NotJudged("no data chunk")


def _fmt(info: dict[str, Any]) -> tuple[int, int, int]:
    f = info.get("format") or {}
    ch, rate, bits = f.get("channels"), f.get("sampleRate"), f.get("bitDepth")
    if f.get("formatTag") not in (1, 0xFFFE) or bits not in (16, 24, 32) or not ch or not rate:
        raise NotJudged(f"PCM format not read (format tag {f.get('formatTag')}, "
                        f"{bits}-bit): only 16/24/32-bit integer PCM is measured")
    return ch, rate, bits


def _decode(raw: bytes, bits: int, ch: int) -> np.ndarray:
    """Integer PCM -> float32 (frames, ch) on [-1, 1)."""
    if bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        x = (np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = (v << 8) >> 8
        x = v.astype(np.float32) / 8388608.0
    return x.reshape(-1, ch)


def blocks(path: str, info: dict[str, Any], frames: int = _BLOCK) -> Iterator[np.ndarray]:
    ch, _rate, bits = _fmt(info)
    start, size = _data_chunk(info)
    ba = ch * bits // 8
    left = size // ba
    with open(path, "rb") as f:
        f.seek(start)
        while left > 0:
            n = min(frames, left)
            raw = f.read(n * ba)
            n = len(raw) // ba
            if n == 0:
                break
            yield _decode(raw[:n * ba], bits, ch)
            left -= n


def _chunk_bytes(path: str, info: dict[str, Any], cid: str) -> bytes | None:
    for c in info.get("chunks", []):
        if c["id"] == cid:
            with open(path, "rb") as f:
                f.seek(c["offset"] + 8)
                return f.read(c["size"])
    return None


def track_labels(path: str, info: dict[str, Any]) -> tuple[list[str], list[int]]:
    """(a label per channel, the 0-based indices of LFE channels)."""
    ch = (info.get("format") or {}).get("channels") or 0
    labels = [f"track {i + 1}" for i in range(ch)]
    lfe: list[int] = []
    chna, axml = _chunk_bytes(path, info, "chna"), _chunk_bytes(path, info, "axml")
    tracks = []
    if chna and axml:
        try:
            tracks = parse_tracks(chna, axml)
        except Exception:               # an unreadable axml leaves the plain labels
            tracks = []
    for t in tracks:
        i = t.track - 1
        if not 0 <= i < ch:
            continue
        if t.name:
            labels[i] = f"track {t.track} ({t.name})"
        # AC_00010004 is the common-definitions LFE (BS.2094-1).
        if t.is_lfe or (t.channel_format or "").upper() == "AC_00010004":
            lfe.append(i)
    # A track that names the common-definitions LFE in chna (AT_00010004_nn)
    # whose formats the axml leaves to the common definitions (BS.2094-1).
    for t in (info.get("chna") or {}).get("tracks", []):
        if t.get("trackRef", "").upper().startswith("AT_00010004") and 0 < t["trackIndex"] <= ch:
            lfe.append(t["trackIndex"] - 1)
            if labels[t["trackIndex"] - 1] == f"track {t['trackIndex']}":
                labels[t["trackIndex"] - 1] += " (LFE, common definition)"
    return labels, sorted(set(lfe))


# ------------------------------------------------------------- one pass
def _welch(x: np.ndarray, rate: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD (power per Hz) of a 1-D signal: Hann, 50 % overlap."""
    w = np.hanning(nfft).astype(np.float64)
    hop = nfft // 2
    acc = np.zeros(nfft // 2 + 1)
    n = 0
    if len(x) < nfft:
        x = np.concatenate([x, np.zeros(nfft - len(x), dtype=x.dtype)])
    for s in range(0, len(x) - nfft + 1, hop):
        acc += np.abs(np.fft.rfft(x[s:s + nfft] * w)) ** 2
        n += 1
    psd = acc / max(n, 1) / (rate * float(np.sum(w * w)))
    psd[1:-1] *= 2.0
    return np.fft.rfftfreq(nfft, 1.0 / rate), psd


def _band_db(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float | None:
    m = (f >= lo) & (f < hi)
    if not m.any():
        return None
    v = float(np.mean(p[m]))
    return 10 * math.log10(v) if v > 0 else -math.inf


def measure_master(path: str, info: dict[str, Any], stereo: str | None = None,
                   stereo_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """The measured rows (S-522, S-524, S-525 and, given a stereo master,
    S-523) as dicts {id, verdict, measured, reference, notes[, plot]}, plus
    the numbers behind them under "data"."""
    ch, rate, bits = _fmt(info)
    labels, lfe = track_labels(path, info)
    peak = np.zeros(ch, dtype=np.float32)
    lfe_parts: list[np.ndarray] = []
    mono_parts: list[np.ndarray] = []
    keep = np.ones(ch, dtype=bool)
    keep[lfe] = False
    hi_nfft = 8192                                    # ~12 Hz bins at 96 kHz
    # per-channel PSD accumulator, (bins, ch), only for a master above 48 kHz
    hi_acc = np.zeros((hi_nfft // 2 + 1, ch)) if rate >= 88200 else None
    hi_n = 0
    hi_w = np.hanning(hi_nfft).astype(np.float32)[None, :, None]
    carry = np.zeros((0, ch), dtype=np.float32)
    frames = 0
    for x in blocks(path, info):
        frames += len(x)
        np.maximum(peak, np.max(np.abs(x), axis=0), out=peak)
        if lfe:
            lfe_parts.append(x[:, lfe[0]].copy())
        if stereo is not None:
            mono_parts.append(x[:, keep].sum(axis=1, dtype=np.float32))
        if hi_acc is not None:
            buf = np.concatenate([carry, x]) if len(carry) else x
            k = len(buf) // hi_nfft
            if k:
                seg = buf[:k * hi_nfft].reshape(k, hi_nfft, ch) * hi_w
                hi_acc += (np.abs(np.fft.rfft(seg, axis=1)) ** 2).sum(axis=0)
                hi_n += k
            carry = buf[k * hi_nfft:].copy()
    rows = [_silence_row(peak, labels, lfe)]
    data: dict[str, Any] = {"frames": frames, "sampleRate": rate,
                            "peakDbfs": [(20 * math.log10(float(p)) if p > 0 else None)
                                         for p in peak],
                            "lfeTracks": [i + 1 for i in lfe]}
    lfe_x = np.concatenate(lfe_parts) if lfe_parts else None
    rows.append(_lfe_row(lfe_x, rate, lfe, labels, data))
    rows.append(_hi_row(hi_acc, hi_n, hi_nfft, rate, bits, labels, data))
    if stereo is not None:
        mono = np.concatenate(mono_parts) if mono_parts else np.zeros(0, np.float32)
        rows.append(sync_row(mono, rate, stereo, stereo_info, data))
    return {"rows": rows, "data": data}


def _row(rid: str, verdict: str, measured: str, reference: str,
         notes: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    d = {"id": rid, "verdict": verdict, "measured": measured, "reference": reference,
         "notes": list(notes or [])}
    d.update(extra)
    return d


def _list(names: list[str], limit: int = 6) -> str:
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


# ------------------------------------------------------------- S-524
def _silence_row(peak: np.ndarray, labels: list[str], lfe: list[int]) -> dict[str, Any]:
    thr = 10 ** (NEAR_SILENT_DBFS / 20)
    digital = [i for i, p in enumerate(peak) if p == 0]
    near = [i for i, p in enumerate(peak) if 0 < p < thr]
    ref = f"every channel peaks at or above {NEAR_SILENT_DBFS:g} dBFS"
    notes = [f"near-silent threshold {NEAR_SILENT_DBFS:g} dBFS peak (provisional)"]
    if not digital and not near:
        return _row("S-524", PASS, f"all {len(peak)} channels carry signal", ref, notes)
    parts = []
    if digital:
        parts.append(f"digitally silent: {_list([labels[i] for i in digital])}")
    if near:
        parts.append(f"near-silent: {_list([labels[i] for i in near])}")
    if any(i in lfe for i in digital + near):
        notes.append("a silent LFE is common when a mix sends nothing to it; "
                     "labels ask for it to be confirmed, not removed")
    return _row("S-524", WARN, f"{len(digital) + len(near)} of {len(peak)} channels: "
                + "; ".join(parts), ref, notes,
                silent=[i + 1 for i in digital], nearSilent=[i + 1 for i in near])


# ------------------------------------------------------------- S-522
def _bands(lo: float, hi: float, per_octave: int = 6) -> list[tuple[float, float, float]]:
    out = []
    k = math.ceil(per_octave * math.log2(lo / 1000.0))
    while True:
        c = 1000.0 * 2 ** (k / per_octave)
        if c > hi:
            break
        out.append((c * 2 ** (-0.5 / per_octave), c, c * 2 ** (0.5 / per_octave)))
        k += 1
    return out


def lfe_mask_db(f: float) -> float:
    """The reference shape: a Butterworth low-pass of LFE_REF_ORDER at
    LFE_REF_FC_HZ, not followed below LFE_FLOOR_DB, plus LFE_TOL_DB."""
    h2 = 1.0 / (1.0 + (f / LFE_REF_FC_HZ) ** (2 * LFE_REF_ORDER))
    return max(10 * math.log10(h2), LFE_FLOOR_DB) + LFE_TOL_DB


def _lfe_row(x: np.ndarray | None, rate: int, lfe: list[int], labels: list[str],
             data: dict[str, Any]) -> dict[str, Any]:
    ref = (f"cut-off 100-150 Hz: at or under a {6 * LFE_REF_ORDER} dB/octave low-pass "
           f"at {LFE_REF_FC_HZ:g} Hz, +{LFE_TOL_DB:g} dB")
    notes = [f"reference shape, tolerance and floor ({LFE_FLOOR_DB:g} dB) provisional; "
             "the specification gives the cut-off range and no slope"]
    if not lfe:
        return _row("S-522", PASS, "no LFE channel", ref, ["nothing to low-pass"])
    if len(lfe) > 1:
        notes.append(f"{len(lfe)} LFE channels; the first ({labels[lfe[0]]}) is judged")
    if x is None or not np.any(x):
        return _row("S-522", PASS, f"{labels[lfe[0]]} is silent", ref,
                    notes + ["nothing to low-pass; S-524 lists the silence"])
    nfft = 1 << int(round(math.log2(rate / 3.0)))          # ~3 Hz bins
    f, p = _welch(x.astype(np.float64), rate, nfft)
    pas = _band_db(f, p, *LFE_PASS_BAND)
    lvl = 10 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-300)
    if pas is None or not math.isfinite(pas) or lvl < LFE_MIN_DBFS:
        return _row("S-522", UNDET, f"LFE level {lvl:.1f} dBFS", ref,
                    notes + [f"too little bass to judge a filter (below {LFE_MIN_DBFS:g} dBFS)"])
    tot = float(np.sum(p))
    below = float(np.sum(p[f <= LFE_PASS_BAND[1]]))
    if tot > 0 and below / tot < 0.01:
        # Almost nothing in the LFE's own band: the shape cannot be read
        # relative to it, and an LFE carrying only higher content is the
        # opposite of low-passed.
        fpk = float(f[int(np.argmax(p))])
        data["lfe"] = {"track": lfe[0] + 1, "levelDbfs": round(lvl, 2),
                       "below80HzDb": round(10 * math.log10(below / tot), 2) if below > 0 else None,
                       "strongestHz": round(fpk, 1)}
        return _row("S-522", WARN, f"{labels[lfe[0]]}: almost nothing below "
                    f"{LFE_PASS_BAND[1]:g} Hz; its energy peaks near {fpk:.0f} Hz", ref,
                    notes + ["less than 1 % of the channel's energy lies in its own band"])
    top = min(LFE_JUDGE_BAND[1], 0.45 * rate)
    curve = []
    worst = (-math.inf, 0.0)
    for lo, c, hi in _bands(25.0, top):
        b = _band_db(f, p, lo, hi)
        if b is None:
            continue
        rel = b - pas if math.isfinite(b) else -200.0
        curve.append((c, rel))
        if c >= LFE_JUDGE_BAND[0]:
            over = rel - lfe_mask_db(c)
            if over > worst[0]:
                worst = (over, c)
    above = float(np.sum(p[f > 200.0]))
    e200 = 10 * math.log10(above / tot) if above > 0 and tot > 0 else -math.inf
    corner = next((c for c, r in curve if c >= 40.0 and r <= -6.0), None)
    data["lfe"] = {"track": lfe[0] + 1, "levelDbfs": round(lvl, 2),
                   "passbandDb": round(pas, 2), "worstExcessDb": round(worst[0], 2),
                   "worstAtHz": round(worst[1], 1), "energyAbove200HzDb": round(e200, 2),
                   "minus6dBHz": None if corner is None else round(corner, 1),
                   "spectrum": [[round(c, 1), round(r, 2)] for c, r in curve]}
    measured = (f"{labels[lfe[0]]}: -6 dB near "
                + (f"{corner:.0f} Hz" if corner else "(not reached)")
                + f"; worst {worst[0]:+.1f} dB against the shape at {worst[1]:.0f} Hz; "
                f"energy above 200 Hz {e200:.1f} dB of the total")
    verdict = PASS if worst[0] <= 0.0 else WARN
    return _row("S-522", verdict, measured, ref, notes, plot=_lfe_svg(curve))


def _lfe_svg(curve: list[tuple[float, float]]) -> str:
    """The LFE's spectrum against the reference shape, as a small inline SVG
    (numbers only, no text from the file)."""
    W, H, L, R, T, B = 520, 200, 40, 10, 10, 26
    f0, f1, d0, d1 = 20.0, 4000.0, -80.0, 10.0
    xs = lambda f: L + (W - L - R) * math.log(f / f0) / math.log(f1 / f0)
    ys = lambda d: T + (H - T - B) * (d1 - max(min(d, d1), d0)) / (d1 - d0)
    pts = " ".join(f"{xs(c):.1f},{ys(r):.1f}" for c, r in curve if f0 <= c <= f1)
    mask = " ".join(f"{xs(c):.1f},{ys(lfe_mask_db(c)):.1f}"
                    for _l, c, _h in _bands(LFE_JUDGE_BAND[0], f1))
    grid = "".join(f'<line x1="{xs(g):.1f}" y1="{T}" x2="{xs(g):.1f}" y2="{H - B}" stroke="#d0d7de"/>'
                   f'<text x="{xs(g):.1f}" y="{H - 8}" font-size="9" text-anchor="middle">{g:g}</text>'
                   for g in (20, 50, 100, 200, 500, 1000, 2000, 4000))
    grid += "".join(f'<line x1="{L}" y1="{ys(d):.1f}" x2="{W - R}" y2="{ys(d):.1f}" stroke="#eaeef2"/>'
                    f'<text x="{L - 4}" y="{ys(d) + 3:.1f}" font-size="9" text-anchor="end">{d:g}</text>'
                    for d in (0, -20, -40, -60, -80))
    band = (f'<rect x="{xs(100):.1f}" y="{T}" width="{xs(150) - xs(100):.1f}" '
            f'height="{H - T - B}" fill="#ddf4ff"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}" font-family="sans-serif">{band}{grid}'
            f'<polyline points="{mask}" fill="none" stroke="#9a6700" stroke-dasharray="4 3"/>'
            f'<polyline points="{pts}" fill="none" stroke="#0969da" stroke-width="1.5"/>'
            f'<text x="{W - R}" y="{T + 10}" font-size="9" text-anchor="end">LFE spectrum, dB re '
            f'20-80 Hz (blue); reference shape + tolerance (dashed); 100-150 Hz shaded; Hz</text>'
            "</svg>")


# ------------------------------------------------------------- S-525
def _hi_row(acc: np.ndarray | None, n: int, nfft: int, rate: int, bits: int,
            labels: list[str], data: dict[str, Any]) -> dict[str, Any]:
    ref = ("content above 24 kHz, with no brick-wall edge at the 48 kHz band "
           "edge (a 96 kHz master is natively 96 kHz)")
    if acc is None:
        return _row("S-525", PASS, f"{rate / 1000:g} kHz master: not applicable", ref)
    notes = [f"thresholds provisional: content {CONTENT_OVER_FLOOR_DB:g} dB over the "
             f"{bits}-bit quantization floor in {HI_BAND[0] / 1000:g}-{HI_BAND[1] / 1000:g} kHz; "
             f"edge {BRICKWALL_DB:g} dB between {EDGE_BAND[0] / 1000:g}-{EDGE_BAND[1] / 1000:g} kHz "
             "and that band"]
    if n == 0:
        return _row("S-525", UNDET, "too short to measure", ref, notes)
    w = np.hanning(nfft)
    psd = acc / n / (rate * float(np.sum(w * w)))
    psd[1:-1] *= 2.0
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    q = 2.0 ** -(bits - 1)
    floor_db = 10 * math.log10((q * q / 12.0) / (rate / 2.0))     # per Hz, one-sided
    native, upsampled, dull = [], [], []
    per = []
    for c in range(psd.shape[1]):
        hi = _band_db(f, psd[:, c], *HI_BAND)
        edge = _band_db(f, psd[:, c], *EDGE_BAND)
        hi = -300.0 if hi is None or not math.isfinite(hi) else hi
        edge = -300.0 if edge is None or not math.isfinite(edge) else edge
        per.append((round(edge - floor_db, 1), round(hi - floor_db, 1)))
        if edge - floor_db < JUDGEABLE_OVER_FLOOR_DB and hi - floor_db < CONTENT_OVER_FLOOR_DB:
            dull.append(c)
        elif hi - floor_db >= CONTENT_OVER_FLOOR_DB and edge - hi < BRICKWALL_DB:
            native.append(c)
        else:
            upsampled.append(c)
    data["above24k"] = {"floorDbPerHz": round(floor_db, 1),
                        "perChannel": [{"track": i + 1, "edgeOverFloorDb": e, "hiOverFloorDb": h}
                                       for i, (e, h) in enumerate(per)]}
    if upsampled and not native:
        return _row("S-525", WARN,
                    f"{len(upsampled)} channel(s) with top-end content stop at the 48 kHz band edge "
                    f"(nothing above 24 kHz, or a brick-wall edge); none shows content above it",
                    ref, notes + ["consistent with a 48 kHz recording upsampled to 96 kHz"],
                    channels=[c + 1 for c in upsampled])
    if native:
        n2 = notes + ([f"{len(upsampled)} channel(s) stop at the band edge: "
                       + _list([labels[c] for c in upsampled])] if upsampled else [])
        return _row("S-525", PASS, f"{len(native)} channel(s) carry content above 24 kHz "
                    "with no edge at 24 kHz", ref, n2)
    return _row("S-525", UNDET, "too little top end on any channel to tell", ref, notes)


# ------------------------------------------------------------- S-523
def _mono_of(path: str, info: dict[str, Any]) -> tuple[np.ndarray, int]:
    _ch, rate, _bits = _fmt(info)
    parts = [x.sum(axis=1, dtype=np.float32) for x in blocks(path, info)]
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), rate


def align(a: np.ndarray, s: np.ndarray, rate: int,
          max_lag_s: float = 10.0) -> tuple[int, float]:
    """(d, r): the master's sample index of the stereo's first sample (a[n+d]
    ~ s[n]) and the normalised correlation at that lag. A coarse search on a
    16x block-mean of both, then every lag within +/-32 samples of it."""
    if len(a) == 0 or len(s) == 0:
        return 0, 0.0
    dec = 16
    na, ns = len(a) // dec, len(s) // dec
    ca = a[:na * dec].reshape(na, dec).mean(axis=1).astype(np.float64)
    cs = s[:ns * dec].reshape(ns, dec).mean(axis=1).astype(np.float64)
    nfft = 1 << int(math.ceil(math.log2(na + ns)))
    xc = np.fft.irfft(np.fft.rfft(ca, nfft) * np.conj(np.fft.rfft(cs, nfft)), nfft)
    maxl = int(max_lag_s * rate / dec)
    lags = np.concatenate([np.arange(0, min(maxl, na) + 1), np.arange(-min(maxl, ns), 0)])
    vals = np.concatenate([xc[:min(maxl, na) + 1], xc[nfft - min(maxl, ns):]])
    coarse = int(lags[int(np.argmax(vals))]) * dec
    best, best_r = coarse, -2.0
    a64, s64 = a.astype(np.float64), s.astype(np.float64)
    for d in range(coarse - 2 * dec, coarse + 2 * dec + 1):
        a0, s0 = max(d, 0), max(-d, 0)
        m = min(len(a) - a0, len(s) - s0)
        if m <= 0:
            continue
        x = a64[a0:a0 + m]
        y = s64[s0:s0 + m]
        den = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
        r = float(np.dot(x, y)) / den if den > 0 else 0.0
        if r > best_r:
            best, best_r = d, r
    return best, best_r


def sync_row(mono: np.ndarray, rate: int, stereo: str, stereo_info: dict[str, Any] | None,
             data: dict[str, Any]) -> dict[str, Any]:
    frame = rate // FRAMES_PER_SECOND
    ref = (f"start aligned (within {LEAD_TOL_MS:g} ms), extra tail under one frame "
           f"({frame} samples)")
    notes = [f"stereo master: {os.path.basename(stereo)}",
             f"lead tolerance {LEAD_TOL_MS:g} ms and minimum correlation "
             f"{MIN_CORRELATION:g} provisional; the frame is the specification's"]
    if stereo_info is None:
        from sentinel.bwf import adm_info
        stereo_info = adm_info(stereo)
    try:
        s, srate = _mono_of(stereo, stereo_info)
    except NotJudged as e:
        return _row("S-523", UNDET, f"the stereo master is not readable: {e}", ref, notes)
    if srate != rate:
        return _row("S-523", UNDET, f"the stereo master is {srate} Hz, the master {rate} Hz",
                    ref, notes + ["not compared across sample rates"])
    d, r = align(mono, s, rate)
    tail = (len(mono) - d) - len(s)
    tol = int(round(LEAD_TOL_MS * rate / 1000.0))
    data["sync"] = {"stereo": stereo, "offsetSamples": d, "extraTailSamples": tail,
                    "correlation": round(r, 4), "frameSamples": frame,
                    "masterFrames": len(mono), "stereoFrames": len(s)}
    head = (f"offset {d:+d} samples ({1000.0 * d / rate:+.2f} ms), extra tail {tail:+d} samples, "
            f"correlation {r:.3f}")
    if r < MIN_CORRELATION:
        return _row("S-523", UNDET, head, ref, notes + [
            "the fold-down and the stereo do not correlate well enough to align them"])
    problems = []
    if d > tol:
        problems.append(f"{d} samples of extra lead before the stereo's first sample")
    elif d < -tol:
        problems.append(f"starts {-d} samples after the stereo")
    if tail >= frame:
        problems.append(f"extra tail of {tail} samples (one frame is {frame})")
    if problems:
        return _row("S-523", FAIL, head + ": " + "; ".join(problems), ref, notes)
    if tail <= -frame:
        return _row("S-523", WARN, head + f": ends {-tail} samples before the stereo", ref, notes)
    return _row("S-523", PASS, head, ref, notes)


def gapless_boundaries(syncs: list[dict[str, Any] | None], names: list[str]) -> dict[str, Any]:
    """S-533 from each track's S-523 measurement, in album order. The boundary
    between tracks i and i+1 moves by track i's extra tail and by track i+1's
    offset; both must be within half a frame of the stereo's."""
    ref = "every boundary within half a frame of the stereo's"
    if not syncs or any(s is None for s in syncs):
        missing = [n for n, s in zip(names, syncs) if s is None]
        return _row("S-533", UNDET, "not measured on " + _list(missing), ref,
                    ["needs each track's stereo master and iamf-sentinel-pro"])
    half = syncs[0]["frameSamples"] // 2
    worst, where = 0, ""
    for i in range(len(syncs) - 1):
        e = max(abs(syncs[i]["extraTailSamples"]), abs(syncs[i + 1]["offsetSamples"]))
        if e > worst:
            worst, where = e, f"{names[i]} / {names[i + 1]}"
    if len(syncs) < 2:
        return _row("S-533", UNDET, "one track: no boundary", ref)
    measured = f"largest boundary error {worst} samples" + (f" ({where})" if where else "")
    return _row("S-533", FAIL if worst > half else PASS, measured, ref,
                [f"half a frame is {half} samples"])
