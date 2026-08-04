"""Own BS.1770-4 loudness / true-peak implementation + WAV reading (L3, Phase 2).

Clean-room DSP written from ITU-R BS.1770-4 (K-weighting pre-filter + RLB
high-pass, 400 ms gated blocks at 75% overlap, absolute -70 LKFS and relative
-10 LU gates, channel weights 1.41 for |elev|<30 and 60<=|az|<=120, LFE
excluded) and Annex 2 (4x oversampled true peak). Cross-checked in tests
against ffmpeg ebur128 to the WP1 G5 gate (<=0.06 LU).

numpy is required for the Python reference path only — the Phase-1 core stays
pure-stdlib, and since the ADR-6b backend switch this module also imports
without numpy when the compiled `sentinel-dsp` kernel provides the measurement
backend (doc 35). Import of this module is deferred until an L3 run is
requested.

Backend selection (ADR-6b, docs 32/33/35): `analyze()` is the file-level
measurement facade — one WAV in, one JSON-shaped dict out (the ADR-4 boundary).
When the `sentinel-dsp` binary is discoverable (``$SENTINEL_DSP`` explicit path,
else ``sentinel-dsp`` on ``$PATH``) it is invoked as a subprocess; otherwise the
numpy reference functions below compute the same document. ``SENTINEL_DSP=off``
(or ``0``) forces the reference path. A present-but-failing kernel raises
`KernelError` — an execution error, never a silent fallback (doc 34's
degradation philosophy: silent downgrades mask regressions).

The per-function numpy implementations remain the *reference* (ADR-3): the
kernel is differential-gated against them (sentinel-dsp/tests/differential.py,
gates ΔIL ≤ 0.001 LU / Δpeak ≤ 0.01 dB / verdict identity / 1e-6 eps).
"""

from __future__ import annotations
import json
import math
import os
import shutil
import struct
import subprocess

try:
    import numpy as np
except ImportError:                     # kernel-backed installs may omit numpy
    np = None                           # (ADR-6b; guarded by _require_np below)


def _require_np():
    if np is None:
        raise ImportError(
            "numpy is required for the Python reference DSP path — install "
            "iamf-sentinel-pro[numpy], or provide the sentinel-dsp kernel "
            "binary (on $PATH or via $SENTINEL_DSP) to use the compiled "
            "backend")

# ---------------------------------------------------------------------------
# WAV reading (WAVE_FORMAT_EXTENSIBLE-tolerant; stdlib `wave` rejects 0xFFFE)
# ---------------------------------------------------------------------------


def read_wav(path: str) -> tuple[int, "np.ndarray"]:
    """Return (sample_rate, float64 array shaped (frames, channels)) in [-1, 1)."""
    _require_np()
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    pos, end = 12, len(data)
    fmt = None
    payload = None
    while pos + 8 <= end:
        cid = data[pos:pos + 4]
        csize = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csize]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            payload = body
        pos += 8 + csize + (csize & 1)
    if fmt is None or payload is None:
        raise ValueError(f"missing fmt/data chunk: {path}")
    (wformat, ch, sr, _brate, _balign, bits) = struct.unpack("<HHIIHH", fmt[:16])
    if wformat == 0xFFFE and len(fmt) >= 26:      # EXTENSIBLE: real format in GUID
        wformat = struct.unpack("<H", fmt[24:26])[0]
    if wformat == 3:                               # IEEE float
        dt = np.float32 if bits == 32 else np.float64
        d = np.frombuffer(payload, dtype=dt).astype(np.float64)
    elif bits == 16:
        d = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 2**15
    elif bits == 24:
        a = np.frombuffer(payload, dtype=np.uint8)
        a = a[: (len(a) // 3) * 3].reshape(-1, 3)
        ints = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        d = ints.astype(np.float64) / 2**23
    elif bits == 32:
        d = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2**31
    else:
        raise ValueError(f"unhandled bit depth {bits}: {path}")
    frames = len(d) // ch
    return sr, d[: frames * ch].reshape(-1, ch)


# ---------------------------------------------------------------------------
# BS.1770-4 K-weighting
# ---------------------------------------------------------------------------

def _k_weighting_coeffs(sr: int):
    """Stage 1 shelf + stage 2 RLB high-pass biquads for sample rate `sr`.

    Designed from the BS.1770-4 analogue prototypes (shelf: f0=1681.9744509
    Hz, G=+3.99984385 dB, Q=0.7071752; high-pass: f0=38.13547088 Hz,
    Q=0.50032705) via the bilinear transform, which reproduces the spec's
    tabulated 48 kHz coefficients to ~1e-6.
    """
    # Stage 1: high shelf
    f0, G, Q = 1681.9744509555319, 3.999843853973347, 0.7071752369554193
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.499666774155
    a0 = 1.0 + K / Q + K * K
    b_shelf = [(Vh + Vb * K / Q + K * K) / a0,
               2.0 * (K * K - Vh) / a0,
               (Vh - Vb * K / Q + K * K) / a0]
    a_shelf = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    # Stage 2: RLB high-pass
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    # Spec form keeps the RLB numerator as exactly [1, -2, 1] (not /a0).
    b_hp = [1.0, -2.0, 1.0]
    a_hp = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    return (np.array(b_shelf), np.array(a_shelf)), (np.array(b_hp), np.array(a_hp))


def _biquad(x: "np.ndarray", b: "np.ndarray", a: "np.ndarray") -> "np.ndarray":
    """Direct-form II transposed biquad over axis 0 (frames)."""
    try:
        from scipy.signal import lfilter
        return lfilter(b, a, x, axis=0)
    except ImportError:
        y = np.empty_like(x)
        z1 = np.zeros(x.shape[1]) if x.ndim == 2 else 0.0
        z2 = np.zeros(x.shape[1]) if x.ndim == 2 else 0.0
        b0, b1, b2 = b
        _, a1, a2 = a
        for n in range(x.shape[0]):
            xn = x[n]
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            y[n] = yn
        return y


# BS.2051-style channel orders emitted by the reference decoders (WP1
# decode-verify convention), with per-channel loudness weights.
#
# Conformance note (F33 fix): weights follow ITU-R BS.1770-4 Tables 4/5
# exactly (read directly — see doc 65): 1.41 requires BOTH ear level
# (|elev| < 30°) AND azimuth in the 60°–120° band. Side surrounds (M±090)
# and 5.1 surrounds (M±110) get 1.41; rear surrounds (M±135) get 1.00;
# heights (U-layer) are unconditionally 1.00; LFE excluded (0).
# This matches the declared-value producer: iamf-tools measures with
# google/loudness_ebur128 using caller-supplied weights and supplies 1.0
# for the rears (Table-5 conformant, 12/12 — doc 65 §3). An earlier
# revision weighted 7.1.x rears at 1.41 on the belief that the ecosystem
# stack did; that premise was refuted (doc 64) and the tables corrected
# (F33). ffmpeg's ebur128 genuinely does over-weight rears and top-backs
# (F29) — that behaviour lives in FFMPEG_WEIGHTS below, never here.
CHANNEL_WEIGHTS: dict[str, list[float]] = {
    # decoder_main --output_layout strings
    "2.0":  [1.0, 1.0],
    "5.1":  [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],                    # L R C LFE Ls Rs (M±110)
    "5.1.2": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0],
    "5.1.4": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0],
    # 7.1.x order: L R C LFE Lss Rss (M±090) Lrs Rrs (M±135) [+ heights]
    "7.1":  [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0],
    "7.1.2": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0],
    "7.1.4": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "3.1.2": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
}

# ffmpeg f_ebur128 BACK_MASK variant (empirically confirmed on a masked 7.1.4
# WAV): side + back + TOP-BACK at 1.41, top-front 1.0. Used by the cross-check
# tests; the primary table above is calibrated against the declared-value
# producer (google/loudness_ebur128 via iamf-tools) on the real samples.
FFMPEG_WEIGHTS: dict[str, list[float]] = {
    "2.0":  [1.0, 1.0],
    "5.1":  [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
    "7.1":  [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41],
    "7.1.4": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41, 1.0, 1.0, 1.41, 1.41],
}


def integrated_loudness(pcm: "np.ndarray", sr: int,
                        weights: list[float] | None = None) -> float:
    """BS.1770-4 gated integrated loudness (LKFS). pcm: (frames, channels)."""
    _require_np()
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    nch = pcm.shape[1]
    # Weights-default correction (doc 33): the previous one-liner produced a
    # 2*nch-long weight vector when weights was None (silent +3.01 LU on mono
    # via broadcasting; ValueError on multichannel). Latent in production
    # because callers always pass explicit layout weights; caught by the
    # ADR-6b C++ kernel differential — same differential-duty class as the
    # Phase-1 param_definition_type correction (doc 19).
    if weights is None:
        w = np.ones(nch)
    else:
        w = np.array((list(weights) + [1.0] * nch)[:nch])
    (bs, a_s), (bh, ah) = _k_weighting_coeffs(sr)
    y = _biquad(_biquad(pcm, bs, a_s), bh, ah)

    block = int(round(0.400 * sr))
    hop = int(round(0.100 * sr))
    if y.shape[0] < block:
        pad = np.zeros((block - y.shape[0], y.shape[1]))
        y = np.vstack([y, pad])
    n_blocks = 1 + (y.shape[0] - block) // hop
    # per-block mean square per channel
    sq = y * y
    csum = np.vstack([np.zeros((1, sq.shape[1])), np.cumsum(sq, axis=0)])
    starts = np.arange(n_blocks) * hop
    z = (csum[starts + block] - csum[starts]) / block          # (blocks, ch)
    lk = -0.691 + 10.0 * np.log10(np.maximum((z * w).sum(axis=1), 1e-12))

    abs_gated = lk > -70.0
    if not abs_gated.any():
        return -70.0
    rel_thr = -0.691 + 10.0 * np.log10(
        np.maximum((z[abs_gated] * w).sum(axis=1).mean(), 1e-12)) - 10.0
    gated = abs_gated & (lk > rel_thr)
    if not gated.any():
        return -70.0
    return float(-0.691 + 10.0 * np.log10(
        np.maximum((z[gated] * w).sum(axis=1).mean(), 1e-12)))


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------

def digital_peak_dbfs(pcm: "np.ndarray") -> float:
    _require_np()
    p = float(np.abs(pcm).max()) if pcm.size else 0.0
    return 20.0 * math.log10(p) if p > 0 else -120.0


_TP_FIR_CACHE: dict[int, "np.ndarray"] = {}


def _tp_fir(taps: int = 48, os: int = 4) -> "np.ndarray":
    key = taps * 1000 + os
    if key not in _TP_FIR_CACHE:
        n = np.arange(taps) - (taps - 1) / 2.0
        h = np.sinc(n / os) * np.kaiser(taps, 12.0)
        # Normalize each polyphase branch to unity DC gain so a zero-stuffed
        # input reconstructs at the original amplitude on every phase.
        for p in range(os):
            s = h[p::os].sum()
            if abs(s) > 1e-12:
                h[p::os] /= s
        _TP_FIR_CACHE[key] = h
    return _TP_FIR_CACHE[key]


def true_peak_dbtp(pcm: "np.ndarray", os: int = 4) -> float:
    """BS.1770-4 Annex 2 style 4x oversampled true peak."""
    _require_np()
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    h = _tp_fir(48, os)
    peak = 0.0
    for ch in range(pcm.shape[1]):
        x = pcm[:, ch]
        up = np.zeros(len(x) * os)
        up[::os] = x
        y = np.convolve(up, h, mode="same")
        # no interpolation-gain compensation needed: filter has unity DC per phase
        peak = max(peak, float(np.abs(y).max()))
    return 20.0 * math.log10(peak) if peak > 0 else -120.0


# ---------------------------------------------------------------------------
# Channel-identity / spatial heuristics (F4 essence-misroute, F19/F20)
# ---------------------------------------------------------------------------

def channel_stats(pcm: "np.ndarray") -> list[dict]:
    """Per-channel RMS (dBFS) and peak (dBFS)."""
    _require_np()
    out = []
    for ch in range(pcm.shape[1]):
        x = pcm[:, ch]
        peak = float(np.abs(x).max()) if x.size else 0.0
        rms = float(np.sqrt((x * x).mean())) if x.size else 0.0
        out.append({
            "rms_dbfs": 20.0 * math.log10(rms) if rms > 1e-9 else -120.0,
            "peak_dbfs": 20.0 * math.log10(peak) if peak > 1e-9 else -120.0,
        })
    return out


def correlation_matrix(pcm: "np.ndarray") -> "np.ndarray":
    """Normalized cross-correlation (zero-lag) between channels; nan-safe."""
    _require_np()
    x = pcm - pcm.mean(axis=0, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=0))
    norm = np.where(norm < 1e-12, 1.0, norm)
    xn = x / norm
    return xn.T @ xn


def duplicate_pairs(pcm: "np.ndarray", corr_gate: float = 0.999,
                    level_gate_db: float = 0.1,
                    silence_floor_db: float = -80.0) -> list[tuple[int, int, float]]:
    """Channel pairs that are effectively the same signal (F4 duplication)."""
    stats = channel_stats(pcm)
    corr = correlation_matrix(pcm)
    out = []
    n = pcm.shape[1]
    for i in range(n):
        if stats[i]["rms_dbfs"] < silence_floor_db:
            continue
        for j in range(i + 1, n):
            if stats[j]["rms_dbfs"] < silence_floor_db:
                continue
            if corr[i, j] >= corr_gate and \
               abs(stats[i]["rms_dbfs"] - stats[j]["rms_dbfs"]) <= level_gate_db:
                out.append((i, j, float(corr[i, j])))
    return out


def silent_channels(pcm: "np.ndarray", floor_db: float = -80.0) -> list[int]:
    stats = channel_stats(pcm)
    active = [s["rms_dbfs"] >= floor_db for s in stats]
    if not any(active):
        return []                      # an all-silent program is a different finding
    return [i for i, a in enumerate(active) if not a]


# Nominal BS.2051 azimuth/elevation (degrees) for the decoder output orders.
SPEAKER_POSITIONS: dict[str, list[tuple[float, float]]] = {
    "2.0":  [(30, 0), (-30, 0)],
    "5.1":  [(30, 0), (-30, 0), (0, 0), (0, -30), (110, 0), (-110, 0)],
    "7.1.4": [(30, 0), (-30, 0), (0, 0), (0, -30), (90, 0), (-90, 0),
              (135, 0), (-135, 0), (45, 45), (-45, 45), (135, 45), (-135, 45)],
}


def energy_vector_profile(pcm: "np.ndarray", sr: int, layout: str,
                          win_s: float = 0.5) -> dict:
    """Windowed Gerzon energy vectors, reported as deviation from the layout's
    equal-energy baseline vector E0 (irregular layouts do not cancel, so raw
    |E| is meaningless as a degeneracy measure). Every window's E pinned to E0
    means all speakers are driven with the same per-channel energy balance the
    whole program — combined with near-total inter-channel correlation, that is
    the origin-collapse signature (F19/F20)."""
    _require_np()
    pos = SPEAKER_POSITIONS.get(layout)
    if pos is None or pcm.shape[1] != len(pos):
        return {"windows": 0, "median_dev": None, "p95_dev": None}
    az = np.radians([p[0] for p in pos])
    el = np.radians([p[1] for p in pos])
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)], axis=1)             # (ch, 3)
    e0 = dirs.mean(axis=0)                             # equal-energy baseline
    win = max(1, int(win_s * sr))
    devs = []
    for s in range(0, pcm.shape[0] - win + 1, win):
        e = (pcm[s:s + win] ** 2).sum(axis=0)         # per-channel energy
        tot = e.sum()
        if tot < 1e-9:
            continue
        v = (e[:, None] * dirs).sum(axis=0) / tot
        devs.append(float(np.linalg.norm(v - e0)))
    if not devs:
        return {"windows": 0, "median_dev": None, "p95_dev": None}
    d = np.array(devs)
    return {"windows": len(devs), "median_dev": float(np.median(d)),
            "p95_dev": float(np.percentile(d, 95))}


def mean_offdiag_correlation(pcm: "np.ndarray") -> float:
    c = correlation_matrix(pcm)
    n = c.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(c[mask]).mean())


# ---------------------------------------------------------------------------
# ADR-6b backend switch (doc 35): sentinel-dsp kernel vs numpy reference
# ---------------------------------------------------------------------------

class KernelError(RuntimeError):
    """The sentinel-dsp kernel was selected but failed to produce a result.

    Deliberately NOT caught by a fallback to the Python path: a
    present-but-broken kernel is an execution error (doc 34's no-silent-
    downgrade rule). Force the reference path with SENTINEL_DSP=off instead.
    """


_KERNEL_TIMEOUT_S = 300         # generous: worst Phase-2 measurement was ~3.6 s


def kernel_path() -> str | None:
    """Resolve the sentinel-dsp binary, or None for the Python reference path.

    Order (doc 35 D-S2): $SENTINEL_DSP — 'off'/'0'/'' forces Python, any other
    value is an explicit binary path (missing → KernelError: an explicitly
    requested kernel that isn't there is an error, not a preference) — then
    `sentinel-dsp` on $PATH.
    """
    env = os.environ.get("SENTINEL_DSP")
    if env is not None:
        if env.strip().lower() in ("", "0", "off"):
            return None
        if not (os.path.isfile(env) and os.access(env, os.X_OK)):
            raise KernelError(f"$SENTINEL_DSP points to a non-executable path: {env}")
        return env
    return shutil.which("sentinel-dsp")


_ANALYZE_KEYS = ("sample_rate", "channels", "frames", "integrated_lufs",
                 "digital_peak_dbfs", "true_peak_dbtp", "channel_stats",
                 "correlation", "energy_profile")


def _analyze_kernel(binary: str, wav_path: str, layout: str | None,
                    weights: list[float] | None, variant: str,
                    win_s: float) -> dict:
    cmd = [binary, wav_path, "--variant", variant, "--win-s", str(win_s)]
    if weights is not None:
        cmd += ["--weights", ",".join(repr(float(w)) for w in weights)]
    if layout:
        cmd += ["--layout", layout]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", timeout=_KERNEL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise KernelError(f"sentinel-dsp timed out after {_KERNEL_TIMEOUT_S}s "
                          f"on {wav_path}") from e
    except OSError as e:
        raise KernelError(f"sentinel-dsp failed to start ({binary}): {e}") from e
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise KernelError(f"sentinel-dsp exited {r.returncode} on {wav_path}: "
                          + " | ".join(tail))
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise KernelError(f"sentinel-dsp emitted invalid JSON on {wav_path}: {e}") from e
    missing = [k for k in _ANALYZE_KEYS if k not in doc]
    if missing:
        raise KernelError(f"sentinel-dsp output missing fields {missing} "
                          f"(binary/contract mismatch? {binary})")
    # normalize to the reference API's shapes
    doc["correlation"]["duplicate_pairs"] = [
        (int(i), int(j), float(r_)) for i, j, r_ in doc["correlation"]["duplicate_pairs"]]
    doc["backend"] = "kernel"
    doc["kernel_binary"] = binary
    return doc


def _analyze_python(wav_path: str, layout: str | None,
                    weights: list[float] | None, variant: str,
                    win_s: float) -> dict:
    _require_np()
    sr, pcm = read_wav(wav_path)
    if weights is None and layout:
        table = FFMPEG_WEIGHTS if variant == "ffmpeg" else CHANNEL_WEIGHTS
        weights = table.get(layout)
    stats = channel_stats(pcm)
    return {
        "backend": "python",
        "sample_rate": sr,
        "channels": pcm.shape[1],
        "frames": pcm.shape[0],
        "variant": variant,
        "layout": layout,
        "integrated_lufs": integrated_loudness(pcm, sr, weights),
        "digital_peak_dbfs": digital_peak_dbfs(pcm),
        "true_peak_dbtp": true_peak_dbtp(pcm),
        "channel_stats": stats,
        "correlation": {
            "duplicate_pairs": duplicate_pairs(pcm),
            "silent_channels": silent_channels(pcm),
            "mean_offdiag": mean_offdiag_correlation(pcm),
        },
        "energy_profile": (energy_vector_profile(pcm, sr, layout, win_s)
                           if layout else
                           {"windows": 0, "median_dev": None, "p95_dev": None}),
    }


def analyze(wav_path: str, layout: str | None = None,
            weights: list[float] | None = None, variant: str = "primary",
            win_s: float = 0.5, backend: str | None = None) -> dict:
    """File-level measurement facade — the ADR-4 shape (one WAV in, one doc out).

    Returns the sentinel-dsp JSON document (doc 33) as a dict, plus
    ``backend`` ("kernel"|"python"). ``backend=None`` auto-selects via
    `kernel_path()`; "python"/"kernel" force a path (forcing "kernel" with no
    binary raises KernelError).
    """
    if backend not in (None, "python", "kernel"):
        raise ValueError(f"unknown backend {backend!r}")
    binary = None
    if backend != "python":
        binary = kernel_path()
        if backend == "kernel" and binary is None:
            raise KernelError("backend='kernel' requested but no sentinel-dsp "
                              "binary found ($SENTINEL_DSP / $PATH)")
    if binary is not None:
        return _analyze_kernel(binary, wav_path, layout, weights, variant, win_s)
    return _analyze_python(wav_path, layout, weights, variant, win_s)
