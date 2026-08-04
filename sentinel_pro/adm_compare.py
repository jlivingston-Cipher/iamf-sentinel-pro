"""Source-referenced fidelity QC (Phase 3): ADM source vs produced IAMF.

Closes the doc-19 scope-boundary item: F17 (importance threshold silently
ineffective) and F18 (block-gain automation silently discarded) are invisible
to single-file validation — they need the *source* ADM for comparison. This
module diffs the produced IAMF against an independent reference render of the
source ADM:

    reference:  EAR (BS.2127, ear-render, subprocess — ADR-3/-4 boundary)
    output:     decoder_main oracle render (sentinel.oracle, subprocess)
    compare:    aligned 100 ms windowed energy envelopes (numpy — Pro side)

Checks (S-33x family, registered in findings.py):

    S-330  authored gain trajectory not realized (F18/F16 class)     FAIL
    S-331  importance filtering ineffective (F17 class)              FAIL
    S-332  source-vs-output level divergence (generic guard)         WARN

The importance reference is built without ADM surgery: the "filtered" source is
the same BW64 with the below-threshold audioObjects' tracks *silenced* (chna
maps audioTrackUID -> track index), which is acoustically identical to removal
and keeps every ID/track offset intact.

IP boundary: original work. BW64/chna/axml reading from ITU-R BS.2088 / EBU
Tech 3285; EAR (BSD-3-Clause) and decoder_main are subprocess oracles only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from sentinel.findings import Finding, Severity, spec
from .oracle import Toolchain

# BS.2051 system strings for ear-render, keyed by the oracle layout names.
EAR_LAYOUTS = {"2.0": "0+2+0", "5.1": "0+5+0", "5.1.2": "2+5+0",
               "5.1.4": "4+5+0", "7.1": "0+7+0", "7.1.4": "4+7+0"}

WINDOW_S = 0.100          # envelope window
NOISE_FLOOR_DB = -80.0    # windows below this (in the reference) are ignored
GAIN_TRAJ_RANGE_DB = 1.5  # pre-registered S-330 threshold (p95-p5 of delta)
GAIN_TRAJ_TREND = 0.8     # |pearson r| of delta vs time
LEVEL_DIVERGENCE_DB = 3.0 # S-332 median |delta| guard
PRESENCE_CORR = 0.25      # S-331 matched-filter presence threshold
RESIDUAL_FLOOR = 1e-6     # skip S-331 if the residual is essentially silent


# ----------------------------------------------------------------- BW64 I/O --

def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """Minimal RIFF/BW64 reader -> (float32 [frames, ch], rate). 16/24/32-bit
    integer PCM and 32-bit float supported."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] not in (b"RIFF", b"BW64", b"RF64"):
        raise ValueError(f"{path}: not a RIFF/BW64 file")
    pos, end = 12, len(data)
    fmt = None
    frames = None
    while pos + 8 <= end:
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, block, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (tag, ch, rate, block, bits)
        elif cid == b"data" and fmt:
            tag, ch, rate, block, bits = fmt
            n = len(body) // block
            if bits == 16:
                x = np.frombuffer(body[:n * block], dtype="<i2").astype(np.float32) / 2 ** 15
            elif bits == 24:
                raw = np.frombuffer(body[:n * block], dtype=np.uint8).reshape(-1, 3)
                x = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8)
                     | (raw[:, 2].astype(np.int32) << 16))
                x = (x - ((x & 0x800000) << 1)).astype(np.float32) / 2 ** 23
            elif bits == 32 and tag == 3:
                x = np.frombuffer(body[:n * block], dtype="<f4").astype(np.float32)
            elif bits == 32:
                x = np.frombuffer(body[:n * block], dtype="<i4").astype(np.float32) / 2 ** 31
            else:
                raise ValueError(f"{path}: unsupported bit depth {bits}")
            frames = x.reshape(n, ch)
        pos += 8 + size + (size & 1)
    if frames is None:
        raise ValueError(f"{path}: no data chunk")
    return frames, fmt[2]


def _find_chunks(data: bytes) -> dict[bytes, tuple[int, int]]:
    """chunk id -> (payload offset, size); first occurrence wins."""
    out, pos, end = {}, 12, len(data)
    while pos + 8 <= end:
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        out.setdefault(cid, (pos + 8, size))
        pos += 8 + size + (size & 1)
    return out


def below_threshold_tracks(adm_path: str, threshold: int) -> tuple[list[int], list[str]]:
    """audioObjects with importance < threshold -> 1-based chna track numbers."""
    with open(adm_path, "rb") as fh:
        data = fh.read()
    chunks = _find_chunks(data)
    if b"axml" not in chunks or b"chna" not in chunks:
        raise ValueError("source has no axml/chna chunk — not an ADM BW64")
    off, size = chunks[b"axml"]
    root = ET.fromstring(data[off:off + size].decode("utf-8", "replace"))
    ns = {"a": root.tag.split("}")[0][1:]} if root.tag.startswith("{") else None
    def iterdeep(el, name):
        return el.iter(f"{{{ns['a']}}}{name}") if ns else el.iter(name)

    uid_refs: list[str] = []
    names: list[str] = []
    for obj in iterdeep(root, "audioObject"):
        imp = obj.get("importance")
        if imp is None or int(imp) >= threshold:
            continue
        names.append(obj.get("audioObjectName", obj.get("audioObjectID", "?")))
        for ref in iterdeep(obj, "audioTrackUIDRef"):
            uid_refs.append((ref.text or "").strip())

    off, size = chunks[b"chna"]
    body = data[off:off + size]
    num_uids = struct.unpack("<H", body[2:4])[0]
    tracks: list[int] = []
    for i in range(num_uids):
        e = body[4 + 40 * i: 4 + 40 * (i + 1)]
        if len(e) < 40:
            break
        track_num = struct.unpack("<H", e[0:2])[0]
        uid = e[2:14].decode("ascii", "replace").strip("\x00 ")
        if uid in uid_refs:
            tracks.append(track_num)
    return sorted(set(tracks)), names


def silence_tracks(adm_path: str, tracks: list[int], out_path: str) -> None:
    """Copy the BW64 with the given 1-based tracks zeroed in the data chunk."""
    with open(adm_path, "rb") as fh:
        data = bytearray(fh.read())
    chunks = _find_chunks(bytes(data))
    f_off, f_size = chunks[b"fmt "]
    _, ch, _, _, block, bits = struct.unpack("<HHIIHH", bytes(data[f_off:f_off + 16]))
    d_off, d_size = chunks[b"data"]
    bytes_per = bits // 8
    n = d_size // block
    view = np.frombuffer(data, dtype=np.uint8, count=n * block, offset=d_off)
    frame_view = view.reshape(n, ch, bytes_per).copy()
    for t in tracks:
        frame_view[:, t - 1, :] = 0
    data[d_off:d_off + n * block] = frame_view.tobytes()
    with open(out_path, "wb") as fh:
        fh.write(bytes(data))


# ------------------------------------------------------------------ renders --

def ear_render(ear_bin: str, adm_path: str, layout: str, out_wav: str,
               timeout: float = 300.0) -> tuple[bool, str]:
    cmd = [ear_bin, "-s", EAR_LAYOUTS[layout], "--enable-block-duration-fix",
           adm_path, out_wav]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)
    ok = r.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 44
    return ok, "\n".join((r.stderr or "").strip().splitlines()[-2:])


def find_ear() -> str | None:
    cand = os.environ.get("SENTINEL_EAR") or shutil.which("ear-render")
    if cand and os.path.exists(cand):
        return cand
    p = "/opt/earvenv/bin/ear-render"      # the documented EAR venv (Dockerfile.adm)
    if os.path.exists(p):
        return p
    return None


# ------------------------------------------------------------------ compare --

def _envelope_db(x: np.ndarray, rate: int) -> np.ndarray:
    """Total-energy envelope (all channels summed) per 100 ms window, in dB."""
    w = max(1, int(rate * WINDOW_S))
    n = (x.shape[0] // w) * w
    e = (x[:n] ** 2).sum(axis=1).reshape(-1, w).mean(axis=1)
    return 10.0 * np.log10(e + 1e-12)


def _align(a: np.ndarray, b: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Trim both to a common start via cross-correlation of sum signals."""
    from scipy.signal import correlate
    sa, sb = a.sum(axis=1), b.sum(axis=1)
    m = min(len(sa), len(sb), rate * 4)
    c = correlate(sa[:m], sb[:m], mode="full", method="fft")
    lag = int(np.argmax(np.abs(c))) - (m - 1)
    if lag > 0:
        a = a[lag:]
    elif lag < 0:
        b = b[-lag:]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def residual_presence(ref: np.ndarray, fref: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    """Matched-filter test for the below-threshold objects' contribution.

    residual = unfiltered_ref - filtered_ref is exactly the rendered signal of
    the objects that a working importance filter would have removed. Returns
    (|corr(residual, out)| on summed channels, residual/ref energy ratio).
    Gain-invariant: a global fold/re-render level shift cannot fake or hide it —
    this is why the level-envelope metric (kept as secondary evidence) cannot
    decide S-331 for static content (filtered vs unfiltered then differ only by
    a constant gain, which any level-offset-tolerant metric must ignore).
    """
    n = min(len(ref), len(fref), len(out))
    s_res = (ref[:n] - fref[:n]).sum(axis=1).astype(np.float64)
    s_out = out[:n].sum(axis=1).astype(np.float64)
    s_ref = ref[:n].sum(axis=1).astype(np.float64)
    e_res, e_out, e_ref = (s_res ** 2).sum(), (s_out ** 2).sum(), (s_ref ** 2).sum()
    if e_res < RESIDUAL_FLOOR or e_out < RESIDUAL_FLOOR:
        return 0.0, 0.0
    corr = float(abs(np.dot(s_res, s_out)) / np.sqrt(e_res * e_out))
    return corr, float(e_res / max(e_ref, 1e-12))


@dataclass
class ADMCompareResult:
    source: str
    output: str
    layout: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    execution_error: str | None = None

    def exit_code(self) -> int:
        if self.execution_error:
            return 2
        return 1 if any(f.severity >= Severity.FAIL for f in self.findings) else 0

    def to_dict(self) -> dict:
        return {
            "sentinel_adm_compare": "0.1.0",
            "source": self.source, "output": self.output, "layout": self.layout,
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


def compare(source_adm: str, output_iamf: str, *, layout: str = "5.1",
            importance_threshold: int | None = None,
            toolchain: Toolchain | None = None,
            ear_bin: str | None = None,
            workdir: str | None = None) -> ADMCompareResult:
    res = ADMCompareResult(source=source_adm, output=output_iamf, layout=layout)
    ear_bin = ear_bin or find_ear()
    if ear_bin is None:
        res.execution_error = "ear-render not found (set $SENTINEL_EAR)"
        return res
    tc = toolchain or Toolchain.discover()
    if not tc.available:
        res.execution_error = "decoder oracle toolchain not found"
        return res
    if layout not in EAR_LAYOUTS:
        res.execution_error = f"unsupported layout {layout!r}"
        return res
    wd = workdir or tempfile.mkdtemp(prefix="sentinel_admc_")

    try:
        # 1. reference render of the source (EAR)
        ref_wav = os.path.join(wd, "ref.wav")
        ok, note = ear_render(ear_bin, source_adm, layout, ref_wav)
        if not ok:
            res.execution_error = f"EAR reference render failed: {note}"
            return res

        # 2. oracle render of the produced IAMF
        dec = tc.decode(output_iamf, layout, "decoder_main")
        if not dec.ok:
            res.execution_error = f"oracle decode failed: {dec.note or dec.stderr_tail}"
            return res

        ref, rate = _read_wav(ref_wav)
        out, orate = _read_wav(dec.wav_path)
        if orate != rate:
            res.execution_error = f"rate mismatch ref {rate} vs out {orate}"
            return res
        ref, out = _align(ref, out, rate)

        env_ref = _envelope_db(ref, rate)
        env_out = _envelope_db(out, rate)
        n = min(len(env_ref), len(env_out))
        env_ref, env_out = env_ref[:n], env_out[:n]
        act = env_ref > NOISE_FLOOR_DB
        if act.sum() < 5:
            res.execution_error = "reference render is (near-)silent — cannot compare"
            return res

        delta = (env_ref - env_out)[act]              # + = output quieter
        t = np.arange(n, dtype=np.float64)[act]
        med = float(np.median(delta))
        detr = delta - med
        p95, p5 = float(np.percentile(detr, 95)), float(np.percentile(detr, 5))
        rng = p95 - p5
        trend = 0.0
        if len(delta) > 3 and float(np.std(delta)) > 1e-9:
            trend = float(np.corrcoef(t, delta)[0, 1])
        res.metrics["level"] = {
            "median_delta_db": round(med, 3),
            "delta_range_p5_p95_db": round(rng, 3),
            "trend_r": round(trend, 3),
            "windows_active": int(act.sum()), "windows_total": int(n),
        }

        # S-330 — authored gain trajectory not realized
        if rng >= GAIN_TRAJ_RANGE_DB and abs(trend) >= GAIN_TRAJ_TREND:
            res.findings.append(_mk(
                "S-330",
                f"source level trajectory diverges from output by {rng:.1f} dB "
                f"(p5–p95) with monotone drift (r={trend:+.2f}) — authored gain "
                f"automation appears not to be realized in the produced IAMF",
                where=f"layout {layout}",
                expected="output level tracks the source's authored gain trajectory",
                found=f"steady divergence, median offset {med:+.1f} dB"))

        # S-332 — generic level divergence guard
        elif abs(med) >= LEVEL_DIVERGENCE_DB:
            res.findings.append(_mk(
                "S-332",
                f"output level diverges from source reference render by "
                f"{med:+.1f} dB (median, {layout}) — re-render/fold level shift",
                where=f"layout {layout}",
                found=f"median delta {med:+.1f} dB"))

        # S-331 — importance filtering effectiveness
        if importance_threshold is not None:
            tracks, names = below_threshold_tracks(source_adm, importance_threshold)
            res.metrics["importance"] = {
                "threshold": importance_threshold,
                "below_threshold_objects": names,
                "silenced_tracks": tracks,
            }
            if tracks:
                filt_adm = os.path.join(wd, "filtered_src.wav")
                silence_tracks(source_adm, tracks, filt_adm)
                filt_wav = os.path.join(wd, "ref_filtered.wav")
                ok, note = ear_render(ear_bin, filt_adm, layout, filt_wav)
                if not ok:
                    res.execution_error = f"EAR filtered render failed: {note}"
                    return res
                fref, _frate = _read_wav(filt_wav)
                # same EAR pipeline as the unfiltered reference -> same latency;
                # the ref-vs-out alignment already absorbed the oracle offset.
                corr, eratio = residual_presence(ref, fref, out)
                res.metrics["importance"].update({
                    "residual_presence_corr": round(corr, 4),
                    "residual_energy_ratio": round(eratio, 4),
                })
                if corr >= PRESENCE_CORR:
                    res.findings.append(_mk(
                        "S-331",
                        f"the rendered contribution of below-threshold objects "
                        f"({', '.join(names)}) is present in the output "
                        f"(matched-filter corr {corr:.2f} vs threshold "
                        f"{PRESENCE_CORR}) despite importance threshold "
                        f"{importance_threshold} — filtering was not applied "
                        f"to the audio",
                        where=f"threshold {importance_threshold}",
                        expected="below-threshold objects absent from output",
                        found=f"residual present (corr {corr:.2f}, "
                              f"{eratio:.0%} of reference energy)"))
        return res
    except Exception as e:  # CI contract: never crash
        res.execution_error = f"{type(e).__name__}: {e}"
        return res
    finally:
        if workdir is None:
            shutil.rmtree(wd, ignore_errors=True)


def render_json(res: ADMCompareResult, indent: int = 2) -> str:
    return json.dumps(res.to_dict(), indent=indent)


def render_text(res: ADMCompareResult) -> str:
    d = res.to_dict()
    lines = [f"sentinel adm-compare — {d['result']}",
             f"  source : {res.source}", f"  output : {res.output}",
             f"  layout : {res.layout}"]
    if res.execution_error:
        lines.append(f"  error  : {res.execution_error}")
    for f in res.findings:
        lines.append(f"  [{f.severity.label}] {f.check_id}: {f.message}")
    if not res.findings and not res.execution_error:
        lines.append("  no findings — output is consistent with the source render")
    lines.append(f"  metrics: {json.dumps(res.metrics)}")
    return "\n".join(lines)
