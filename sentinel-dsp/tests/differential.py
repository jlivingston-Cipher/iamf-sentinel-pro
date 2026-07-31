#!/usr/bin/env python3
"""sentinel-dsp vs dsp.py differential (ADR-6b acceptance gate).

Pre-registered gates (doc 32): integrated loudness within 0.001 LU; true/digital
peak within 0.01 dB; identical duplicate-pair and silent-channel verdicts;
energy-profile and off-diagonal correlation within 1e-6; on synthetic vectors
(this driver) and real decoder-oracle renders (--renders DIR).

Usage: differential.py --binary PATH --oss OSS_ROOT --pro PRO_ROOT
                       [--renders DIR:LAYOUT ...] [--out results.json]
"""
import argparse
import json
import os
import struct
import subprocess
import sys



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--oss", required=True)
    ap.add_argument("--pro", required=True)
    ap.add_argument("--renders", nargs="*", default=[],
                    help="extra real WAVs as path:layout pairs")
    ap.add_argument("--out", default="differential_results.json")
    ap.add_argument("--workdir", default="/tmp/dsp_vectors")
    args = ap.parse_args()

    sys.path.insert(0, args.oss)
    sys.path.insert(0, args.pro)
    from sentinel_pro import dsp  # noqa: E402
    import numpy as np            # noqa: E402

    os.makedirs(args.workdir, exist_ok=True)
    SR = 48000


    def write_wav(path, pcm, sr=SR, bits=24):
        """pcm: float array (frames, ch) in [-1,1)."""
        frames, ch = pcm.shape
        if bits == 24:
            ints = np.clip(np.round(pcm * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
            flat = ints.reshape(-1)
            b = np.empty((flat.size, 3), dtype=np.uint8)
            u = flat.astype(np.int64) & 0xFFFFFF
            b[:, 0] = u & 0xFF
            b[:, 1] = (u >> 8) & 0xFF
            b[:, 2] = (u >> 16) & 0xFF
            raw = b.tobytes()
            fmt = struct.pack("<HHIIHH", 1, ch, sr, sr * ch * 3, ch * 3, 24)
        elif bits == 16:
            ints = np.clip(np.round(pcm * (1 << 15)), -(1 << 15), (1 << 15) - 1).astype("<i2")
            raw = ints.reshape(-1).tobytes()
            fmt = struct.pack("<HHIIHH", 1, ch, sr, sr * ch * 2, ch * 2, 16)
        elif bits == 32:  # float32
            raw = pcm.reshape(-1).astype("<f4").tobytes()
            fmt = struct.pack("<HHIIHH", 3, ch, sr, sr * ch * 4, ch * 4, 32)
        else:
            raise ValueError(bits)
        with open(path, "wb") as fh:
            data_len = len(raw)
            fh.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + data_len) + b"WAVE")
            fh.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
            fh.write(b"data" + struct.pack("<I", data_len) + raw)


    def sine(freq, secs=10.0, level_db=-18.0, sr=SR, phase=0.0):
        t = np.arange(int(secs * sr)) / sr
        return (10 ** (level_db / 20.0)) * np.sin(2 * np.pi * freq * t + phase)


    # ---------------------------------------------------------------- vectors
    vectors = []  # (name, path, layout|None, variant)

    # V1: stereo sine, the reference case
    p = np.stack([sine(997), sine(1499)], axis=1)
    f = os.path.join(args.workdir, "v1_stereo.wav"); write_wav(f, p)
    vectors.append(("v1_stereo_sine", f, "2.0", "primary"))

    # V2: 5.1 unique sines (WP1 convention), LFE at 40 Hz
    p = np.stack([sine(313 + 139 * k) for k in range(5)] + [sine(40)], axis=1)
    p = p[:, [0, 1, 2, 5, 3, 4]]  # L R C LFE Ls Rs
    f = os.path.join(args.workdir, "v2_5dot1.wav"); write_wav(f, p)
    vectors.append(("v2_5dot1_unique", f, "5.1", "primary"))

    # V3: 7.1.4 unique sines — both weight variants
    p = np.stack([sine(313 + 139 * k) for k in range(12)], axis=1)
    f = os.path.join(args.workdir, "v3_7dot1dot4.wav"); write_wav(f, p)
    vectors.append(("v3_714_primary", f, "7.1.4", "primary"))
    vectors.append(("v3_714_ffmpeg", f, "7.1.4", "ffmpeg"))

    # V4: gating exercise — loud bursts / silence / quiet section
    t = np.arange(int(10.0 * SR)) / SR
    x = sine(997, 10.0, -16.0)
    x[int(3 * SR):int(6 * SR)] *= 10 ** ((-80 + 16) / 20.0)   # below absolute gate
    x[int(6 * SR):int(8 * SR)] *= 10 ** ((-45 + 16) / 20.0)   # below relative gate
    p = np.stack([x, x * 0.9], axis=1)
    f = os.path.join(args.workdir, "v4_gating.wav"); write_wav(f, p)
    vectors.append(("v4_gating", f, "2.0", "primary"))

    # V5: duplicates + one silent channel (5.1: Rs := Ls, C silent)
    base = [sine(313 + 139 * k) for k in range(6)]
    base[2] = np.zeros_like(base[2])       # C silent
    base[5] = base[4].copy()               # Rs duplicates Ls
    p = np.stack(base, axis=1)
    f = os.path.join(args.workdir, "v5_dup_silent.wav"); write_wav(f, p)
    vectors.append(("v5_dup_silent", f, "5.1", "primary"))

    # V6: intersample true peak — fs/4 sine with 45° phase, near full scale
    x = 0.99 * np.sin(2 * np.pi * 12000 * t + np.pi / 4)
    p = np.stack([x], axis=1)
    f = os.path.join(args.workdir, "v6_truepeak.wav"); write_wav(f, p, bits=32)
    vectors.append(("v6_truepeak_f32", f, None, "primary"))

    # V7: short file (< 400 ms → padding path), 16-bit
    p = np.stack([sine(997, 0.25), sine(1499, 0.25)], axis=1)
    f = os.path.join(args.workdir, "v7_short.wav"); write_wav(f, p, bits=16)
    vectors.append(("v7_short_16bit", f, "2.0", "primary"))

    # V8: all-silent program
    p = np.zeros((SR * 2, 6))
    f = os.path.join(args.workdir, "v8_silent.wav"); write_wav(f, p)
    vectors.append(("v8_all_silent", f, "5.1", "primary"))

    # V9: noise (broadband, seeds the filters differently from sines)
    rng = np.random.default_rng(20260721)
    p = 0.1 * rng.standard_normal((SR * 10, 2))
    f = os.path.join(args.workdir, "v9_noise.wav"); write_wav(f, p, bits=32)
    vectors.append(("v9_noise_f32", f, "2.0", "primary"))

    for spec in args.renders:
        path, _, layout = spec.rpartition(":")
        vectors.append((f"render_{os.path.basename(path)}", path, layout or None, "primary"))

    # ---------------------------------------------------------------- run both
    GATE_LU = 0.001
    GATE_DB = 0.01
    GATE_EPS = 1e-6

    results, worst = [], {"il": 0.0, "dp": 0.0, "tp": 0.0, "offdiag": 0.0, "energy": 0.0}
    fails = 0
    for name, path, layout, variant in vectors:
        cmd = [args.binary, path, "--variant", variant]
        if layout:
            cmd += ["--layout", layout]
        cpp = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)

        sr, pcm = dsp.read_wav(path)
        table = dsp.FFMPEG_WEIGHTS if variant == "ffmpeg" else dsp.CHANNEL_WEIGHTS
        weights = table.get(layout) if layout else None
        il = dsp.integrated_loudness(pcm, sr, weights)
        dp = dsp.digital_peak_dbfs(pcm)
        tp = dsp.true_peak_dbtp(pcm)
        dups = [[i, j, float(c)] for i, j, c in dsp.duplicate_pairs(pcm)]
        sil = dsp.silent_channels(pcm)
        off = dsp.mean_offdiag_correlation(pcm)
        ep = dsp.energy_vector_profile(pcm, sr, layout) if layout else {"windows": 0}

        d_il = abs(cpp["integrated_lufs"] - il)
        d_dp = abs(cpp["digital_peak_dbfs"] - dp)
        d_tp = abs(cpp["true_peak_dbtp"] - tp)
        d_off = abs(cpp["correlation"]["mean_offdiag"] - off)
        same_dups = ([[i, j] for i, j, _ in dups] ==
                     [[i, j] for i, j, _ in cpp["correlation"]["duplicate_pairs"]])
        same_sil = (sil == cpp["correlation"]["silent_channels"])
        if ep.get("median_dev") is not None and cpp["energy_profile"]["median_dev"] is not None:
            d_ep = max(abs(ep["median_dev"] - cpp["energy_profile"]["median_dev"]),
                       abs(ep["p95_dev"] - cpp["energy_profile"]["p95_dev"]))
            same_ep_shape = ep["windows"] == cpp["energy_profile"]["windows"]
        else:
            d_ep = 0.0
            same_ep_shape = (ep.get("windows", 0) == cpp["energy_profile"]["windows"])

        ok = (d_il <= GATE_LU and d_dp <= GATE_DB and d_tp <= GATE_DB and
              d_off <= GATE_EPS and d_ep <= GATE_EPS and same_dups and same_sil and same_ep_shape)
        fails += 0 if ok else 1
        worst["il"] = max(worst["il"], d_il); worst["dp"] = max(worst["dp"], d_dp)
        worst["tp"] = max(worst["tp"], d_tp); worst["offdiag"] = max(worst["offdiag"], d_off)
        worst["energy"] = max(worst["energy"], d_ep)
        results.append({
            "vector": name, "ok": ok, "layout": layout, "variant": variant,
            "py": {"il": il, "dp": dp, "tp": tp, "dups": dups, "silent": sil},
            "cpp": {"il": cpp["integrated_lufs"], "dp": cpp["digital_peak_dbfs"],
                    "tp": cpp["true_peak_dbtp"],
                    "dups": cpp["correlation"]["duplicate_pairs"],
                    "silent": cpp["correlation"]["silent_channels"]},
            "delta": {"il_lu": d_il, "dp_db": d_dp, "tp_db": d_tp,
                      "offdiag": d_off, "energy": d_ep},
        })
        print(f"{'PASS' if ok else 'FAIL'} {name:24s} ΔIL={d_il:.2e} LU  ΔDP={d_dp:.2e} dB"
              f"  ΔTP={d_tp:.2e} dB  dups={'=' if same_dups else '≠'} silent={'=' if same_sil else '≠'}")

    doc = {"gates": {"il_lu": GATE_LU, "peak_db": GATE_DB, "eps": GATE_EPS},
           "vectors": len(results), "failures": fails, "worst_delta": worst,
           "results": results}
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n{len(results) - fails}/{len(results)} vectors within gates; "
          f"worst ΔIL={worst['il']:.2e} LU, ΔTP={worst['tp']:.2e} dB → {args.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
