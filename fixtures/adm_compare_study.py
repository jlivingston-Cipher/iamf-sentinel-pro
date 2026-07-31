#!/usr/bin/env python3
"""Phase-3 acceptance study — source-referenced QC (E-B1/E-B2/E-B3).

Encodes the three pre-registered WP3 corpus cases with iamf-tools encoder_main,
then runs sentinel adm-compare on each and records verdicts vs expectations.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinel_pro.adm_compare import compare, find_ear     # noqa: E402
from sentinel_pro.oracle import Toolchain                  # noqa: E402

ROOT = os.environ.get("SENTINEL_TOOLCHAIN", "/home/claude/iamf-wp1")
CORPUS = os.environ.get("SENTINEL_ADM_CORPUS", f"{ROOT}/adm-corpus")
ENC = f"{ROOT}/src/build-iamf/encoder_main"
OUT = f"{ROOT}/out/phase3-adm-compare"

CASES = [
    # (tag, encode file, enc thr, cmp thr, expected finding, compare-source file)
    ("E-B1-gainauto", "dlb_obj_gainauto", None, None, "S-330", None),
    ("E-B2-importance", "dlb_importance", 5, 5, "S-331", None),
    # negative control: output encoded from a source whose below-threshold
    # tracks were silenced (simulates a WORKING importance filter); compared
    # against the ORIGINAL source with the same threshold -> S-331 must stay quiet.
    ("E-B2n-negative-control", "dlb_importance_prefiltered", None, 5, None,
     "dlb_importance"),
    ("E-B3-clean-bed", "cd_bed_5dot1", None, None, None, None),
]
LAYOUTS = ["5.1", "2.0"]     # primary first; 2.0 as secondary evidence


def encode(name: str, threshold, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cmd = [ENC, f"--adm_filename={CORPUS}/{name}.wav",
           "--adm_profile_version=base", f"--output_iamf_directory={outdir}"]
    if threshold is not None:
        cmd.append(f"--adm_importance_threshold={threshold}")
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    produced = sorted(f for f in os.listdir(outdir) if f.endswith(".iamf"))
    return {"ok": r.returncode == 0 and bool(produced), "rc": r.returncode,
            "wall_s": round(time.monotonic() - t0, 2),
            "iamf": os.path.join(outdir, produced[0]) if produced else None,
            "stderr_tail": (r.stderr or r.stdout).strip().splitlines()[-3:]}


def main():
    tc = Toolchain.discover(ROOT)
    ear = find_ear()
    assert tc.available, "toolchain not available"
    assert ear, "ear-render not found"
    os.makedirs(OUT, exist_ok=True)
    results = {"toolchain": ROOT, "ear": ear, "cases": []}

    # build the negative-control source: below-threshold tracks silenced
    from sentinel_pro.adm_compare import below_threshold_tracks, silence_tracks
    orig = f"{CORPUS}/dlb_importance.wav"
    tracks, _names = below_threshold_tracks(orig, 5)
    silence_tracks(orig, tracks, f"{CORPUS}/dlb_importance_prefiltered.wav")

    for tag, name, enc_thr, cmp_thr, expected, cmp_src in CASES:
        enc = encode(name, enc_thr, os.path.join(OUT, tag))
        entry = {"tag": tag, "corpus_file": name, "encode": enc,
                 "expected_finding": expected, "layouts": {}}
        if enc["ok"]:
            for layout in LAYOUTS:
                res = compare(f"{CORPUS}/{cmp_src or name}.wav", enc["iamf"],
                              layout=layout,
                              importance_threshold=cmp_thr, toolchain=tc,
                              ear_bin=ear)
                d = res.to_dict()
                got = sorted({f["check_id"] for f in d["findings"]})
                d["s33x_found"] = [c for c in got if c.startswith("S-33")]
                entry["layouts"][layout] = d
            primary = entry["layouts"][LAYOUTS[0]]
            found = primary["s33x_found"]
            if expected is None:
                entry["verdict"] = "MATCH" if not found else f"MISS (unexpected {found})"
            else:
                entry["verdict"] = "MATCH" if expected in found else \
                    f"MISS (wanted {expected}, got {found})"
        else:
            entry["verdict"] = f"ENCODE-FAIL rc={enc['rc']}"
        results["cases"].append(entry)
        print(f"{tag}: {entry['verdict']}", flush=True)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "phase3_admcompare_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("written", os.path.normpath(path))
    ok = all(e["verdict"] == "MATCH" for e in results["cases"])
    print("STUDY:", "ALL-MATCH" if ok else "HAS-MISSES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
