#!/usr/bin/env python3
"""Phase-2 acceptance evidence collector.

Runs the full L3 pipeline over the real WP1/WP3 samples + the regenerated F4
pair, records per-layout measured-vs-declared numbers from BOTH oracles, the
own-BS.1770-vs-ffmpeg cross-check on the actual decoded renders (WP1 G5 gate),
timing, and the fired checks — machine-readable ground truth for doc 19 and
the expectation-verdict table.

    python3 -m fixtures.acceptance_run [out.json]
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

from sentinel_pro import dsp
from sentinel.engine import validate
from sentinel_pro.oracle import Toolchain, SOUND_SYSTEM_TO_LAYOUT
from sentinel.layouts import LAYOUT_TYPE_SS

SAMPLES = [
    ("stereo_iamftools.iamf", "/tmp/sentinel_build/wp1/wp1-samples/stereo_iamftools.iamf", "generic"),
    ("stereo_ffmpeg.iamf", "/tmp/sentinel_build/wp1/wp1-samples/stereo_ffmpeg.iamf", "generic"),
    ("cd_bed_stereo.iamf", "/tmp/sentinel_build/wp3/wp3-samples/cd_bed_stereo.iamf", "generic"),
    ("dlb_obj_static1.iamf", "/tmp/sentinel_build/wp3/wp3-samples/dlb_obj_static1.iamf", "generic"),
    ("youtube_candidate_5dot1.mp4", "/tmp/sentinel_build/wp1/wp1-samples/youtube_candidate_5dot1.mp4", "youtube"),
    ("7dot1dot4_correct.iamf", "/tmp/sentinel_build/wp2-l3/7dot1dot4_correct.iamf", "generic"),
    ("7dot1dot4_wrongorder.iamf", "/tmp/sentinel_build/wp2-l3/7dot1dot4_wrongorder.iamf", "generic"),
]

FFMPEG = shutil.which("ffmpeg")


def ffmpeg_integrated(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path,
                        "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    hits = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr)
    return float(hits[-1]) if hits else None


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "phase2_acceptance.json"
    evidence = {"toolchain": {}, "samples": {}}
    tc = Toolchain.discover()
    evidence["toolchain"] = {"decoder_main": tc.decoder_main, "iamfdec": tc.iamfdec}

    for name, path, profile in SAMPLES:
        if not os.path.exists(path):
            evidence["samples"][name] = {"error": "missing"}
            continue
        entry = {"profile": profile}
        t0 = time.monotonic()
        rep = validate(path, profile=profile, toolchain=Toolchain.discover())
        entry["l3_wall_seconds"] = round(time.monotonic() - t0, 2)
        entry["exit_code"] = rep.exit_code()
        entry["findings"] = [f.to_dict() for f in rep.findings]

        # measured-vs-declared detail from both oracles + ffmpeg cross-check
        mtc = Toolchain.discover()
        oracle_input = mtc.prepare_input(path)
        layouts = {}
        for mp in rep.model.mix_presentations:
            for sm in mp.sub_mixes:
                for ll in sm.layouts:
                    if ll.layout_type != LAYOUT_TYPE_SS:
                        continue
                    lay = SOUND_SYSTEM_TO_LAYOUT.get(
                        ll.sound_system if ll.sound_system is not None else -1)
                    if lay is None or ll.label in layouts:
                        continue
                    ld = {"declared_integrated": ll.integrated_loudness,
                          "declared_digital_peak": ll.digital_peak,
                          "declared_true_peak": ll.true_peak}
                    for dec in mtc.oracles():
                        d = mtc.decode(oracle_input, lay, dec)
                        if d.ok:
                            sr, pcm = dsp.read_wav(d.wav_path)
                            ld[dec] = {
                                "integrated": round(dsp.integrated_loudness(
                                    pcm, sr, dsp.CHANNEL_WEIGHTS.get(lay)), 3),
                                "digital_peak": round(dsp.digital_peak_dbfs(pcm), 3),
                                "true_peak": round(dsp.true_peak_dbtp(pcm), 3),
                            }
                            if dec == "decoder_main" and FFMPEG:
                                ff = ffmpeg_integrated(d.wav_path)
                                if ff is not None:
                                    ld["ffmpeg_ebur128_integrated"] = ff
                                    ld["own_vs_ffmpeg_LU"] = round(
                                        abs(ld[dec]["integrated"] - ff), 3)
                        else:
                            ld[dec] = {"decode_failed": d.note or d.returncode}
                    layouts[ll.label] = ld
        entry["layouts"] = layouts
        mtc.cleanup()
        evidence["samples"][name] = entry
        print(f"{name}: exit={entry['exit_code']} "
              f"checks={sorted({f['check_id'] for f in entry['findings']})} "
              f"({entry['l3_wall_seconds']}s)")

    with open(out_path, "w") as fh:
        json.dump(evidence, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
