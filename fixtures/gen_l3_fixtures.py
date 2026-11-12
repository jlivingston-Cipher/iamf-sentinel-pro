#!/usr/bin/env python3
"""Generate the Phase-2 L3 acceptance pair: the WP1 F4 essence-misroute file.

Encodes the 7.1.4 identifier-tone essence twice through FFmpeg's stream_group
path (subprocess-only, ADR-4):

  7dot1dot4_correct.iamf     — coupled pairs first, C/LFE last (the IAMF rule)
  7dot1dot4_wrongorder.iamf  — the "natural reading" order (L/R)(C)(LFE)(pairs…)
                               that WP1 proved encodes silently corrupt (F4)

Both files are descriptor-identical (same declared 7.1.4 topology) — only the
essence routing differs. Phase-1 checks cannot tell them apart; L3
channel-identity must. Requires the rebuilt toolchain (FFmpeg n8.1.2 + WP1
essence); run after build_toolchain.sh:

    python3 -m fixtures.gen_l3_fixtures [outdir]
"""

import os
import subprocess
import sys

ROOT = os.environ.get("SENTINEL_TOOLCHAIN", "/home/claude/iamf-wp1")
FF = f"{ROOT}/bin/ffmpeg-install/bin/ffmpeg"
ESSENCE = f"{ROOT}/essence/7dot1dot4_48k_24b.wav"
OUT = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(os.environ.get("SENTINEL_SAMPLES") or "/tmp/sentinel_build",
                    "wp2-l3"))

# WAV order: L R C LFE Lss Rss Lrs Rrs Ltf Rtf Ltb Rtb
CORRECT = [
    ("frontLR", "stereo|c0=c0|c1=c1", "128k"),
    ("sideLR", "stereo|c0=c4|c1=c5", "128k"),
    ("backLR", "stereo|c0=c6|c1=c7", "128k"),
    ("topfLR", "stereo|c0=c8|c1=c9", "128k"),
    ("topbLR", "stereo|c0=c10|c1=c11", "128k"),
    ("center", "mono|c0=c2", "64k"),
    ("lfe", "mono|c0=c3", "64k"),
]
# The natural-but-wrong reading: C and LFE right after the fronts (F4).
WRONG = [CORRECT[0], CORRECT[5], CORRECT[6]] + CORRECT[1:5]


def encode(streams, outfile):
    fc, maps = [], []
    for label, pan, _br in streams:
        fc.append(f"[0:a]pan={pan}[{label}]")
        maps += ["-map", f"[{label}]"]
    cmd = [FF, "-y", "-hide_banner", "-i", ESSENCE,
           "-filter_complex", ";".join(fc)] + maps
    for i, (_l, _p, br) in enumerate(streams):
        cmd += [f"-c:a:{i}", "libopus", f"-b:a:{i}", br]
    st = "".join(f":st={i}" for i in range(len(streams)))
    ae = (f"type=iamf_audio_element:id=1{st}:audio_element_type=channel"
          f",layer=ch_layout=7.1.4")
    mp = ("type=iamf_mix_presentation:id=2:stg=0:annotations=en-us=TestMix,"
          "submix=parameter_id=100:parameter_rate=48000:default_mix_gain=0.0"
          "|element=stg=0:parameter_id=101:parameter_rate=48000"
          ":default_mix_gain=0.0:annotations=en-us=MainElement"
          "|layout=sound_system=stereo|layout=sound_system=7.1.4")
    cmd += ["-stream_group", ae, "-stream_group", mp]
    for i in range(len(streams)):
        cmd += ["-streamid", f"{i}:{i}"]
    cmd.append(outfile)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(outfile) and os.path.getsize(outfile) > 0
    print(f"{'OK ' if ok else 'FAIL'} {outfile}")
    if not ok:
        print("\n".join(r.stderr.strip().splitlines()[-8:]))
    return ok


def main() -> int:
    if not os.access(FF, os.X_OK):
        print(f"toolchain ffmpeg not found at {FF}")
        return 2
    if not os.path.exists(ESSENCE):
        print(f"essence not found at {ESSENCE} (run gen_essence.py / build S8)")
        return 2
    os.makedirs(OUT, exist_ok=True)
    ok = encode(CORRECT, os.path.join(OUT, "7dot1dot4_correct.iamf"))
    ok &= encode(WRONG, os.path.join(OUT, "7dot1dot4_wrongorder.iamf"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
