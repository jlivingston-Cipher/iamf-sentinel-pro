"""Phase 2 acceptance tests — L3 rendered QC, DSP cross-checks, R4 depth, R7 full.

DSP tests run everywhere (numpy + ffmpeg on PATH). Oracle tests need the
rebuilt WP1/WP3 toolchain (decoder_main/iamfdec) and the real sample
bitstreams; they skip cleanly when absent so the suite stays green in
toolchain-less environments (the Phase-1 property).
"""

import os
import shutil
import struct
import subprocess
import re

import pytest

np = pytest.importorskip("numpy")

from sentinel_pro import dsp                                    # noqa: E402
from sentinel_pro.oracle import Toolchain                       # noqa: E402
from sentinel.engine import validate                        # noqa: E402
from sentinel.findings import Severity                      # noqa: E402
from sentinel.diff import diff_files                        # noqa: E402
from sentinel.container.mp4 import parse_mp4                # noqa: E402

SAMPLE_ROOTS = [
    "/tmp/sentinel_build/wp1/wp1-samples",
    "/tmp/sentinel_build/wp3/wp3-samples",
    "/tmp/sentinel_build/wp2-l3",          # Phase-2 regenerated fixtures (F4 pair)
    os.path.join(os.path.dirname(__file__), "..", "samples-l3"),   # shipped copies
]

FFMPEG = shutil.which("ffmpeg")

_WP1 = "/home/claude/iamf-wp1"           # sandbox WP1 layout (test-tier only)


def _discover() -> Toolchain:
    """Toolchain via the library rules, falling back to the WP1 sandbox layout."""
    tc = Toolchain.discover()
    if tc.decoder_main is None and tc.iamfdec is None and os.path.isdir(_WP1):
        tc = Toolchain.discover(_WP1)
    return tc

TOOLCHAIN = _discover()

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
needs_oracle = pytest.mark.skipif(not TOOLCHAIN.available,
                                  reason="decoder oracle not built")


def sample(name: str) -> str:
    for root in SAMPLE_ROOTS:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    pytest.skip(f"sample {name} not available")


def checks_fired(report):
    return {f.check_id for f in report.findings}


def write_wav(path, pcm, sr=48000):
    data = (np.clip(pcm, -1, 1) * (2**31 - 1)).astype("<i4").tobytes()
    ch = pcm.shape[1]
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 4, ch * 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def ffmpeg_measure(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path,
                        "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8")
    integ = float(re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr)[-1])
    tp = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", r.stderr)
    return integ, (float(tp[-1]) if tp else None)


# ---------------------------------------------------------------------------
# Own BS.1770-4 vs ffmpeg ebur128 — the WP1 G5 cross-check gate (E1)
# ---------------------------------------------------------------------------

def _tones(nch, sr=48000, secs=10, level_db=-18.0):
    t = np.arange(sr * secs) / sr
    pcm = np.stack([np.sin(2 * np.pi * (313 + 139 * k) * t) for k in range(nch)],
                   axis=1) * 10 ** (level_db / 20)
    if nch >= 4:
        pcm[:, 3] = np.sin(2 * np.pi * 40 * t) * 10 ** (level_db / 20)
    return pcm


def test_k_weighting_matches_spec_table():
    (bs, a_s), (bh, ah) = dsp._k_weighting_coeffs(48000)
    assert np.allclose(bs, [1.53512485958697, -2.69169618940638, 1.19839281085285],
                       atol=1e-10)
    assert np.allclose(a_s, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-10)
    assert np.allclose(ah, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-10)


def test_reference_level_997hz():
    """A 0 dBFS 997 Hz sine measures -3.01 LKFS (the canonical BS.1770 anchor)."""
    sr = 48000
    t = np.arange(sr * 10) / sr
    x = np.sin(2 * np.pi * 997 * t)[:, None]
    assert abs(dsp.integrated_loudness(x, sr, [1.0]) - (-3.01)) < 0.02


@needs_ffmpeg
@pytest.mark.parametrize("nch,layout", [(2, "2.0"), (6, "5.1"), (8, "7.1")])
def test_crosscheck_ffmpeg_g5_gate(nch, layout, tmp_path):
    """Engine cross-check (filters/gating), so ffmpeg's OWN weight variant is
    used: identical to the primary table for 2.0/5.1, but ffmpeg over-weights
    7.1 rears at 1.41 (BACK_MASK, F29) where BS.1770-4 says 1.00 (F33)."""
    pcm = _tones(nch)
    p = str(tmp_path / f"xc_{layout.replace('.', '_')}.wav")
    write_wav(p, pcm)
    ff_i, ff_tp = ffmpeg_measure(p)
    own = dsp.integrated_loudness(pcm, 48000, dsp.FFMPEG_WEIGHTS[layout])
    assert abs(ff_i - own) <= 0.06, f"{layout}: ffmpeg {ff_i} vs own {own}"
    own_tp = dsp.true_peak_dbtp(pcm)
    assert abs(ff_tp - own_tp) <= 0.3


@needs_ffmpeg
def test_crosscheck_ffmpeg_7_1_4_convention(tmp_path):
    """12-ch: ffmpeg's BACK_MASK weights (side/back/top-back 1.41) reproduce its
    figure; the primary table follows BS.1770-4 Tables 4/5 (F33) and matches
    the declared-value producer (see test_l3_iamftools_7_1_4_native_convention)."""
    pcm = _tones(12)
    p = str(tmp_path / "xc_7_1_4.wav")
    # masked EXTENSIBLE so ffmpeg knows the layout
    mask = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x200 | 0x400 | \
        0x1000 | 0x4000 | 0x8000 | 0x20000
    data = (np.clip(pcm, -1, 1) * (2**31 - 1)).astype("<i4").tobytes()
    guid = struct.pack("<H", 1) + bytes([0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
                                         0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71])
    fmt = struct.pack("<HHIIHH", 0xFFFE, 12, 48000, 48000 * 48, 48, 32) + \
        struct.pack("<HHI", 22, 32, mask) + guid
    with open(p, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(data)) + data)
    ff_i, _ = ffmpeg_measure(p)
    # channel semantic order under the mask: FL FR FC LFE BL BR SL SR TFL TFR TBL TBR
    w = [1, 1, 1, 0, 1.41, 1.41, 1.41, 1.41, 1.0, 1.0, 1.41, 1.41]
    own = dsp.integrated_loudness(pcm, 48000, w)
    assert abs(ff_i - own) <= 0.06


def test_channel_weights_bs1770_conformance():
    """F33 regression pin: the primary tables follow BS.1770-4 Tables 4/5
    exactly. 1.41 requires ear level AND 60°–120° azimuth — so sides (M±090)
    and 5.1 surrounds (M±110) are 1.41, rears (M±135) are 1.00, heights are
    unconditionally 1.00, LFE 0. The ffmpeg variant table deliberately keeps
    ffmpeg's non-conformant BACK_MASK behaviour (F29) — the two tables must
    NOT agree on 7.1.x rears."""
    W = dsp.CHANNEL_WEIGHTS
    for lay in ("5.1", "5.1.2", "5.1.4"):
        assert W[lay][3] == 0.0 and W[lay][4:6] == [1.41, 1.41], lay
        assert all(w == 1.0 for w in W[lay][6:]), f"{lay}: heights must be 1.0"
    for lay in ("7.1", "7.1.2", "7.1.4"):
        assert W[lay][3] == 0.0, lay
        assert W[lay][4:6] == [1.41, 1.41], f"{lay}: Lss/Rss (M±090) must be 1.41"
        assert W[lay][6:8] == [1.0, 1.0], f"{lay}: Lrs/Rrs (M±135) must be 1.00 (F33)"
        assert all(w == 1.0 for w in W[lay][8:]), f"{lay}: heights must be 1.0"
    assert dsp.FFMPEG_WEIGHTS["7.1"][6:8] == [1.41, 1.41], \
        "ffmpeg variant must keep BACK_MASK rears (F29 calibration)"
    assert dsp.FFMPEG_WEIGHTS["7.1.4"][10:12] == [1.41, 1.41], \
        "ffmpeg variant must keep BACK_MASK top-backs (F29 calibration)"


def test_true_peak_intersample():
    """An 11997 Hz sine at -6 dBFS peaks between samples; 4x oversampling sees it."""
    sr = 48000
    t = np.arange(sr * 5) / sr
    x = (np.sin(2 * np.pi * 11997 * t) * 10 ** (-6 / 20))[:, None]
    assert abs(dsp.true_peak_dbtp(x) - (-6.0)) < 0.15
    assert dsp.true_peak_dbtp(x) > dsp.digital_peak_dbfs(x) - 0.01


# ---------------------------------------------------------------------------
# Channel-identity / spatial heuristics (unit level)
# ---------------------------------------------------------------------------

def test_duplicate_and_silent_detection():
    pcm = _tones(6)
    pcm[:, 4] = pcm[:, 2]          # C duplicated into Ls (the F4 shape)
    pcm[:, 5] = 0.0                # Rs lost
    dups = dsp.duplicate_pairs(pcm)
    assert any({i, j} == {2, 4} for i, j, _ in dups)
    assert dsp.silent_channels(pcm) == [5]


def test_no_false_duplicates_on_distinct_tones():
    pcm = _tones(12)
    assert dsp.duplicate_pairs(pcm) == []
    assert dsp.silent_channels(pcm) == []


def test_origin_collapse_signature():
    sr = 48000
    t = np.arange(sr * 5) / sr
    mono = np.sin(2 * np.pi * 313 * t) * 10 ** (-18 / 20)
    collapsed = np.stack([mono] * 12, axis=1)          # everything identical
    prof = dsp.energy_vector_profile(collapsed, sr, "7.1.4")
    assert prof["windows"] >= 4 and prof["p95_dev"] < 0.05
    assert dsp.mean_offdiag_correlation(collapsed) > 0.95
    # distinct-tone essence must NOT look degenerate
    healthy = _tones(12)
    assert dsp.mean_offdiag_correlation(healthy) < 0.95


# ---------------------------------------------------------------------------
# L3 end-to-end against the real WP1/WP3 samples (E2) — needs the oracle
# ---------------------------------------------------------------------------

@needs_oracle
def test_l3_iamftools_declared_match():
    """iamf-tools measured its own loudness; Sentinel must agree (E2 row 1)."""
    r = validate(sample("stereo_iamftools.iamf"), toolchain=_discover())
    assert r.execution_error is None
    fired = checks_fired(r)
    assert "S-310" not in fired and "S-312" not in fired
    assert r.exit_code() == 0


@needs_oracle
def test_l3_ffmpeg_zero_loudness_measured_mismatch():
    """The F7 file: declared 0.0, measured ~-18.9 -> S-310 FAIL (E2 row 2)."""
    r = validate(sample("stereo_ffmpeg.iamf"), toolchain=_discover())
    fired = checks_fired(r)
    assert "S-301" in fired          # Phase-1 descriptor check still fires
    assert "S-310" in fired          # Phase-2 measured half
    s310 = next(f for f in r.findings if f.check_id == "S-310")
    assert s310.severity == Severity.FAIL
    assert r.exit_code() == 1


@needs_oracle
def test_l3_wp3_cd_bed():
    """WP3 proved ADM-path loudness truthful to <=0.05 LU (E2 row 3)."""
    r = validate(sample("cd_bed_stereo.iamf"), toolchain=_discover())
    assert "S-310" not in checks_fired(r)


@needs_oracle
def test_l3_wp3_dolby_scene():
    """Dolby->3OA: stereo declared loudness truthful; S-302 WARN persists;
    no S-310/S-314 (E2 row 4). Scene renders legitimately drive symmetric
    speakers with identical waveforms, so bed-only identity checks must NOT
    fire (E4 refinement, executed)."""
    r = validate(sample("dlb_obj_static1.iamf"), toolchain=_discover())
    fired = checks_fired(r)
    assert "S-302" in fired
    assert "S-310" not in fired
    assert "S-314" not in fired
    assert "S-320" not in fired and "S-321" not in fired


@needs_oracle
def test_l3_youtube_mp4():
    """MP4 decode path + declared loudness match under the youtube profile (E2 row 5)."""
    r = validate(sample("youtube_candidate_5dot1.mp4"), profile="youtube",
                 toolchain=_discover())
    assert r.execution_error is None
    fired = checks_fired(r)
    assert "S-310" not in fired and "S-312" not in fired
    assert r.exit_code() == 0        # S-403 casing is WARN, not failing


@needs_oracle
def test_l3_iamftools_7_1_4_native_convention():
    """Calibration pin (E1/E2 resolution): an iamf-tools 7.1.4 encode with
    google-engine-declared NATIVE loudness must not trip S-310 — proving the
    primary CHANNEL_WEIGHTS convention (tops 1.0) matches the declared-value
    producer. S-311 WARN is allowed (F27: declared peak is pre-codec; Opus
    overshoot measured +3.46 dB on the top-back tones)."""
    r = validate(sample("iamftools/7dot1dot4.iamf"), toolchain=_discover())
    fired = checks_fired(r)
    assert "S-310" not in fired, [f.message for f in r.findings if f.check_id == "S-310"]
    assert "S-313" not in fired          # oracles agree
    assert r.exit_code() == 0


@needs_oracle
def test_l3_f4_essence_misroute_detected():
    """THE Phase-2 acceptance headline (E2 rows 6-7): the WP1 wrong-order
    encode is descriptor-clean but PCM-scrambled; channel identity sees it."""
    bad = sample("7dot1dot4_wrongorder.iamf")
    good = sample("7dot1dot4_correct.iamf")
    tc = _discover()
    r_bad = validate(bad, toolchain=tc)
    fired_bad = checks_fired(r_bad)
    assert "S-320" in fired_bad or "S-321" in fired_bad, \
        f"F4 essence misroute NOT caught; fired={fired_bad}"
    r_good = validate(good, toolchain=_discover())
    fired_good = checks_fired(r_good)
    assert "S-320" not in fired_good and "S-321" not in fired_good


@needs_oracle
def test_diff_render_equal_same_essence():
    """R7 full: the two stereo encodes carry the same essence -> RENDER-EQUAL,
    while the declared-loudness delta stays reported (E5)."""
    res = diff_files(sample("stereo_iamftools.iamf"), sample("stereo_ffmpeg.iamf"),
                     toolchain=_discover())
    assert res.render_compared
    assert res.structural_diffs == []
    assert res.render_diffs == [], f"render diffs: {[e.to_dict() for e in res.render_diffs]}"
    assert all(c >= 0.98 for c in res.render_channel_corr)
    assert res.loudness_diffs          # the F7 +18.84 delta
    assert res.exit_code() == 0


@needs_oracle
def test_diff_render_catches_scramble():
    """R7 full negative case: wrong-order vs correct 7.1.4 differ audibly."""
    res = diff_files(sample("7dot1dot4_correct.iamf"),
                     sample("7dot1dot4_wrongorder.iamf"),
                     toolchain=_discover())
    assert res.render_compared
    assert res.render_diffs, "scrambled pair reported RENDER-EQUAL"
    assert res.exit_code() == 1


# ---------------------------------------------------------------------------
# R4 container depth (E5)
# ---------------------------------------------------------------------------

def test_mp4_timing_collected():
    p = sample("youtube_candidate_5dot1.mp4")
    info = parse_mp4(open(p, "rb").read())
    soun = [t for t in info.tracks if t.handler == "soun"]
    assert soun and soun[0].stts, "no audio stts collected"
    assert soun[0].timescale


def test_av_duration_coherent_on_youtube_sample():
    r = validate(sample("youtube_candidate_5dot1.mp4"))
    assert "S-406" not in checks_fired(r)


# ------------------------------------------------- weights-default correction (doc 33)
def test_integrated_loudness_default_weights():
    """weights=None must mean unity per channel — the pre-fix expression built a
    2*nch vector (mono: +3.01 LU silently; stereo: ValueError). Caught by the
    sentinel-dsp C++ kernel differential (ADR-6b, doc 33)."""
    sr = 48000
    t = np.arange(sr * 5) / sr
    mono = (10 ** (-18 / 20)) * np.sin(2 * np.pi * 997 * t)[:, None]
    assert abs(dsp.integrated_loudness(mono, sr, None)
               - dsp.integrated_loudness(mono, sr, [1.0])) < 1e-12
    stereo = np.concatenate([mono, mono], axis=1)
    assert abs(dsp.integrated_loudness(stereo, sr, None)
               - dsp.integrated_loudness(stereo, sr, [1.0, 1.0])) < 1e-12
