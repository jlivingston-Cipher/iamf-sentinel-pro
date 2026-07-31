"""Phase-3 source-referenced QC tests.

Pure-function tests always run (numpy-gated); the end-to-end compare() runs are
the acceptance study (needs EAR + toolchain) and are exercised by
fixtures/adm_compare_study.py rather than here.
"""
import os

import pytest

np = pytest.importorskip("numpy")

from sentinel_pro.adm_compare import (_read_wav, _find_chunks, below_threshold_tracks,
                                  silence_tracks, _envelope_db, EAR_LAYOUTS)
from sentinel.findings import REGISTRY

CORPUS = os.environ.get("SENTINEL_ADM_CORPUS", "/home/claude/iamf-wp1/adm-corpus")
IMP = os.path.join(CORPUS, "dlb_importance.wav")

needs_corpus = pytest.mark.skipif(not os.path.exists(IMP),
                                  reason="WP3 ADM corpus not generated")


def test_s33x_registered():
    for cid, frefs in (("S-330", ("F16", "F18")), ("S-331", ("F17",)),
                       ("S-332", ("F21",))):
        assert cid in REGISTRY
        assert REGISTRY[cid].f_refs == frefs


def test_ear_layouts_cover_oracle_names():
    from sentinel_pro.oracle import SOUND_SYSTEM_TO_LAYOUT
    assert {"2.0", "5.1", "7.1.4"} <= set(EAR_LAYOUTS)
    assert set(EAR_LAYOUTS) <= set(SOUND_SYSTEM_TO_LAYOUT.values())


def test_envelope_db_tracks_gain_step():
    rate = 48000
    x = np.ones((rate, 1), dtype=np.float32) * 0.1
    x[rate // 2:] *= 10 ** (-6 / 20)          # -6 dB step at 0.5 s
    env = _envelope_db(x, rate)
    assert abs((env[0] - env[-1]) - 6.0) < 0.1


def test_residual_presence_discriminates():
    from sentinel_pro.adm_compare import residual_presence, PRESENCE_CORR
    t = np.arange(48000) / 48000.0
    a = np.sin(2 * np.pi * 313 * t)[:, None].astype(np.float64)   # kept object
    b = np.sin(2 * np.pi * 452 * t)[:, None].astype(np.float64)   # below-threshold
    ref, fref = a + b, a
    out_shipped = 0.5 * (a + b)      # filter ignored (+ arbitrary fold gain)
    out_filtered = 0.7 * a           # filter honored (+ different gain)
    c1, _ = residual_presence(ref, fref, out_shipped)
    c2, _ = residual_presence(ref, fref, out_filtered)
    assert c1 >= PRESENCE_CORR and c1 > 0.6
    assert c2 < 0.05
    # gain-invariance: scaling the output must not change the verdict
    c3, _ = residual_presence(ref, fref, 0.05 * (a + b))
    assert abs(c3 - c1) < 1e-9


@needs_corpus
def test_below_threshold_tracks_importance_file():
    # dlb_importance: 8 objects, importance [10,9,8,7,5,4,3,1] (WP3 gen).
    tracks, names = below_threshold_tracks(IMP, 5)
    assert len(tracks) == 3                    # importance 4, 3, 1
    tracks2, _ = below_threshold_tracks(IMP, 11)
    assert len(tracks2) == 8                   # everything below 11
    tracks0, _ = below_threshold_tracks(IMP, 0)
    assert tracks0 == []


@needs_corpus
def test_silence_tracks_zeroes_only_targets(tmp_path):
    tracks, _ = below_threshold_tracks(IMP, 5)
    out = str(tmp_path / "filtered.wav")
    silence_tracks(IMP, tracks, out)
    a, rate = _read_wav(IMP)
    b, rate2 = _read_wav(out)
    assert rate == rate2 and a.shape == b.shape
    for t in range(a.shape[1]):
        rms_b = float(np.sqrt((b[:, t] ** 2).mean()))
        if (t + 1) in tracks:
            assert rms_b == 0.0
        else:
            assert abs(rms_b - float(np.sqrt((a[:, t] ** 2).mean()))) < 1e-9
    # chunk layout preserved byte-for-byte outside data
    ca = _find_chunks(open(IMP, 'rb').read())
    cb = _find_chunks(open(out, 'rb').read())
    assert ca.keys() == cb.keys() and ca[b'axml'] == cb[b'axml']
