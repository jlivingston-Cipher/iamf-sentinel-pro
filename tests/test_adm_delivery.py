"""The measured delivery checks (sentinel_pro.adm_delivery): S-522 LFE low-pass,
S-524 silent channels, S-525 a false 96 kHz, S-523 sync with the stereo master,
S-533 gapless boundaries. Every master here is synthesised with a known answer:
filters are applied as exact magnitude responses in the frequency domain, so
the shape under test is the shape written."""
import math
import os
import struct

import pytest

np = pytest.importorskip("numpy")

from sentinel_pro import adm_delivery as D  # noqa: E402

SR = 48000
BED = [("RoomCentricLeft", False), ("RoomCentricRight", False), ("RoomCentricCenter", False),
       ("RoomCentricLFE", True), ("RoomCentricLeftSideSurround", False),
       ("RoomCentricRightSideSurround", False), ("RoomCentricLeftRearSurround", False),
       ("RoomCentricRightRearSurround", False), ("RoomCentricLeftTopSurround", False),
       ("RoomCentricRightTopSurround", False)]


def _rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def noise(n, seed, rate=SR, colour="white"):
    x = _rng(seed).standard_normal(n)
    if colour == "pink":
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / rate)
        X[1:] /= np.sqrt(f[1:])
        X[0] = 0
        x = np.fft.irfft(X, n)
    return x / np.sqrt(np.mean(x * x))


def lowpass(x, fc, order, rate=SR):
    """Zero-phase Butterworth magnitude |H| applied exactly."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / rate)
    X *= 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    return np.fft.irfft(X, len(x))


def _axml(tracks):
    """chna -> audioTrackFormat -> audioStreamFormat -> audioChannelFormat, as
    the Dolby renderer writes it."""
    chans, uids = [], []
    for k, (name, td, lfe) in enumerate(tracks, start=1):
        ac, uid = f"AC_{0x1000 + k:04x}0001", f"ATU_{k:08x}"
        at, as_ = f"AT_{0x1000 + k:04x}0001_01", f"AS_{0x1000 + k:04x}0001"
        uids.append(f'<audioTrackFormat audioTrackFormatID="{at}"><audioStreamFormatIDRef>{as_}'
                    f'</audioStreamFormatIDRef></audioTrackFormat><audioStreamFormat '
                    f'audioStreamFormatID="{as_}"><audioChannelFormatIDRef>{ac}'
                    f'</audioChannelFormatIDRef></audioStreamFormat>')
        freq = '<frequency typeDefinition="lowPass">120</frequency>' if lfe else ""
        chans.append(f'<audioChannelFormat audioChannelFormatID="{ac}" audioChannelFormatName='
                     f'"{name}" typeDefinition="{td}">{freq}<audioBlockFormat audioBlockFormatID='
                     f'"AB_{0x1000 + k:04x}0001_00000001"><position coordinate="azimuth">0'
                     f'</position></audioBlockFormat></audioChannelFormat>')
        uids.append(f'<audioTrackUID UID="{uid}"><audioChannelFormatIDRef>{ac}'
                    f'</audioChannelFormatIDRef></audioTrackUID>')
    return ('<?xml version="1.0" encoding="UTF-8"?><ebuCoreMain xmlns="urn:ebu:metadata-schema:'
            'ebucore"><coreMetadata><format><audioFormatExtended version="ITU-R_BS.2076-2">'
            + "".join(chans + uids) + "</audioFormatExtended></format></coreMetadata></ebuCoreMain>"
            ).encode()


def _pcm24(x):
    v = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int64).reshape(-1)
    v &= 0xFFFFFF
    b = np.empty((v.size, 3), dtype=np.uint8)
    b[:, 0], b[:, 1], b[:, 2] = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
    return b.tobytes()


def write_wav(path, audio, rate=SR, tracks=None):
    """24-bit WAVE; with `tracks` [(name, typeDefinition, is_lfe)] an ADM master."""
    ch = audio.shape[1]
    fmt = struct.pack("<HHIIHH", 1, ch, rate, rate * ch * 3, ch * 3, 24)
    chunks = [(b"fmt ", fmt)]
    if tracks is not None:
        chna = struct.pack("<HH", ch, ch)
        for k in range(1, ch + 1):
            chna += (struct.pack("<H", k) + f"ATU_{k:08x}".encode()
                     + f"AT_{0x1000 + k:04x}0001_01".encode() + f"AP_{0x1000 + k:04x}0001".encode()
                     + b"\0")
        chunks += [(b"chna", chna), (b"axml", _axml(tracks))]
    chunks.append((b"data", _pcm24(audio)))
    body = b"WAVE"
    for cid, b in chunks:
        body += cid + struct.pack("<I", len(b)) + b + (b"\0" if len(b) & 1 else b"")
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return str(path)


def master(tmp_path, lfe=None, objects=2, n=4 * SR, rate=SR, name="m.wav", seed=1, beds=None):
    """A 7.1.2 bed (LFE on track 4) plus objects, every non-LFE track noise at
    -20 dBFS; `lfe` replaces the LFE track (default: pink low-passed 24 dB/oct
    at 120 Hz)."""
    tracks = [(nm, "DirectSpeakers", is_lfe) for nm, is_lfe in BED] + \
             [(f"Object {i + 1}", "Objects", False) for i in range(objects)]
    cols = [0.1 * noise(n, seed + 10 * k, rate) for k in range(len(tracks))]
    cols[3] = 0.1 * lowpass(noise(n, seed + 3, rate, "pink"), 120, 4, rate) if lfe is None else lfe
    if beds is not None:
        for k, v in beds.items():
            cols[k] = v
    return write_wav(tmp_path / name, np.stack(cols, axis=1), rate, tracks)


def measure(path, stereo=None):
    from sentinel.bwf import adm_info
    res = D.measure_master(path, adm_info(path), stereo=stereo)
    return {r["id"]: r for r in res["rows"]}, res["data"]


# ---------------------------------------------------------------- S-522
def test_lfe_mask_is_the_stated_shape():
    assert D.lfe_mask_db(150.0) == pytest.approx(-3.0103 + D.LFE_TOL_DB, abs=1e-3)
    assert D.lfe_mask_db(300.0) == pytest.approx(10 * math.log10(1 / 17) + D.LFE_TOL_DB, abs=1e-6)
    assert D.lfe_mask_db(20000.0) == D.LFE_FLOOR_DB + D.LFE_TOL_DB


def test_lfe_filtered_at_120_hz_passes(tmp_path):
    r, data = measure(master(tmp_path))
    assert r["S-522"]["verdict"] == "PASS", r["S-522"]["measured"]
    assert data["lfe"]["track"] == 4 and data["lfeTracks"] == [4]
    assert r["S-522"]["plot"].startswith("<svg ")


def test_lfe_unfiltered_pink_warns(tmp_path):
    r, data = measure(master(tmp_path, lfe=0.1 * noise(4 * SR, 7, colour="pink")))
    assert r["S-522"]["verdict"] == "WARN"
    assert data["lfe"]["worstExcessDb"] > 3.0


def test_lfe_gentlest_shape_at_150_hz_passes_where_a_200_hz_energy_rule_fails(tmp_path):
    """12 dB/oct at 150 Hz on flat content: within the specification's range,
    and the reference shape itself. It leaves about -8.9 dB of its energy above
    200 Hz (integrating its magnitude response over a flat spectrum), which a
    -18 dB energy rule fails."""
    r, data = measure(master(tmp_path, lfe=0.1 * lowpass(noise(4 * SR, 8), 150, 2)))
    assert r["S-522"]["verdict"] == "PASS", r["S-522"]["measured"]
    assert -10.0 < data["lfe"]["energyAbove200HzDb"] < -8.0


def test_lfe_cut_off_outside_the_range_warns(tmp_path):
    r, _ = measure(master(tmp_path, lfe=0.1 * lowpass(noise(4 * SR, 9), 400, 4)))
    assert r["S-522"]["verdict"] == "WARN"


def test_lfe_silent_or_absent(tmp_path):
    r, _ = measure(master(tmp_path, lfe=np.zeros(4 * SR)))
    assert r["S-522"]["verdict"] == "PASS" and "silent" in r["S-522"]["measured"]
    assert r["S-524"]["verdict"] == "WARN" and "RoomCentricLFE" in r["S-524"]["measured"]
    p = write_wav(tmp_path / "nolfe.wav", 0.1 * np.stack([noise(SR, 1), noise(SR, 2)], 1),
                  tracks=[("A", "Objects", False), ("B", "Objects", False)])
    r, _ = measure(p)
    assert r["S-522"]["verdict"] == "PASS" and r["S-522"]["measured"] == "no LFE channel"


def test_lfe_with_no_bass_warns_and_says_where_it_sits(tmp_path):
    t = np.arange(4 * SR) / SR
    r, data = measure(master(tmp_path, lfe=0.1 * np.sin(2 * np.pi * 1000 * t)))
    assert r["S-522"]["verdict"] == "WARN" and "peaks near" in r["S-522"]["measured"]
    assert data["lfe"]["strongestHz"] == pytest.approx(1000, abs=3)


def test_common_definition_lfe_is_found_from_chna(tmp_path):
    """A bed that leaves its formats to the common definitions: the axml does
    not define AC_00010004, chna names AT_00010004_01."""
    p = write_wav(tmp_path / "cd.wav", 0.1 * np.stack([noise(SR, 1), noise(SR, 2)], 1),
                  tracks=[("A", "Objects", False), ("B", "Objects", False)])
    data = bytearray(open(p, "rb").read())
    i = data.find(b"AT_10020001_01")
    data[i:i + 14] = b"AT_00010004_01"
    open(p, "wb").write(bytes(data))
    r, d = measure(p)
    assert d["lfeTracks"] == [2] and r["S-522"]["verdict"] == "WARN"   # white noise in the LFE


def test_lfe_too_quiet_to_judge(tmp_path):
    r, _ = measure(master(tmp_path, lfe=1e-5 * lowpass(noise(4 * SR, 9), 400, 4)))
    assert r["S-522"]["verdict"] == "UNDETERMINED"


# ---------------------------------------------------------------- S-524
def test_silent_and_near_silent_channels_are_listed(tmp_path):
    n = 4 * SR
    quiet = 10 ** (-90 / 20) * np.sign(noise(n, 4))           # peak -90 dBFS
    r, data = measure(master(tmp_path, beds={10: np.zeros(n), 11: quiet}))
    row = r["S-524"]
    assert row["verdict"] == "WARN"
    assert row["silent"] == [11] and row["nearSilent"] == [12]
    assert "Object 1" in row["measured"] and "Object 2" in row["measured"]
    r, _ = measure(master(tmp_path, name="ok.wav"))
    assert r["S-524"]["verdict"] == "PASS"


def test_near_silent_threshold_is_the_named_one(tmp_path):
    n = SR
    just = 10 ** ((D.NEAR_SILENT_DBFS + 1) / 20) * np.sign(noise(n, 5))
    r, _ = measure(master(tmp_path, n=n, beds={10: just}))
    assert r["S-524"]["verdict"] == "PASS"


# ---------------------------------------------------------------- S-525
def _upsample2(x):
    """48 kHz -> 96 kHz by ideal (FFT zero-pad) interpolation."""
    n = len(x)
    X = np.fft.rfft(x)
    Y = np.zeros(n + 1, dtype=complex)
    Y[:len(X)] = X
    return np.fft.irfft(Y, 2 * n) * 2


def test_96k_native_passes_upsampled_warns_tone_undetermined(tmp_path):
    n96 = 2 * SR * 2
    tracks = [("A", "Objects", False), ("B", "Objects", False)]
    native = 0.1 * np.stack([noise(n96, 1, 96000), noise(n96, 2, 96000)], 1)
    r, data = measure(write_wav(tmp_path / "n.wav", native, 96000, tracks))
    assert r["S-525"]["verdict"] == "PASS", r["S-525"]["measured"]
    up = 0.1 * np.stack([_upsample2(noise(n96 // 2, 1)), _upsample2(noise(n96 // 2, 2))], 1)
    r, data = measure(write_wav(tmp_path / "u.wav", up, 96000, tracks))
    assert r["S-525"]["verdict"] == "WARN", r["S-525"]["measured"]
    t = np.arange(n96) / 96000
    tone = 0.1 * np.stack([np.sin(2 * np.pi * 1000 * t)] * 2, 1)
    r, _ = measure(write_wav(tmp_path / "t.wav", tone, 96000, tracks))
    assert r["S-525"]["verdict"] == "UNDETERMINED"
    r, _ = measure(master(tmp_path, n=SR))
    assert r["S-525"]["verdict"] == "PASS" and "not applicable" in r["S-525"]["measured"]


# ---------------------------------------------------------------- S-523
def _pair(tmp_path, *, lead=0, tail=0, late=0, unrelated=False, n=3 * SR, lfe=None):
    """A master and its stereo master: the stereo is a fold-down of the master's
    non-LFE tracks plus a little independent noise. `lead` puts extra silence
    before the master, `tail` extra after it (negative: cut short), `late`
    starts the master that many samples into the stereo's content."""
    path = master(tmp_path, n=n, name="song.wav", lfe=lfe)
    from sentinel.bwf import adm_info
    x = np.concatenate(list(D.blocks(path, adm_info(path))))
    keep = [i for i in range(x.shape[1]) if i != 3]
    mono = x[:, keep].sum(axis=1).astype(np.float64)
    if unrelated:
        mono = noise(n, 99) * float(np.std(mono))
    st = np.stack([0.5 * mono + 0.01 * noise(n, 50), 0.5 * mono + 0.01 * noise(n, 51)], 1)
    st = 0.5 * st / np.max(np.abs(st))
    m = x
    if late:
        m = m[late:]
    if lead:
        m = np.concatenate([np.zeros((lead, m.shape[1])), m])
    if tail > 0:
        m = np.concatenate([m, np.zeros((tail, m.shape[1]))])
    elif tail < 0:
        m = m[:tail]
    tracks = [(nm, "DirectSpeakers", lf) for nm, lf in BED] + \
             [(f"Object {i + 1}", "Objects", False) for i in range(x.shape[1] - 10)]
    write_wav(tmp_path / "song.wav", m, SR, tracks)
    return str(tmp_path / "song.wav"), write_wav(tmp_path / "song_stereo.wav", st)


def test_sync_aligned_passes(tmp_path):
    p, s = _pair(tmp_path)
    r, data = measure(p, s)
    assert r["S-523"]["verdict"] == "PASS", r["S-523"]["measured"]
    assert data["sync"]["offsetSamples"] == 0 and data["sync"]["extraTailSamples"] == 0
    assert data["sync"]["correlation"] > 0.9


@pytest.mark.parametrize("kw,verdict,offset,tail", [
    ({"lead": 480}, "FAIL", 480, 0),            # extra lead silence
    ({"lead": 48}, "PASS", 48, 0),              # within the 1 ms tolerance
    ({"lead": 49}, "FAIL", 49, 0),
    ({"lead": 49, "tail": -49}, "FAIL", 49, -49),
    ({"tail": 1999}, "PASS", 0, 1999),          # under one frame
    ({"tail": 2000}, "FAIL", 0, 2000),          # one frame
    ({"late": 300}, "FAIL", -300, 0),           # starts after the stereo
    ({"tail": -3000}, "WARN", 0, -3000),        # ends before the stereo
])
def test_sync_offsets_and_tails(tmp_path, kw, verdict, offset, tail):
    p, s = _pair(tmp_path, **kw)
    r, data = measure(p, s)
    assert (data["sync"]["offsetSamples"], data["sync"]["extraTailSamples"]) == (offset, tail)
    assert r["S-523"]["verdict"] == verdict, r["S-523"]["measured"]


def test_sync_folds_down_without_the_lfe(tmp_path):
    """The stereo master carries no LFE; a loud LFE unrelated to it must not
    weaken the alignment (with it in the fold-down the correlation is ~0.34)."""
    p, s = _pair(tmp_path, lfe=0.9 * lowpass(noise(3 * SR, 77), 100, 4) * 8)
    r, data = measure(p, s)
    assert r["S-523"]["verdict"] == "PASS" and data["sync"]["correlation"] > 0.9


def test_sync_unrelated_stereo_is_undetermined(tmp_path):
    p, s = _pair(tmp_path, unrelated=True)
    r, _ = measure(p, s)
    assert r["S-523"]["verdict"] == "UNDETERMINED" and "correlate" in r["S-523"]["notes"][-1]


def test_sync_rate_mismatch_is_undetermined(tmp_path):
    p = master(tmp_path, n=SR)
    s = write_wav(tmp_path / "s44.wav", 0.1 * np.stack([noise(44100, 1)] * 2, 1), 44100)
    r, _ = measure(p, s)
    assert r["S-523"]["verdict"] == "UNDETERMINED" and "44100" in r["S-523"]["measured"]


# ---------------------------------------------------------------- S-533
def _s(off, tail):
    return {"offsetSamples": off, "extraTailSamples": tail, "frameSamples": 2000}


def test_gapless_boundaries_at_half_a_frame():
    names = ["01.wav", "02.wav", "03.wav"]
    ok = D.gapless_boundaries([_s(0, 1000), _s(-1000, 0), _s(0, 0)], names)
    assert ok["verdict"] == "PASS"
    bad = D.gapless_boundaries([_s(0, 0), _s(0, 1001), _s(0, 0)], names)
    assert bad["verdict"] == "FAIL" and "02.wav / 03.wav" in bad["measured"]
    assert D.gapless_boundaries([_s(0, 0), None], names[:2])["verdict"] == "UNDETERMINED"


# ------------------------------------------------- through the core's adm-check
def _core_has_seam():
    try:
        from sentinel import adm_check
    except ImportError:
        return False
    return hasattr(adm_check, "_measure_plugin")


@pytest.mark.skipif(not _core_has_seam(), reason="installed core has no adm-check measurement seam")
def test_adm_check_fills_the_rows_with_this_plugin(tmp_path):
    from sentinel import adm_check
    p, s = _pair(tmp_path, lead=480)
    res = adm_check.run([str(tmp_path)])
    rows = {r["id"]: r for r in res["masters"][0]["rows"]}
    assert res["stereoPairs"] == {p: s}
    assert rows["S-523"]["verdict"] == "FAIL" and rows["S-522"]["verdict"] == "PASS"
    assert rows["S-524"]["verdict"] == "PASS" and rows["S-525"]["verdict"] == "PASS"
    assert "<svg " in adm_check.render_html(res)
