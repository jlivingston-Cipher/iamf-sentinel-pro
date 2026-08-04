"""adm-compare I/O units (toolchain-free, no EAR): WAV reading across sample
formats, ear-render discovery, the result/renderer contracts, and compare()'s
three early error returns. The executed pipeline (EAR + decoder) stays with
the acceptance study.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentinel_pro.adm_compare import (ADMCompareResult, _read_wav,   # noqa: E402
                                      compare, find_ear, render_json,
                                      render_text)
from sentinel_pro.oracle import Toolchain                            # noqa: E402


def _write_wav(path, pcm, sr=48000, bits=16, magic=b"WAVE"):
    frames, ch = pcm.shape
    if bits == 16:
        ints = np.clip(np.round(pcm * (1 << 15)), -(1 << 15), (1 << 15) - 1)
        raw = ints.astype("<i2").reshape(-1).tobytes()
    elif bits == 24:
        ints = np.clip(np.round(pcm * (1 << 23)), -(1 << 23), (1 << 23) - 1)
        flat = ints.astype(np.int64).reshape(-1) & 0xFFFFFF
        b = np.empty((flat.size, 3), dtype=np.uint8)
        b[:, 0], b[:, 1], b[:, 2] = flat & 0xFF, (flat >> 8) & 0xFF, (flat >> 16) & 0xFF
        raw = b.tobytes()
    elif bits == 32:
        ints = np.clip(np.round(pcm * (1 << 31)), -(1 << 31), (1 << 31) - 1)
        raw = ints.astype("<i4").reshape(-1).tobytes()
    fmt = struct.pack("<HHIIHH", 1, ch, sr, sr * ch * (bits // 8),
                      ch * (bits // 8), bits)
    body = (magic + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(raw)) + raw)
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)


# ------------------------------------------------------------------ _read_wav

@pytest.mark.parametrize("bits", [16, 24, 32])
def test_read_wav_integer_formats_round_trip(tmp_path, bits):
    t = np.arange(4800) / 48000.0
    pcm = np.stack([0.5 * np.sin(2 * np.pi * 997 * t),
                    0.25 * np.sin(2 * np.pi * 1499 * t)], axis=1)
    p = tmp_path / f"x{bits}.wav"
    _write_wav(str(p), pcm, bits=bits)
    got, rate = _read_wav(str(p))
    assert rate == 48000 and got.shape == pcm.shape
    tol = {16: 1e-4, 24: 1e-6, 32: 1e-6}[bits]   # reader returns float32
    assert float(np.abs(got - pcm).max()) < tol


def test_read_wav_rejects_non_riff(tmp_path):
    p = tmp_path / "not.wav"
    p.write_bytes(b"garbage bytes, definitely not RIFF")
    with pytest.raises(ValueError):
        _read_wav(str(p))


# ------------------------------------------------------------------- find_ear

def test_find_ear_env_wins(tmp_path, monkeypatch):
    fake = tmp_path / "ear-render"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("SENTINEL_EAR", str(fake))
    assert find_ear() == str(fake)


def test_find_ear_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_EAR", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))          # empty PATH
    got = find_ear()
    # None unless this host carries the documented /opt/earvenv install
    assert got is None or got == "/opt/earvenv/bin/ear-render"


# ------------------------------------------------- result + renderer contracts

def test_exit_code_error_beats_findings():
    r = ADMCompareResult(source="s", output="o", layout="5.1")
    assert r.exit_code() == 0
    r.execution_error = "boom"
    assert r.exit_code() == 2


def test_render_json_and_text_error_shape():
    r = ADMCompareResult(source="s.wav", output="o.iamf", layout="5.1")
    r.execution_error = "ear-render not found (set $SENTINEL_EAR)"
    doc = json.loads(render_json(r))
    assert doc["result"] == "ERROR" and doc["exit_code"] == 2
    assert doc["execution_error"].startswith("ear-render not found")
    txt = render_text(r)
    assert txt.startswith("sentinel adm-compare — ERROR")
    assert "  error  : ear-render not found" in txt


def test_render_text_clean_shape():
    r = ADMCompareResult(source="s.wav", output="o.iamf", layout="5.1")
    txt = render_text(r)
    assert "no findings — output is consistent" in txt
    assert "metrics:" in txt


# ------------------------------------------------- compare() early error arms

def test_compare_without_ear_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_EAR", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    if os.path.exists("/opt/earvenv/bin/ear-render"):
        pytest.skip("host carries the documented EAR install")
    res = compare(str(tmp_path / "s.wav"), str(tmp_path / "o.iamf"))
    assert res.exit_code() == 2
    assert "ear-render not found" in res.execution_error


def test_compare_without_toolchain_errors(tmp_path, monkeypatch):
    fake_ear = tmp_path / "ear-render"
    fake_ear.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ear.chmod(0o755)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))          # ear yes, oracles no
    res = compare(str(tmp_path / "s.wav"), str(tmp_path / "o.iamf"),
                  ear_bin=str(fake_ear))
    assert res.exit_code() == 2
    assert "toolchain not found" in res.execution_error


def test_compare_unsupported_layout_errors(tmp_path):
    fake_ear = tmp_path / "ear-render"
    fake_ear.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ear.chmod(0o755)
    dm = tmp_path / "root/src/build-iamf/decoder_main"
    dm.parent.mkdir(parents=True)
    dm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dm.chmod(0o755)
    tc = Toolchain.discover(str(tmp_path / "root"))
    res = compare(str(tmp_path / "s.wav"), str(tmp_path / "o.iamf"),
                  layout="22.2", ear_bin=str(fake_ear), toolchain=tc)
    assert res.exit_code() == 2
    assert "unsupported layout" in res.execution_error


# ------------------------------------------------------- BW64 (BS.2088) walk

from sentinel_pro.adm_compare import (_find_chunks, below_threshold_tracks,  # noqa: E402
                                      silence_tracks)


def _write_bw64_adm(path, frames=480, ch=2, importance=3):
    """Minimal BW64/RF64 ADM: ds64 + fmt(16-bit) + chna + axml + sentinel-size
    data. The data chunk's 32-bit size field carries the 0xFFFFFFFF sentinel and
    the real size lives only in ds64.dataSize — the shape every >4 GiB master
    uses (exercised here without a multi-gigabyte payload)."""
    fmt = struct.pack("<HHIIHH", 1, ch, 48000, 48000 * ch * 2, ch * 2, 16)
    # per-track distinct byte patterns so zeroing is attributable
    raw = bytearray()
    for _ in range(frames):
        for c in range(ch):
            raw += struct.pack("<h", 0x1111 * (c + 1))
    chna = struct.pack("<HH", ch, ch)
    for i in range(ch):
        chna += struct.pack("<H", i + 1)
        chna += (f"ATU_{i + 1:08d}".encode().ljust(12, b"\x00"))
        chna += (f"AT_0003{0x1001 + i:04x}_01".encode().ljust(14, b"\x00"))
        chna += (f"AP_0003{0x1001 + i:04x}".encode().ljust(11, b"\x00") + b"\x00")
    axml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebuCore_2016">'
        '<coreMetadata><format>'
        '<audioFormatExtended version="ITU-R_BS.2076-2">'
        '<audioProgramme audioProgrammeID="APR_1001"></audioProgramme>'
        f'<audioObject audioObjectID="AO_1001" audioObjectName="Obj1" '
        f'importance="{importance}">'
        '<audioTrackUIDRef>ATU_00000001</audioTrackUIDRef></audioObject>'
        '<audioObject audioObjectID="AO_1002" audioObjectName="Obj2">'
        '<audioTrackUIDRef>ATU_00000002</audioTrackUIDRef></audioObject>'
        '</audioFormatExtended></format></coreMetadata></ebuCoreMain>'
    ).encode()

    def chunk(cid, payload, sentinel=False):
        size = 0xFFFFFFFF if sentinel else len(payload)
        out = cid + struct.pack("<I", size) + payload
        if len(payload) & 1:
            out += b"\x00"
        return out

    ds64 = struct.pack("<QQQI", 0, len(raw), frames, 0)
    body = (b"WAVE" + chunk(b"ds64", ds64) + chunk(b"fmt ", fmt)
            + chunk(b"chna", chna) + chunk(b"axml", axml)
            + chunk(b"data", bytes(raw), sentinel=True))
    with open(path, "wb") as fh:
        fh.write(b"BW64" + struct.pack("<I", 0xFFFFFFFF) + body)
    return len(raw)


def test_bw64_find_chunks_resolves_data_sentinel(tmp_path):
    p = str(tmp_path / "m.wav")
    data_len = _write_bw64_adm(p)
    with open(p, "rb") as fh:
        chunks = _find_chunks(fh.read())
    assert chunks[b"data"][1] == data_len          # sentinel resolved via ds64
    assert b"axml" in chunks and b"chna" in chunks


def test_bw64_below_threshold_tracks(tmp_path):
    p = str(tmp_path / "m.wav")
    _write_bw64_adm(p, importance=3)
    tracks, names = below_threshold_tracks(p, 10)
    assert tracks == [1] and names == ["Obj1"]     # Obj2 has no importance attr


def test_bw64_silence_tracks_round_trip(tmp_path):
    src, dst = str(tmp_path / "m.wav"), str(tmp_path / "out.wav")
    _write_bw64_adm(src)
    silence_tracks(src, [1], dst)                  # raised ValueError pre-fix
    a, b = open(src, "rb").read(), open(dst, "rb").read()
    assert len(a) == len(b)
    off, size = _find_chunks(a)[b"data"]
    assert a[:off] == b[:off] and a[off + size:] == b[off + size:]
    frames = np.frombuffer(b, dtype="<i2", count=size // 2, offset=off).reshape(-1, 2)
    assert not frames[:, 0].any()                  # track 1 zeroed
    assert (frames[:, 1] == 0x2222).all()          # track 2 untouched
