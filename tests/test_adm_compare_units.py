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
    fake.write_text("#!/bin/sh\nexit 0\n")
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
    fake_ear.write_text("#!/bin/sh\nexit 0\n")
    fake_ear.chmod(0o755)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))          # ear yes, oracles no
    res = compare(str(tmp_path / "s.wav"), str(tmp_path / "o.iamf"),
                  ear_bin=str(fake_ear))
    assert res.exit_code() == 2
    assert "toolchain not found" in res.execution_error


def test_compare_unsupported_layout_errors(tmp_path):
    fake_ear = tmp_path / "ear-render"
    fake_ear.write_text("#!/bin/sh\nexit 0\n")
    fake_ear.chmod(0o755)
    dm = tmp_path / "root/src/build-iamf/decoder_main"
    dm.parent.mkdir(parents=True)
    dm.write_text("#!/bin/sh\nexit 0\n")
    dm.chmod(0o755)
    tc = Toolchain.discover(str(tmp_path / "root"))
    res = compare(str(tmp_path / "s.wav"), str(tmp_path / "o.iamf"),
                  layout="22.2", ear_bin=str(fake_ear), toolchain=tc)
    assert res.exit_code() == 2
    assert "unsupported layout" in res.execution_error
