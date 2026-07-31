"""Toolchain-free oracle units: discovery precedence, oracle listing,
input preparation (raw passthrough + MP4 reconstruction), and decode()'s
judgment notes with stub decoders. Real-decoder rendering stays with the
toolchain suites.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentinel_pro.oracle import (SOUND_SYSTEM_TO_LAYOUT, DecodeResult,  # noqa: E402
                                 Toolchain)

from conftest import (NO_OSS_SRC_REASON, OSS_SRC,      # noqa: E402
                      needs_windows_pathext)

if OSS_SRC is None:                                   # pragma: no cover
    pytest.skip(NO_OSS_SRC_REASON, allow_module_level=True)
from fixtures.build import build, channel_spec        # noqa: E402
from fixtures.mp4wrap import wrap_mp4                 # noqa: E402


def _fake_root(tmp_path, name="root"):
    root = tmp_path / name
    dm = root / "src/build-iamf/decoder_main"
    dm.parent.mkdir(parents=True)
    dm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dm.chmod(0o755)
    idc = root / "src/build-iamfdec/iamfdec"
    idc.parent.mkdir(parents=True)
    idc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    idc.chmod(0o755)
    return root


# ------------------------------------------------------------------ discovery

def test_discover_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))    # nothing on PATH
    tc = Toolchain.discover()
    assert not tc.available
    assert tc.oracles() == []


def test_discover_explicit_root_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    root = _fake_root(tmp_path)
    tc = Toolchain.discover(str(root))
    assert tc.available
    # Compare as paths, not strings: Toolchain builds its members by string
    # concatenation with "/" (which keeps plan/evidence output platform-
    # invariant, doc 97 3.2), so on Windows the product returns mixed
    # separators and str(WindowsPath) does not. Both name the same file.
    assert Path(tc.decoder_main) == root / "src/build-iamf/decoder_main"
    assert Path(tc.iamfdec) == root / "src/build-iamfdec/iamfdec"
    assert set(tc.oracles()) == {"decoder_main", "iamfdec"}


def test_discover_env_root(tmp_path, monkeypatch):
    root = _fake_root(tmp_path)
    monkeypatch.setenv("SENTINEL_TOOLCHAIN", str(root))
    tc = Toolchain.discover()
    assert Path(tc.decoder_main) == root / "src/build-iamf/decoder_main"


@needs_windows_pathext
def test_discover_path_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    dm = bindir / "decoder_main"
    dm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dm.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    tc = Toolchain.discover()
    assert Path(tc.decoder_main) == dm
    assert tc.iamfdec is None


# -------------------------------------------------------------- prepare_input

def test_prepare_input_raw_passthrough(tmp_path):
    p = tmp_path / "x.iamf"
    p.write_bytes(build(channel_spec("stereo")))
    tc = Toolchain()
    try:
        assert tc.prepare_input(str(p)) == str(p)
    finally:
        tc.cleanup()


def test_prepare_input_reconstructs_mp4(tmp_path):
    raw = build(channel_spec("stereo"))
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes(wrap_mp4(raw))
    tc = Toolchain()
    try:
        out = tc.prepare_input(str(mp4))
        assert out != str(mp4) and out.endswith(".iamf")
        assert os.path.getsize(out) > 0
    finally:
        tc.cleanup()


def test_prepare_input_missing_file_passes_through(tmp_path):
    tc = Toolchain()
    try:
        assert tc.prepare_input(str(tmp_path / "nope.mp4")) == str(tmp_path / "nope.mp4")
    finally:
        tc.cleanup()


# --------------------------------------------------------------------- decode

def test_decode_notes_when_decoder_absent(tmp_path):
    tc = Toolchain()
    try:
        r = tc.decode(str(tmp_path / "x.iamf"), "5.1", "decoder_main")
        assert r == DecodeResult(False, "decoder_main", "5.1",
                                 note="decoder_main not available")
        r = tc.decode(str(tmp_path / "x.iamf"), "5.1", "iamfdec")
        assert r.ok is False and "not available" in r.note
    finally:
        tc.cleanup()


def test_decode_unknown_layout_mapping(tmp_path):
    dm = tmp_path / "iamfdec"
    dm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dm.chmod(0o755)
    tc = Toolchain(iamfdec=str(dm))
    try:
        r = tc.decode(str(tmp_path / "x.iamf"), "22.2", "iamfdec")
        assert r.ok is False and "no sound-system mapping" in r.note
    finally:
        tc.cleanup()


def test_decode_exit_zero_but_no_output_is_not_ok(tmp_path):
    # the F8 judgment: judge by the rendered file, never by rc
    dm = tmp_path / "decoder_main"
    dm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")                  # writes nothing
    dm.chmod(0o755)
    tc = Toolchain(decoder_main=str(dm))
    try:
        r = tc.decode(str(tmp_path / "x.iamf"), "5.1", "decoder_main")
        assert r.ok is False
    finally:
        tc.cleanup()


def test_sound_system_table_is_bijective_enough():
    # every mapped layout name resolves back to exactly one sound system
    for ss, name in SOUND_SYSTEM_TO_LAYOUT.items():
        back = [k for k, v in SOUND_SYSTEM_TO_LAYOUT.items() if v == name]
        assert back == [ss]
