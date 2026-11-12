"""ADR-6b backend switch tests (doc 35): discovery, contracts, mini differential.

Toolchain-free by design — this is the CI half that runs on every change to
either implementation (doc 33 "CI: the synthetic half"). The kernel-requiring
tests locate the binary via $SENTINEL_DSP (a real path), else `sentinel-dsp`
on $PATH; they skip only when no binary exists anywhere (the combined-suite
configuration always provides one, keeping the 0-skipped bar).
"""
import json
import os
import shutil
import stat
import struct
import subprocess
import sys

import pytest

from conftest import write_exec_stub          # noqa: E402

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentinel_pro import dsp  # noqa: E402

SR = 48000

# Differential gates — doc 32 pre-registration, unchanged.
GATE_LU = 0.001
GATE_DB = 0.01
GATE_EPS = 1e-6


def _find_kernel():
    env = os.environ.get("SENTINEL_DSP", "")
    if env and env.strip().lower() not in ("0", "off") and os.path.isfile(env):
        return env
    tc = os.environ.get("SENTINEL_TOOLCHAIN")
    if tc:
        found = shutil.which("sentinel-dsp", path=os.path.join(tc, "bin"))
        if found:
            return found
    return shutil.which("sentinel-dsp")


def write_wav(path, pcm, sr=SR, bits=24):
    frames, ch = pcm.shape
    if bits == 24:
        ints = np.clip(np.round(pcm * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
        u = ints.reshape(-1) & 0xFFFFFF
        b = np.empty((u.size, 3), dtype=np.uint8)
        b[:, 0] = u & 0xFF
        b[:, 1] = (u >> 8) & 0xFF
        b[:, 2] = (u >> 16) & 0xFF
        raw, fmt = b.tobytes(), struct.pack("<HHIIHH", 1, ch, sr, sr * ch * 3, ch * 3, 24)
    elif bits == 16:
        raw = np.clip(np.round(pcm * (1 << 15)), -(1 << 15),
                      (1 << 15) - 1).astype("<i2").reshape(-1).tobytes()
        fmt = struct.pack("<HHIIHH", 1, ch, sr, sr * ch * 2, ch * 2, 16)
    else:
        raise ValueError(bits)
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw)) + b"WAVE")
        fh.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        fh.write(b"data" + struct.pack("<I", len(raw)) + raw)


def sine(freq, secs=4.0, level_db=-18.0, phase=0.0):
    t = np.arange(int(secs * SR)) / SR
    return (10 ** (level_db / 20.0)) * np.sin(2 * np.pi * freq * t + phase)


@pytest.fixture
def stereo_wav(tmp_path):
    p = str(tmp_path / "stereo.wav")
    write_wav(p, np.stack([sine(997), sine(1499)], axis=1))
    return p


# ---------------------------------------------------------------- discovery

def test_discovery_env_explicit_wins(monkeypatch, tmp_path):
    stub = tmp_path / "stub-dsp"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("SENTINEL_DSP", str(stub))
    assert dsp.kernel_path() == str(stub)


@pytest.mark.parametrize("val", ["off", "0", "", "OFF"])
def test_discovery_off_forces_python(monkeypatch, val):
    monkeypatch.setenv("SENTINEL_DSP", val)
    assert dsp.kernel_path() is None


def test_discovery_bad_explicit_path_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTINEL_DSP", str(tmp_path / "no-such-binary"))
    with pytest.raises(dsp.KernelError):
        dsp.kernel_path()


def test_discovery_path_fallback(monkeypatch, tmp_path):
    # %PATHEXT%-aware on Windows: the stub is `sentinel-dsp.cmd` there and
    # `which()` must still resolve the bare name (item 25).
    stub = write_exec_stub(tmp_path, "sentinel-dsp")
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    found = dsp.kernel_path()
    assert found is not None and os.path.samefile(found, stub)


def test_exec_stub_selfcheck(tmp_path):
    """The stub helper's own contract, proven on every leg (item 25).

    This is the meta-test that keeps the Windows `.cmd` mechanics honest:
    exit code, stdout and stderr all round-trip through a real subprocess.
    """
    stub = write_exec_stub(tmp_path, "stubcheck", stdout="out here",
                           stderr="err here", exit_code=5)
    r = subprocess.run([str(stub)], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert r.returncode == 5
    assert "out here" in r.stdout
    assert "err here" in r.stderr


def test_discovery_absent_means_python(monkeypatch, tmp_path):
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))          # empty dir
    assert dsp.kernel_path() is None


def _root_with_kernel(base, name="root"):
    """A toolchain root holding bin/sentinel-dsp (a stub)."""
    binr = base / name / "bin"
    binr.mkdir(parents=True)
    return base / name, write_exec_stub(binr, "sentinel-dsp")


def test_discovery_toolchain_root_before_path(monkeypatch, tmp_path):
    """IA-0074: the kernel sits in the toolchain root (bin/sentinel-dsp), where
    the recipes install it; a root the caller resolved wins over $PATH, as in
    iamf-loom's own lookup. Before, only $SENTINEL_DSP and $PATH were read and
    the gate measured with the numpy reference beside a built kernel."""
    root, in_root = _root_with_kernel(tmp_path)
    on_path_dir = tmp_path / "pathdir"
    on_path_dir.mkdir()
    write_exec_stub(on_path_dir, "sentinel-dsp")
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(on_path_dir))
    found = dsp.kernel_path([str(root)])
    assert found is not None and os.path.samefile(found, in_root)
    # without the root, PATH is still the fallback
    found = dsp.kernel_path()
    assert found is not None and not os.path.samefile(found, in_root)


def test_discovery_env_toolchain_root(monkeypatch, tmp_path):
    root, in_root = _root_with_kernel(tmp_path)
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.setenv("SENTINEL_TOOLCHAIN", str(root))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    found = dsp.kernel_path()
    assert found is not None and os.path.samefile(found, in_root)


def test_discovery_root_without_kernel_falls_through(monkeypatch, tmp_path):
    (tmp_path / "bare" / "bin").mkdir(parents=True)
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert dsp.kernel_path([str(tmp_path / "bare")]) is None


@pytest.mark.parametrize("val", ["off", None])
def test_discovery_env_dsp_beats_root(monkeypatch, tmp_path, val):
    """$SENTINEL_DSP still decides first: 'off' forces Python even with a
    kernel in the root, and an explicit path wins over the root's."""
    root, _ = _root_with_kernel(tmp_path)
    if val is None:
        other = tmp_path / "explicit"
        other.mkdir()
        explicit = write_exec_stub(other, "sentinel-dsp")
        if not os.access(explicit, os.X_OK):
            pytest.skip("explicit-path form needs an executable stub")
        monkeypatch.setenv("SENTINEL_DSP", str(explicit))
        assert os.path.samefile(dsp.kernel_path([str(root)]), explicit)
    else:
        monkeypatch.setenv("SENTINEL_DSP", val)
        assert dsp.kernel_path([str(root)]) is None


def test_toolchain_records_roots_and_analyze_uses_them(monkeypatch, tmp_path,
                                                       stereo_wav):
    """The wiring: Toolchain.discover keeps its roots, and analyze(roots=...)
    reaches the root's kernel (a stub that fails, so the error names it)."""
    from sentinel_pro.oracle import Toolchain
    root = tmp_path / "tc"
    (root / "bin").mkdir(parents=True)
    write_exec_stub(root / "bin", "sentinel-dsp", stderr="rootkernel",
                    exit_code=4)
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    tc = Toolchain.discover(str(root))
    assert tc.roots == [str(root)]
    with pytest.raises(dsp.KernelError, match="exited 4.*rootkernel"):
        dsp.analyze(stereo_wav, layout="2.0", roots=tc.roots)
    # and without the roots the same call takes the reference path
    assert dsp.analyze(stereo_wav, layout="2.0")["backend"] == "python"


# ---------------------------------------------------------------- contracts

def test_analyze_python_matches_reference(monkeypatch, stereo_wav):
    monkeypatch.setenv("SENTINEL_DSP", "off")
    a = dsp.analyze(stereo_wav, layout="2.0")
    assert a["backend"] == "python"
    sr, pcm = dsp.read_wav(stereo_wav)
    assert a["sample_rate"] == sr and a["channels"] == 2
    assert a["integrated_lufs"] == dsp.integrated_loudness(
        pcm, sr, dsp.CHANNEL_WEIGHTS["2.0"])
    assert a["digital_peak_dbfs"] == dsp.digital_peak_dbfs(pcm)
    assert a["true_peak_dbtp"] == dsp.true_peak_dbtp(pcm)
    assert a["correlation"]["silent_channels"] == dsp.silent_channels(pcm)


def test_forced_kernel_without_binary_raises(monkeypatch, stereo_wav, tmp_path):
    monkeypatch.delenv("SENTINEL_DSP", raising=False)
    monkeypatch.delenv("SENTINEL_TOOLCHAIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(dsp.KernelError):
        dsp.analyze(stereo_wav, layout="2.0", backend="kernel")


def test_broken_kernel_is_error_not_fallback(monkeypatch, stereo_wav, tmp_path):
    """D-S3: present-but-failing kernel never silently downgrades to numpy."""
    stub = write_exec_stub(tmp_path, "sentinel-dsp",
                           stderr="kaboom", exit_code=3)
    monkeypatch.setenv("SENTINEL_DSP", str(stub))
    with pytest.raises(dsp.KernelError, match="exited 3.*kaboom"):
        dsp.analyze(stereo_wav, layout="2.0")


def test_kernel_invalid_json_is_error(monkeypatch, stereo_wav, tmp_path):
    stub = write_exec_stub(tmp_path, "sentinel-dsp", stdout="not json")
    monkeypatch.setenv("SENTINEL_DSP", str(stub))
    with pytest.raises(dsp.KernelError, match="invalid JSON"):
        dsp.analyze(stereo_wav, layout="2.0")


def test_kernel_missing_fields_is_error(monkeypatch, stereo_wav, tmp_path):
    stub = write_exec_stub(tmp_path, "sentinel-dsp",
                           stdout='{"sample_rate": 48000}')
    monkeypatch.setenv("SENTINEL_DSP", str(stub))
    with pytest.raises(dsp.KernelError, match="missing fields"):
        dsp.analyze(stereo_wav, layout="2.0")


# ------------------------------------------------- mini differential (E-K2)

def _vectors(tmp_path):
    out = []
    p = str(tmp_path / "d_stereo.wav")
    write_wav(p, np.stack([sine(997), sine(1499)], axis=1))
    out.append((p, "2.0"))
    base = [sine(313 + 139 * k) for k in range(6)]
    base[2] = np.zeros_like(base[2])       # C silent
    base[5] = base[4].copy()               # Rs := Ls
    p = str(tmp_path / "d_dupsilent.wav")
    write_wav(p, np.stack(base, axis=1))
    out.append((p, "5.1"))
    p = str(tmp_path / "d_short.wav")
    write_wav(p, np.stack([sine(997, 0.25), sine(1499, 0.25)], axis=1), bits=16)
    out.append((p, "2.0"))
    return out


def test_kernel_matches_python_within_gates(monkeypatch, tmp_path):
    binary = _find_kernel()
    if not binary:
        pytest.skip("no sentinel-dsp binary available ($SENTINEL_DSP / $PATH)")
    for path, layout in _vectors(tmp_path):
        monkeypatch.setenv("SENTINEL_DSP", binary)
        k = dsp.analyze(path, layout=layout)
        monkeypatch.setenv("SENTINEL_DSP", "off")
        py = dsp.analyze(path, layout=layout)
        assert k["backend"] == "kernel" and py["backend"] == "python"
        assert abs(k["integrated_lufs"] - py["integrated_lufs"]) <= GATE_LU
        assert abs(k["digital_peak_dbfs"] - py["digital_peak_dbfs"]) <= GATE_DB
        assert abs(k["true_peak_dbtp"] - py["true_peak_dbtp"]) <= GATE_DB
        assert (k["correlation"]["silent_channels"]
                == py["correlation"]["silent_channels"])
        assert ([(i, j) for i, j, _ in k["correlation"]["duplicate_pairs"]]
                == [(i, j) for i, j, _ in py["correlation"]["duplicate_pairs"]])
        assert abs(k["correlation"]["mean_offdiag"]
                   - py["correlation"]["mean_offdiag"]) <= GATE_EPS
        ke, pe = k["energy_profile"], py["energy_profile"]
        assert ke["windows"] == pe["windows"]
        if pe["median_dev"] is not None:
            assert abs(ke["median_dev"] - pe["median_dev"]) <= GATE_EPS
            assert abs(ke["p95_dev"] - pe["p95_dev"]) <= GATE_EPS


def test_kernel_json_is_parseable_document(monkeypatch, tmp_path, stereo_wav):
    """The raw subprocess emits one valid JSON document (the ADR-4 boundary)."""
    binary = _find_kernel()
    if not binary:
        pytest.skip("no sentinel-dsp binary available ($SENTINEL_DSP / $PATH)")
    r = subprocess.run([binary, stereo_wav, "--layout", "2.0"],
                       capture_output=True, text=True, encoding="utf-8",
                       timeout=60)
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["tool"] == "sentinel-dsp"
    assert doc["channels"] == 2
