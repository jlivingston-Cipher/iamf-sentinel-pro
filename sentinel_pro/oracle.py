"""Reference-decoder subprocess oracles for L3 rendered QC (Phase 2, ADR-3).

Decoders are invoked strictly as subprocesses (the licensing firewall, ADR-4)
and never trusted blindly:

- iamf-tools ``decoder_main`` is the primary rendering oracle (the engine whose
  sibling measured the declared loudness in iamf-tools-encoded files).
- libiamf ``iamfdec`` is the second opinion. Its sharp edges are handled per
  WP1 F8: ``-o3`` takes a full output FILE path, and it exits 0 after decoding
  0 frames on some failures — success is judged by output-file size, never rc.
- Divergence between the two oracles is itself a reported finding
  (ORACLE_DIVERGENCE, PRD Open Question 5).

Every invocation gets a unique working directory (WP3 F22: fixed /tmp names
corrupt concurrent runs — never inherit that hazard).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import shutil
import subprocess
import tempfile

# decoder_main --output_layout strings we can request, keyed by the
# IAMF sound_system value used in mix-presentation loudness layouts.
SOUND_SYSTEM_TO_LAYOUT: dict[int, str] = {
    0: "2.0",       # A
    1: "5.1",       # B
    2: "5.1.2",     # C
    3: "5.1.4",     # D
    8: "7.1",       # I
    9: "7.1.4",     # J
    10: "7.1.2",
    11: "3.1.2",
}

LAYOUT_CHANNELS: dict[str, int] = {
    "2.0": 2, "5.1": 6, "5.1.2": 8, "5.1.4": 10,
    "7.1": 8, "7.1.2": 10, "7.1.4": 12, "3.1.2": 6, "binaural": 2,
}


@dataclass
class DecodeResult:
    ok: bool
    decoder: str
    layout: str
    wav_path: str | None = None
    returncode: int | None = None
    stderr_tail: str = ""
    note: str = ""


@dataclass
class Toolchain:
    """Locations of the oracle binaries. Any member may be None (absent)."""
    decoder_main: str | None = None
    iamfdec: str | None = None
    work_root: str | None = None
    _tmp: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.decoder_main is not None or self.iamfdec is not None

    def oracles(self) -> list[str]:
        out = []
        if self.decoder_main:
            out.append("decoder_main")
        if self.iamfdec:
            out.append("iamfdec")
        return out

    # -- discovery ---------------------------------------------------------
    @classmethod
    def discover(cls, root: str | None = None) -> "Toolchain":
        """Find oracle binaries from an explicit root, $SENTINEL_TOOLCHAIN,
        or $PATH."""
        candidates = []
        if root:
            candidates.append(root)
        env = os.environ.get("SENTINEL_TOOLCHAIN")
        if env:
            candidates.append(env)

        tc = cls()
        for c in candidates:
            dm = os.path.join(c, "src/build-iamf/decoder_main")
            if tc.decoder_main is None and os.access(dm, os.X_OK):
                tc.decoder_main = dm
            for rel in ("src/build-iamfdec/iamfdec",
                        "src/build-libiamf/test/tools/iamfdec/iamfdec"):
                idc = os.path.join(c, rel)
                if tc.iamfdec is None and os.access(idc, os.X_OK):
                    tc.iamfdec = idc
        for name, attr in (("decoder_main", "decoder_main"), ("iamfdec", "iamfdec")):
            if getattr(tc, attr) is None:
                p = shutil.which(name)
                if p:
                    setattr(tc, attr, p)
        return tc

    # -- decode ------------------------------------------------------------
    def _workdir(self) -> str:
        d = tempfile.mkdtemp(prefix="sentinel-l3-",
                             dir=self.work_root or None)
        self._tmp.append(d)
        return d

    def cleanup(self):
        for d in self._tmp:
            shutil.rmtree(d, ignore_errors=True)
        self._tmp.clear()

    def materialize_raw(self, obu_bytes: bytes, tag: str = "extracted") -> str:
        """Write reconstructed raw-IAMF bytes to a working file for the oracles."""
        wd = self._workdir()
        p = os.path.join(wd, f"{tag}.iamf")
        with open(p, "wb") as fh:
            fh.write(obu_bytes)
        return p

    def prepare_input(self, path: str) -> str:
        """Return a path the oracles can decode: raw files pass through; MP4s
        are reconstructed to raw .iamf via the clean-room extractor (the
        reference decoders are raw-only or crash-prone on A/V MP4)."""
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return path
        if len(data) < 8 or data[4:8] != b"ftyp":
            return path
        from sentinel.container.mp4 import parse_mp4, extract_iamf_stream
        raw = extract_iamf_stream(data, parse_mp4(data))
        if raw is None:
            return path
        return self.materialize_raw(raw)

    def decode(self, path: str, layout: str, decoder: str,
               mix_id: int | None = None, timeout: float = 120.0) -> DecodeResult:
        """Render `path` to `layout` with `decoder`; judge by output file."""
        wd = self._workdir()
        wav = os.path.join(wd, f"render_{decoder}_{layout.replace('.', '_')}.wav")
        if decoder == "decoder_main":
            if not self.decoder_main:
                return DecodeResult(False, decoder, layout, note="decoder_main not available")
            cmd = [self.decoder_main, f"--input_filename={path}",
                   f"--output_filename={wav}", f"--output_layout={layout}"]
            if mix_id is not None:
                cmd.append(f"--mix_id={mix_id}")
        elif decoder == "iamfdec":
            if not self.iamfdec:
                return DecodeResult(False, decoder, layout, note="iamfdec not available")
            # iamfdec -s<ss> takes the sound-system value itself (WP1 decode_verify).
            ss = next((k for k, v in SOUND_SYSTEM_TO_LAYOUT.items() if v == layout), None)
            if ss is None:
                return DecodeResult(False, decoder, layout,
                                    note=f"no sound-system mapping for {layout}")
            mode = "1" if path.lower().endswith(".mp4") else "0"
            # -disable_limiter: the default -1 dB peak limiter would bias
            # loudness/peak measurement (WP1 G8 note, resolved Phase 2).
            cmd = [self.iamfdec, f"-i{mode}", f"-s{ss}", "-disable_limiter"]
            if mix_id is not None:
                cmd += ["-mp", str(mix_id)]
            cmd += ["-o3", wav, path]
        else:
            return DecodeResult(False, decoder, layout, note=f"unknown decoder {decoder}")

        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            return DecodeResult(False, decoder, layout, note="decode timeout")
        except OSError as e:
            return DecodeResult(False, decoder, layout, note=f"spawn failed: {e}")

        tail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-3:])
        # F8: iamfdec can exit 0 having written nothing — judge by the file.
        ok = r.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 44
        return DecodeResult(ok, decoder, layout, wav_path=wav if ok else None,
                            returncode=r.returncode, stderr_tail=tail)
