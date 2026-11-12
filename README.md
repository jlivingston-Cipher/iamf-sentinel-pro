# iamf-sentinel-pro — L3 rendered QC plugin for iamf-sentinel

The free core (`iamf-sentinel`) reads the file. This plugin *decodes* it and measures the
truth:

- **L3 rendered QC** — decode each mix presentation × declared layout through the reference
  decoders (as subprocess oracles) and measure **BS.1770-4** integrated loudness and true
  peak with an independent, calibrated implementation — catching declared-vs-measured
  loudness lies that descriptor checks can't see.
- **Channel identity on decoded PCM** — catches the pure essence-misroute corruption
  (descriptor-clean, PCM-scrambled) that even L2 cannot see from the bitstream alone.
- **Source-referenced ADM fidelity QC** (`adm-compare`, S-33x) — compares an IAMF encode
  against its BW64/ADM master. Two of its checks judge every static bed channel and object
  on its own — S-333 its level, S-334 its direction, calibrated to BS.2127 rather than to
  any converter — straight from the LPCM an ADM import writes (`--conversion-only` runs
  them without a renderer or decoder).
- **Intent-conformance QC** (`intent-compare`, S-34x) — compares an authoring session's
  **intent sidecar** against the delivered ADM BW64: bed roster, predicted objects, authored
  trajectory, stem levels and gain automation, and (via EAR isolation renders) the rendered
  dominant speaker. This is the class that is structurally valid, loudness-conformant, and
  still not what the session authored — the dropped object, the frozen position, the lost
  gain move. The roster, trajectory and level legs need no EAR; the isolation legs do, and
  the tool refuses rather than silently comparing less.
- **ADM delivery measurements** (`sentinel_pro.adm_delivery`, S-522..S-525, S-533) — what the
  core's `sentinel adm-check` cannot judge from metadata, measured from the master's audio in one
  pass: the LFE low-pass against a stated filter shape (drawn as a spectrum in the HTML report),
  silent and near-silent channels, whether a 96 kHz master holds content above 24 kHz without a
  brick-wall edge at the 48 kHz band edge, and, given the stereo master, the offset and extra
  tail found by cross-correlating a fold-down against it (and an album's gapless boundaries).
  Thresholds that are not the specification's own numbers are named in each row and provisional.
- **Platform profile packs** (e.g. YouTube ingest) with the loudness-measurement calibration
  that keeps *correct* files from being flagged at tight tolerances.

The core discovers this package by import (see `PLUGIN_SEAM.md`): its presence enables
`sentinel validate --l3`, `diff --render`, `adm-compare`, `intent-compare`, and the profile
packs in `sentinel_pro/profiles/`.

> **Free software, Apache-2.0.**

Code comments throughout cite an internal numbered design docset (`doc NN`), ADRs, and pre-registered expectation labels — **[DESIGN-NOTES.md](DESIGN-NOTES.md)** explains the notation and indexes every cited document.

## Install & backends

```bash
pip install iamf-sentinel-pro            # kernel-backed: no scientific stack required
pip install "iamf-sentinel-pro[numpy]"   # adds the numpy reference backend (quotes: zsh eats bare brackets)
```

Measurement runs on the compiled **`sentinel-dsp`** kernel when the binary is present
(`$SENTINEL_DSP` explicit path, else `bin/sentinel-dsp` under the toolchain root, else
`$PATH`; `SENTINEL_DSP=off` forces the reference path) or on the numpy reference implementation. A present-but-broken kernel is an
**execution error**, never a silent fallback. `adm-compare` and `diff --render` always
need the `[numpy]` extra.

Build the kernel:

```bash
cmake -S sentinel-dsp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
```

Every change to either measurement implementation is gated by the differential suite
(`sentinel-dsp/tests/differential.py` + `tests/test_dsp_backend.py`).

**Verified platforms.** Every push runs the Pro test suite on **Linux, macOS, and Windows**
against **Python 3.11 and 3.12** — [`ci.yml`](.github/workflows/ci.yml) is the claim; the matrix
is the evidence. That matrix covers the **Python** side of Pro; the C++ kernel and its
differential against `dsp.py` are gated separately, on Linux only, by
[`dsp-differential.yml`](.github/workflows/dsp-differential.yml) — a green matrix here says
nothing about the kernel, and vice versa. Oracle-gated, sample-gated and kernel-gated tests
skip in that environment by design, and on Windows so do the tests that depend on POSIX
executable semantics (shell stubs, `PATH` resolution of extensionless files). Nothing is
claimed for a platform that does not have a green leg.

L3 and adm-compare additionally need the reference toolchain (decoder_main / iamfdec) and,
for adm-compare, EAR (`ear-render`); `adm-compare --conversion-only` needs neither. The OCI recipes under `oci/` package the toolchain
reproducibly; `oci/layer-scan.py` is the release-CI hygiene check.

## Related projects

- [`iamf-sentinel`](https://github.com/jlivingston-Cipher/iamf-sentinel) — the free, stdlib-pure
  core this package plugs into (L1/L2 + CI contract)
- [`iamf-loom`](https://github.com/jlivingston-Cipher/iamf-loom) — manifest-driven IAMF packager;
  its validation gate uses this stack's measurement
- [`iamf-adm-corpus`](https://github.com/jlivingston-Cipher/iamf-adm-corpus) — synthetic ADM
  corpus + harness behind the `adm-compare` method
- [`iamf-sentinel-mcp`](https://github.com/jlivingston-Cipher/iamf-sentinel-mcp) — MCP server
  exposing the validator and packager to agent runtimes
- [`Inseglet`](https://github.com/inseglet-audio/inseglet) — authoring-side companion (a native
  REAPER MCP extension); its BS.1770-4 bed-weight tables are pinned against this package's
  conformance vectors as an independent second witness

## License

Apache-2.0 (see `LICENSE` / `NOTICE`). This tree was designed as the commercial half of an
open-core split; the project has since adopted a full open-source posture and both sides are
Apache-2.0 — the plugin seam remains as architecture, not a license boundary. The measurement
DSP (Python and C++), the ADM reader, and all checks are original clean-room works (ITU-R
BS.1770-4, ITU-R BS.2088 / EBU Tech 3285, ISO/IEC 14496-12). Reference decoders and EAR are
invoked only as subprocess oracles. Commercial support and consulting are available — see the
core repo's `SUPPORT.md`.
