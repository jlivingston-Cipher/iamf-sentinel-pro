# Design notes — reading the provenance references in this codebase

This project was developed against an internal, numbered design docset: every work
cycle produced a numbered document recording what was decided, what was measured, and
the evidence behind both. Code comments cite those records rather than restating them
(`doc 35`, `ADR-6b`, `E-B2`, …). The docset itself is not published, but the notation
is simple and the index below preserves the context each citation carries.

The core validator's copy of this file (`iamf-sentinel/DESIGN-NOTES.md`) carries the
shared notation; entries specific to this plugin are indexed here.

## Notation

| Form | Meaning |
|---|---|
| `doc NN` (§ optional) | A numbered internal design/evidence document; see the index below. |
| `ADR-1` … `ADR-6b` | Architecture Decision Records. The ones that shape this tree: **ADR-3** (clean-room rule: parser from the spec, DSP from ITU-R BS.1770-4, no code derivation from GPL/reference sources), **ADR-4** (reference decoders and EAR are subprocess-only oracles, never linked), **ADR-6/6b** (the C++ measurement kernel: scope, then the port, differential-gated against the Python reference). |
| `WP1`, `WP3` | Work packages: WP1 = toolchain validation, WP3 = ADM→IAMF fidelity study (the source of adm-compare and its thresholds). |
| `F1` … `F34` | Failure-mode register entries — mapped to checks in the core repo's `F_TO_CHECK.md`. |
| `G1` … `G11` | WP1 validation gates (doc 10): measured facts/criteria for the reference toolchain (e.g. G5: the own-BS.1770-vs-ffmpeg loudness cross-check on decoded renders). |
| `S-NNN` | Sentinel check IDs (`sentinel checks -v`); this plugin contributes the S-3xx rendered-QC checks. |
| `E-…` | A pre-registered expectation: written before the experiment ran, then confirmed or refuted (e.g. `E-B2` from the doc-31 adm-compare study). |

## Index of cited documents

| Doc | What it established |
|---|---|
| 03 | The verified-findings register (evidence-graded F-numbers). |
| 10 | WP1 toolchain validation: gates G1–G11 and the first failure modes. |
| 13 | The packager (iamf-loom) PRD. |
| 19 | Sentinel Phase 2: this plugin's L3 rendered QC (decoder oracles + BS.1770-4). |
| 26 | The open-core seam split — the architecture now described in the core repo's `PLUGIN_SEAM.md`. |
| 31 | Phase 3b: adm-compare (source-referenced fidelity QC) and its pre-registered expectations (E-B…); opened F30. |
| 32 | ADR-6 revisit: kernel scope decision (what the C++ kernel does and does not cover). |
| 33 | The `sentinel-dsp` kernel port, with the 15-vector differential gate against `dsp.py` (loudness within 1e-3 LU pre-registered; achieved ~1e-11). |
| 34 | Seam refactor accompanying the kernel work. |
| 35 | The backend switch: kernel primary, numpy an optional extra; a present-but-broken kernel is an execution error, never a silent numpy fallback (the standing contract in `dsp.kernel_path`). |
| 36 | OCI image line, first cycle: base image shape and build args. |
| 38 | OCI stage 2: the toolchain image (both `TOOLCHAIN_REALIZATION`s). |
| 39 | OCI path A: fully in-image source build of the toolchain. |
| 40 | OCI residuals: `.dockerignore` discipline; the adm-compare image variant built on demand, not published as a v1 tag. |
| 41 | OCI runtime hygiene: build-stage isolation, wheel-only runtime, the `oci/layer-scan.py` full-layer audit. |
| 48 | The decision to publish the whole stack as free software, Apache-2.0. |
| 64 | Normative-revision audit: established that IAMF v1.1.0 cites BS.1770-4 by revision (so -4, not -5, is the conformance target). |
| 65 | BS.1770-4 Tables 4/5 read: the conformant channel weights in `dsp.py`/`main.cpp` (the `FFMPEG_WEIGHTS` variant table exists to reproduce a third-party measurement path for calibration comparison — see the comments at its definition). |

The differential test (`sentinel-dsp/tests/differential.py`) and the acceptance
fixtures are the living, shipped form of the kernel's evidence; the index exists so
the remaining pointers stay meaningful.
