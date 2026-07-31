# The plugin seam — `iamf-sentinel` ↔ `iamf-sentinel-pro`

## Current contract (operative)

The seam is **architecture, not a license boundary** — both sides are Apache-2.0 (full
open-source posture, docs 48/51). The core stays stdlib-pure and CI-frictionless; the
decoder oracles, DSP, and their dependencies live behind plugin discovery.

- **Discovery is by import.** The core discovers `sentinel_pro` with a guarded import; its
  presence enables `sentinel validate --l3`, `diff --render`, `adm-compare`, and the
  profile packs in `sentinel_pro/profiles/`. No entry-point plumbing.
- **Degradation contracts (all three enforced by tests):**
  1. Without the plugin, an `--l3` / `--render` request degrades to a one-line
     "requires iamf-sentinel-pro" notice — the free checks still run.
  2. A **toolchain passed without the plugin is an execution error** (S-000 FAIL, exit 2):
     the caller asked for rendered QC; silently exiting 0 without it would be a false pass.
  3. A **present-but-broken `sentinel-dsp` kernel is an execution error** (`KernelError` →
     S-000, exit 2), never a silent numpy fallback — that would mask kernel regressions.
     `SENTINEL_DSP=off` is the explicit escape hatch to the reference path.
- **File assignment:** the core (`iamf-sentinel`) owns the L1/L2 clean-room parser, the
  descriptor-derived checks, the container walk, fixtures, and the CI contract; the plugin
  (`iamf-sentinel-pro`) owns `dsp.py` (BS.1770-4 + identity DSP), `oracle.py` (decoder
  subprocess wrappers), `l3_rendered.py` (S-31x/S-32x), `adm_compare.py`, the platform
  profile packs, and the `sentinel-dsp` C++ kernel. The frozen detail is in the design
  record below.
- **Measurement backend:** `sentinel_pro.dsp.analyze()` is the file-level facade; the
  compiled kernel (`$SENTINEL_DSP` explicit path, else `$PATH`) or the numpy reference
  (`[numpy]` extra) computes the same document, gated by the differential suite.

Everything below is the executed design record — kept intact, in order, as history.

---

## Design record — the open-core seam as executed

This documents the D1 split boundary as executed against the Phase-2 tree (`sentinel-phase2.zip`,
doc 19). The split is unusually clean because Phase 2 already isolated the L3 code behind lazy imports
gated on `toolchain is not None` — doc 21 predicted "the Phase-2 tree already almost has this shape,"
and it does.

### Verified this session

- The OSS core (`sentinel-oss/`) **imports and runs L1/L2 with no numpy and no `sentinel_pro` on the
  path.** Proven: `python -m sentinel validate cd_bed_stereo.iamf` → exit 0, correct report, on the
  isolated OSS tree.
- **Zero eager numpy imports** in the OSS core. The two numpy references (`cli.py`, `diff.py`) are both
  lazy and live inside `--l3` / `--render` code paths only.
- The OSS test suite passes standalone: **36 passed, 12 skipped** (the 12 skips are the L3 tests that
  require the decoder oracle) — i.e. every free-core test is green without the plugin.

### File assignment (frozen)

**OSS core — `iamf-sentinel`, Apache-2.0, stdlib-pure (`sentinel-oss/`):**
`sentinel/` — `reader.py`, `parser.py` (L1), `model.py`, `layouts.py`, `findings.py`, `report.py`,
`engine.py`, `cli.py`, `diff.py` (descriptor half); `checks/{base,l1_structural,l2_semantics,
l3_loudness,container}.py` (note: `l3_loudness` is the *descriptor-derived* declared-value checks —
stdlib, no decoder — and is correctly OSS); `container/mp4.py`; `profiles/generic.toml` (the CI
contract). Plus `fixtures/{build,mutate,gen_fixtures}.py`, `tests/test_sentinel.py`, `F_TO_CHECK.md`.

**Pro plugin — `iamf-sentinel-pro`, Apache-2.0 since doc 51 (numpy optional since doc 35), needs the toolchain (`sentinel-pro/`):**
`sentinel_pro/` — `dsp.py` (own BS.1770-4 + identity DSP — the F29 calibration moat), `oracle.py`
(decoder-oracle subprocess wrappers), `l3_rendered.py` (the `S-31x/S-32x` measured checks).
Plus `profiles/youtube.toml` (first profile pack — see the open question below),
`fixtures/gen_l3_fixtures.py` (F4 essence-misroute pair generator; needs the toolchain),
`tests/test_l3.py`.

### The one refactor (≈10 lines) — turn 3 lazy imports into graceful plugin discovery

Today the three L3 hand-offs import from within the (now-removed) `sentinel` subtree. Repoint them at
`sentinel_pro` and wrap in try/except so the free core degrades to a one-line notice instead of an
`ImportError`:

1. `sentinel/engine.py` (~line 128), inside `if ctx.toolchain is not None:`
   ```python
   try:
       from sentinel_pro import l3_rendered
   except ImportError:
       report.findings.append(finding(... "S-000", "L3 requested but iamf-sentinel-pro not installed"))
   else:
       modules.append(l3_rendered)
   ```
2. `sentinel/cli.py` (~line 26), toolchain discovery: `from sentinel_pro.oracle import Toolchain`
   inside a `try/except ImportError` that makes `--l3` print `requires iamf-sentinel-pro` and exit
   cleanly (free checks still run).
3. `sentinel/diff.py` (~lines 105–107), the `--render` path: `from sentinel_pro import dsp` and
   `from sentinel_pro.l3_rendered import CHANNEL_NAMES, _native_analysis_layout`, same guard →
   `render_note = "requires iamf-sentinel-pro"`.

`sentinel_pro` registers its checks by being import-discoverable; no entry-point plumbing is required
for the MVP, though a `setuptools` entry point is the clean v1 (`[project.entry-points."sentinel.plugins"]`).

### Two open sub-questions to settle in the repo-layout session (flagged, not silently decided)

1. **`profiles/youtube.toml` — OSS or Pro?** Doc 21 puts *profile packs* on the commercial side, and
   this scaffold follows that (youtube in `sentinel-pro`). But the citation wedge ("iamf-sentinel-clean =
   won't bounce on YouTube") is strongest if at least one *recognizable platform* profile is visible for
   free. Options: (a) keep only `generic` free (as scaffolded); (b) make `youtube` a free loss-leader and
   reserve *future* packs (Samsung TV Plus, etc.) + auto-update + SLA for Pro. **Recommend (b)** — it
   costs little (the youtube profile is already public via WP1 G9) and directly funds the reference
   strategy — but it is a founder call, so the scaffold leaves it in Pro pending your decision.
2. **`diff --render` lives in an OSS file** (`diff.py`) but needs Pro DSP. Kept in OSS with a lazy guard
   (option above). Alternative: move the whole rendered-diff into `sentinel_pro`. The guard is simpler and
   keeps `diff` (descriptor) fully free; recommended as-is.

### License posture (confirm with counsel — D1 sub-decision)

OSS core: **Apache-2.0** (patent grant; matches the Apache/BSD ecosystem norm and the D4 goodwill posture).
Alternative considered: BSD-3-Clause-Clear (upstream iamf-tools/libiamf's license) — weaker patent
position. The core is original clean-room work (ADR-3), so either is available to us; Apache-2.0 is the
recommendation. **The public repo lands only after D2 clearance and the D4 filings** (doc 21 sequencing).

---

### Update — wiring executed (July 21, 2026, doc 34)

The refactor described above is **done**, against the Phase-3 tree (docs 30–31), in this
package (`d1-open-core-phase3.zip`, superseding `d1-open-core-scaffold.zip`):

- The three OSS hand-offs (`engine.py`, `cli.py`, `diff.py`) now import from `sentinel_pro`
  under `try/except` with graceful degradation. One deliberate refinement vs the sketch above:
  in `engine.py`, a toolchain passed without the plugin is an **execution error** (S-000 FAIL,
  exit 2) — the caller asked for rendered QC; silently exiting 0 without it would be a false
  pass.
- `load_profile` searches the pro package's `profiles/` when importable (platform packs are
  Pro-side), and the OSS suite's youtube-profile test skips when no pack is present.
- The Pro tree's package-internal imports are rewired (the scaffold's copies were verbatim and
  would not have imported); `adm_compare.py` (Phase 3b) is included Pro-side per doc 31;
  `sentinel_pro/__init__.py` + `pyproject.toml` added.
- `wasm/make_inspector.py` now builds the embedded core zip from these trees directly.

Verified: OSS standalone 47 passed/1 skipped (no numpy, no plugin); combined suite **76
passed, 0 skipped** with toolchain+samples+corpus+EAR; WASM native-vs-browser differential
23/23 identical; the F4 `--l3` headline reproduces through the seam. Full evidence: doc 34.

### Update — measurement backend switch executed (July 21, 2026, doc 35)

A second seam now exists *inside* the Pro plugin: `sentinel_pro.dsp.analyze()` is the
file-level measurement facade (one WAV in, one JSON-shaped document out — the ADR-4 boundary
applied to our own kernel). When the compiled `sentinel-dsp` binary is present (`$SENTINEL_DSP`
explicit path, else `$PATH`) measurement runs in the kernel subprocess; otherwise the numpy
reference functions compute the same document. `SENTINEL_DSP=off` forces the reference path.

Contract points (mirroring the doc-34 degradation philosophy):

- A present-but-failing kernel raises `KernelError` → S-000 execution error, exit 2. Never a
  silent fallback to numpy — that would mask kernel regressions.
- numpy is now an **extra** of `iamf-sentinel-pro` (`[numpy]`): a kernel-backed install runs
  `validate --l3` with no scientific stack (the ADR-6b OCI-image property). `adm-compare` and
  `diff --render` still require the extra — their math is outside the kernel surface.
- The kernel source lives at `sentinel-pro/sentinel-dsp/` (Pro-side IP; moved from the
  standalone doc-33 landing). The differential (synthetic half, toolchain-free) gates every
  change to either implementation: `sentinel-dsp/tests/differential.py` + the
  `tests/test_dsp_backend.py` mini-differential, wired in
  `.github/workflows/dsp-differential.yml`.

### Update — both sides public (July 27, 2026, docs 48/51)

D1 was revisited to **full open source (A′, doc 48)**: `iamf-sentinel-pro` — and Loom, and
the `sentinel-dsp` kernel — are re-licensed **Apache-2.0** (doc 51). The seam documented
above **stays exactly as built**, but its meaning changes: it is architecture, not a license
boundary — the core remains stdlib-pure and CI-frictionless while the decoder oracles, DSP,
and their dependencies live behind the plugin discovery. All degradation contracts
(graceful notice without the plugin; toolchain-without-plugin = execution error; broken
kernel = execution error, never silent fallback) are unchanged.

Historical wording above ("commercial plugin", the doc-21 sequencing, "lands only after D2
clearance") is kept as a record of the executed split. **Open sub-question 1
(`profiles/youtube.toml` — OSS or Pro?) is mooted:** everything is free; the pack stays
Pro-side purely as an architecture choice (packs ship with the measurement stack that
enforces them). Publication is now gated on the founder's name/release-shape calls and the
D4 filings (docs 48/50/51).
