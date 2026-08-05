"""iamf-sentinel-pro — L3 rendered-QC plugin for the iamf-sentinel core (Apache-2.0).

L3 rendered QC (decoder oracles + clean-room BS.1770-4 + F29 calibration),
source-referenced ADM fidelity QC (S-33x), the F4 essence-misroute killer on
decoded PCM, and platform profile packs. Requires the ``sentinel`` open core
(iamf-sentinel). Measurement runs on the compiled ``sentinel-dsp`` kernel when
the binary is present (ADR-6b backend switch, doc 35) or on the numpy
reference path (``iamf-sentinel-pro[numpy]``); adm-compare and ``diff
--render`` always need numpy. L3/adm-compare additionally need the reference
toolchain (decoder_main / iamfdec) and, for adm-compare, EAR (ear-render).

Intent-conformance QC (S-34x, ``sentinel intent-compare``) compares an
authoring session's intent sidecar against the delivered ADM BW64 — the class
that is structurally valid, loudness-conformant, and still not what the session
authored. Its per-object isolation legs need EAR (``ear-render``); the roster,
trajectory and level legs do not.

The core discovers this package by import (see PLUGIN_SEAM.md): its presence
enables ``--l3``, ``diff --render``, ``adm-compare``, ``intent-compare``, and
the profile packs in ``sentinel_pro/profiles/``.
"""
__version__ = "0.3.4"
