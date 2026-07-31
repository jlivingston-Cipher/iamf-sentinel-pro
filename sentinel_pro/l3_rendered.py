"""L3 rendered QC (S-31x/S-32x) — Phase 2, PRD R3.

Decodes each mix presentation's declared layouts through the reference-decoder
oracles (subprocess-only, ADR-3), measures BS.1770-4 loudness/true peak with
Sentinel's own DSP (dsp.py, cross-checked against ffmpeg ebur128 to the WP1
G5 gate), compares measured vs declared, and runs channel-identity and spatial
heuristics on the decoded PCM — the half of the F4/F19/F20/F21 catalogue that
descriptors cannot see.

Only runs when the engine is given a Toolchain (``--l3``); without one the
module contributes nothing, keeping Phase-1 behavior byte-identical.
"""

from __future__ import annotations
from collections.abc import Iterator

from sentinel.checks.base import CheckContext, finding
from sentinel.findings import Finding
from sentinel import model as m
from sentinel.layouts import LOUDSPEAKER_LAYOUT, LAYOUT_TYPE_BINAURAL, LAYOUT_TYPE_SS
from .oracle import SOUND_SYSTEM_TO_LAYOUT, LAYOUT_CHANNELS

# Channel names in the reference decoders' output order (WP1 decode-verify
# convention, BS.2051), for actionable messages.
CHANNEL_NAMES: dict[str, list[str]] = {
    "2.0": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "5.1.2": ["L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf"],
    "5.1.4": ["L", "R", "C", "LFE", "Ls", "Rs", "Ltf", "Rtf", "Ltb", "Rtb"],
    "7.1": ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs"],
    "7.1.2": ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs", "Ltf", "Rtf"],
    "7.1.4": ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs",
              "Ltf", "Rtf", "Ltb", "Rtb"],
    "3.1.2": ["L", "R", "C", "LFE", "Ltf", "Rtf"],
}

# LFE indices per layout (excluded from silent-channel judgment: a silent LFE
# is normal content, not a misroute signature).
_LFE_INDEX = {k: (v.index("LFE") if "LFE" in v else None)
              for k, v in CHANNEL_NAMES.items()}


def _native_analysis_layout(mod: m.IAMFModel, ae_ids) -> tuple[str | None, bool]:
    """(analysis layout, beds_only) for channel-identity/spatial analysis.

    beds_only is True when every referenced element is channel-based — the
    precondition for the duplication/silence checks (S-320/S-321): bed channels
    are discrete essence, so near-perfect duplication is a misroute signature.
    A scene (ambisonics) render legitimately drives symmetric speakers with
    identical waveforms (executed on dlb_obj_static1: single object -> r=1.0
    speaker pairs), so scenes get only the origin-collapse heuristic (S-322).
    """
    best = None
    best_ch = 0
    beds_only = True
    for aid in ae_ids:
        ae = mod.audio_elements.get(aid)
        if not ae:
            continue
        if ae.ambisonics is not None:
            beds_only = False
            if 12 > best_ch:
                best, best_ch = "7.1.4", 12       # widest spatial view of a scene
        for layer in ae.channel_layers:
            cl = LOUDSPEAKER_LAYOUT.get(layer.loudspeaker_layout)
            if not cl:
                continue
            # "Stereo" analyses as the 2.0 render. "Mono" is deliberately
            # excluded: rendered 2.0 from a mono bed has identical L/R by
            # construction, so duplication is no misroute signature there.
            name = "2.0" if cl.name == "Stereo" else cl.name
            if name in LAYOUT_CHANNELS and cl.channels > best_ch:
                best, best_ch = name, cl.channels
    return best, beds_only


def _decode_cached(ctx: CheckContext, layout: str, decoder: str,
                   mix_id: int | None = None):
    cache = ctx.l3_cache.setdefault("decodes", {})
    key = (layout, decoder, mix_id)
    if key not in cache:
        cache[key] = ctx.toolchain.decode(ctx.source_path, layout, decoder,
                                          mix_id=mix_id)
    return cache[key]


def _measured(ctx: CheckContext, layout: str, decoder: str,
              mix_id: int | None = None):
    """(DecodeResult, measurements|None) cached per (layout, decoder, mix)."""
    from . import dsp
    cache = ctx.l3_cache.setdefault("measures", {})
    key = (layout, decoder, mix_id)
    if key in cache:
        return cache[key]
    dec = _decode_cached(ctx, layout, decoder, mix_id)
    meas = None
    if dec.ok:
        # One file-level measurement call (ADR-6b backend switch, doc 35):
        # kernel subprocess when the sentinel-dsp binary is present, numpy
        # reference otherwise. The full analysis document replaces the old
        # cached PCM — the channel-identity heuristics read it instead.
        a = dsp.analyze(dec.wav_path, layout=layout)
        meas = {
            "sr": a["sample_rate"],
            "integrated": a["integrated_lufs"],
            "digital_peak": a["digital_peak_dbfs"],
            "true_peak": a["true_peak_dbtp"],
            "analysis": a,
        }
    cache[key] = (dec, meas)
    return cache[key]


def run(mod: m.IAMFModel, ctx: CheckContext) -> Iterator[Finding]:
    """X-76a shape: per sub-mix, the declared-layout oracle checks
    (S-310..S-314) then the channel-identity/spatial heuristics
    (S-320..S-322) — two generators, yielded in the pre-split order."""
    tc = getattr(ctx, "toolchain", None)
    if tc is None or not tc.available:
        return
    if not mod.mix_presentations or not mod.audio_elements:
        return

    tol_lu = float(ctx.l3_options.get("loudness_tolerance_lu", 0.5))
    tol_pk = float(ctx.l3_options.get("peak_tolerance_db", 0.3))
    oracles = tc.oracles()
    primary = oracles[0]

    for mp in mod.mix_presentations:
        mix_id = mp.mix_presentation_id if len(mod.mix_presentations) > 1 else None
        for sm in mp.sub_mixes:
            where0 = f"mix_presentation {mp.mix_presentation_id}"
            yield from _declared_layout_findings(ctx, sm, where0, mix_id,
                                                 oracles, primary,
                                                 tol_lu, tol_pk)
            yield from _identity_spatial_findings(mod, ctx, sm, where0,
                                                  mix_id, primary)


def _declared_layout_findings(ctx: CheckContext, sm: m.SubMix, where0: str,
                              mix_id: int | None, oracles: list[str],
                              primary: str, tol_lu: float,
                              tol_pk: float) -> Iterator[Finding]:
    """Oracle decode + measured-vs-declared per declared layout (S-310..S-314)."""
    for ll in sm.layouts:
        if ll.layout_type == LAYOUT_TYPE_BINAURAL:
            continue          # binauralizer disabled toolchain-wide (F10)
        if ll.layout_type != LAYOUT_TYPE_SS:
            continue
        layout = SOUND_SYSTEM_TO_LAYOUT.get(
            ll.sound_system if ll.sound_system is not None else -1)
        where = f"{where0} / {ll.label}"
        if layout is None:
            continue          # no decoder mapping (e.g. 22.2) — skip
        results = {d: _measured(ctx, layout, d, mix_id) for d in oracles}
        ok = {d: r for d, (r, _mm) in results.items() if r.ok}
        if not ok:
            def _why(r):
                if r.note:
                    return r.note
                last = r.stderr_tail.splitlines()[-1] if r.stderr_tail else ""
                return f"rc={r.returncode} {last}".strip()
            tails = "; ".join(f"{d}: {_why(r)}"
                              for d, (r, _mm) in results.items())
            yield finding(ctx, "S-312",
                          f"declared layout {ll.label} failed to decode on every "
                          f"available oracle", where=where, found=tails)
            continue
        if len(oracles) > 1 and len(ok) == 1 and len(results) > 1:
            missing = [d for d in oracles if d not in ok]
            yield finding(ctx, "S-313",
                          f"{'/'.join(missing)} could not decode a layout that "
                          f"{'/'.join(ok)} decoded", where=where,
                          found=results[missing[0]][0].note or
                          results[missing[0]][0].stderr_tail)

        # measured vs declared — primary oracle's render
        use = primary if primary in ok else next(iter(ok))
        meas = results[use][1]
        if ll.integrated_loudness is not None:
            delta = meas["integrated"] - ll.integrated_loudness
            if abs(delta) > tol_lu:
                yield finding(ctx, "S-310",
                              f"measured integrated loudness {meas['integrated']:.2f} LKFS "
                              f"differs from declared {ll.integrated_loudness:.2f} LKFS "
                              f"by {delta:+.2f} LU (tolerance ±{tol_lu} LU, oracle {use})",
                              where=where,
                              expected=f"{ll.integrated_loudness:.2f} LKFS",
                              found=f"{meas['integrated']:.2f} LKFS")
        if ll.digital_peak is not None:
            dp_delta = meas["digital_peak"] - ll.digital_peak
            if abs(dp_delta) > tol_pk:
                yield finding(ctx, "S-311",
                              f"measured digital peak {meas['digital_peak']:.2f} dBFS "
                              f"differs from declared {ll.digital_peak:.2f} dBFS by "
                              f"{dp_delta:+.2f} dB (oracle {use})", where=where,
                              expected=f"{ll.digital_peak:.2f} dBFS",
                              found=f"{meas['digital_peak']:.2f} dBFS")
        if ll.true_peak is not None:
            tp_delta = meas["true_peak"] - ll.true_peak
            if abs(tp_delta) > tol_pk:
                yield finding(ctx, "S-311",
                              f"measured true peak {meas['true_peak']:.2f} dBTP differs "
                              f"from declared {ll.true_peak:.2f} dBTP by {tp_delta:+.2f} dB "
                              f"(oracle {use})", where=where,
                              expected=f"{ll.true_peak:.2f} dBTP",
                              found=f"{meas['true_peak']:.2f} dBTP")

        # measured headroom (F21 full)
        if meas["true_peak"] >= -0.1:
            yield finding(ctx, "S-314",
                          f"rendered {ll.label} true peak {meas['true_peak']:.2f} dBTP "
                          "at/above full scale (measured on decoded PCM)",
                          where=where, found=f"{meas['true_peak']:.2f} dBTP")

        # oracle divergence on loudness (both rendered this layout)
        if len(ok) == 2:
            m1 = results[oracles[0]][1]
            m2 = results[oracles[1]][1]
            div = m1["integrated"] - m2["integrated"]
            if abs(div) > tol_lu:
                yield finding(ctx, "S-313",
                              f"oracles disagree on integrated loudness: "
                              f"{oracles[0]}={m1['integrated']:.2f} LKFS vs "
                              f"{oracles[1]}={m2['integrated']:.2f} LKFS "
                              f"(Δ {div:+.2f} LU) — ORACLE_DIVERGENCE", where=where,
                              found=f"Δ {div:+.2f} LU")


def _identity_spatial_findings(mod: m.IAMFModel, ctx: CheckContext,
                               sm: m.SubMix,
                               where0: str, mix_id: int | None,
                               primary: str) -> Iterator[Finding]:
    """Channel identity + spatial heuristics on the native analysis render
    (S-320..S-322, per presentation)."""
    layout, beds_only = _native_analysis_layout(mod, sm.audio_element_ids)
    if layout is None:
        return
    for cand in (layout, "5.1", "2.0"):
        dec, meas = _measured(ctx, cand, primary, mix_id)
        if dec.ok:
            layout = cand
            break
    if not dec.ok or meas is None:
        return
    analysis = meas["analysis"]
    corr = analysis["correlation"]
    names = CHANNEL_NAMES.get(layout,
                              [str(i) for i in range(analysis["channels"])])
    where = f"{where0} / rendered {layout} ({primary})"

    if beds_only:
        lfe_i = _LFE_INDEX.get(layout)
        silent = [i for i in corr["silent_channels"] if i != lfe_i]
        if silent:
            labels = ", ".join(names[i] if i < len(names) else str(i)
                               for i in silent)
            yield finding(ctx, "S-320",
                          f"{len(silent)} rendered channel(s) silent while the "
                          f"program carries signal: {labels} — F4 missing-channel "
                          "signature", where=where, found=labels)

        dups = corr["duplicate_pairs"]
        if dups:
            labels = ", ".join(
                f"{names[i] if i < len(names) else i}≈{names[j] if j < len(names) else j}"
                f" (r={r:.4f})" for i, j, r in dups[:6])
            yield finding(ctx, "S-321",
                          f"{len(dups)} channel pair(s) carry effectively the same "
                          f"signal: {labels} — F4 duplication signature", where=where,
                          found=labels)

    prof = analysis["energy_profile"]
    if prof["windows"] >= 4 and prof["p95_dev"] is not None:
        if prof["p95_dev"] < 0.05 and corr["mean_offdiag"] > 0.95:
            yield finding(ctx, "S-322",
                          "energy vector pinned to the equal-energy baseline for "
                          "the whole program with near-identical channels — "
                          "origin-collapse signature "
                          f"(p95 |E-E0|={prof['p95_dev']:.3f})", where=where,
                          found=f"p95 |E-E0|={prof['p95_dev']:.3f}")
