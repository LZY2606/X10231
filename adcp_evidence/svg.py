"""Server-side SVG renderers (profile, vectors, beam quality matrix)."""
from __future__ import annotations

import html
from typing import Sequence


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def profile_svg(ensembles: list[dict], width: int = 560,
                height: int = 300) -> str:
    """Vertical velocity profile; 3-beam and 4-beam ensembles are separate."""
    rows = []
    for ens in ensembles:
        if ens["earth_mean_ms"] is None:
            continue
        rows.append(ens)
    if not rows:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
    max_speed = max(
        max(abs(v) for v in e["earth_mean_ms"]) for e in rows
    ) or 1.0
    colors = {"four_beam": "#2563eb", "three_beam": "#d97706"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<line x1="120" y1="10" x2="120" y2="%d" stroke="#94a3b8"/>'
        % (height - 20),
    ]
    names = ("east", "north", "up")
    for i, n in enumerate(names):
        parts.append(
            f'<text x="130" y="{22 + i*14}" fill="{colors["four_beam"] if i<2 else "#16a34a"}">'
            f'{n}</text>')
    rowh = min(34, (height - 40) // max(1, len(rows)))
    for i, ens in enumerate(rows):
        y = 24 + i * rowh
        color = colors.get(ens["solution_kind"], "#64748b")
        vals = ens["earth_mean_ms"]
        parts.append(
            f'<text x="10" y="{y + 10}">cell {ens["cell"]} '
            f'({_esc(ens["solution_kind"].replace("_", " "))}, n={ens["n"]})'
            f'</text>')
        for comp in range(3):
            length = int(90 * vals[comp] / max_speed)
            parts.append(
                f'<rect x="130" y="{y + comp*10 - 6}" width="{abs(length)}" '
                f'height="7" fill="{color if comp<2 else "#16a34a"}"/>')
            parts.append(
                f'<text x="{230 if length>=0 else 226}" y="{y + comp*10 + 2}">'
                f'{vals[comp]:+.3f}</text>')
        parts.append(f'<text x="280" y="{y+10}" fill="#64748b">'
                     f'id fp:{_esc(ens.get("fingerprint", "")[:0]) or ""}'
                     f'{", ".join(ens["members"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def vector_chain_svg(chain: dict, size: int = 320) -> str:
    """Three triads: instrument, ship, earth vectors step by step."""
    triads = [
        ("instrument xyz (fwd,stbd,down)", chain.get("instrument_ms"),
         "#0f766e"),
        ("ship (fwd,stbd,up)",
         [v for v in chain.get("ship_fsu_ms", [])], "#7c3aed"),
        ("earth ENU", chain.get("earth_ms"), "#2563eb"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{3*78 + 10}" font-family="sans-serif" font-size="11">']
    for i, (label, vec, color) in enumerate(triads):
        y = 18 + i * 78
        parts.append(f'<text x="8" y="{y}" fill="{color}">{_esc(label)}</text>')
        if not vec:
            parts.append(f'<text x="8" y="{y + 18}">—</text>')
            continue
        names = ("x", "y", "z") if i < 1 else (
            ("f", "s", "u") if i == 1 else ("E", "N", "U"))
        for comp in range(3):
            bar = int(70 * min(1.0, abs(vec[comp]) / 0.6))
            parts.append(
                f'<rect x="70" y="{y + 8 + comp*16}" width="{bar}" '
                f'height="10" fill="{color}"/>')
            parts.append(
                f'<text x="8" y="{y + 18 + comp*16}">{names[comp]}</text>')
            parts.append(
                f'<text x="{150 + bar}" y="{y + 18 + comp*16}">'
                f'{vec[comp]:+.4f} m/s</text>')
    parts.append("</svg>")
    return "".join(parts)


def beam_quality_svg(beam_quality: dict, width: int = 520) -> str:
    color = {"valid": "#16a34a", "marginal": "#d97706", "invalid": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="92" '
        'font-family="sans-serif" font-size="11">']
    for i, v in enumerate(beam_quality["verdicts"]):
        x = 12 + i * 126
        state = ("invalid" if not v["valid"] else
                 "marginal" if v["marginal"] else "valid")
        parts.append(
            f'<rect x="{x}" y="8" width="116" height="70" rx="6" '
            f'fill="{color[state]}" fill-opacity="0.12" '
            f'stroke="{color[state]}"/>')
        parts.append(
            f'<text x="{x+8}" y="26">beam {v["beam"]+1}: {state}</text>')
        corr = "NA" if v["correlation"] is None else f'{v["correlation"]:.0f}'
        amp = "NA" if v["amplitude"] is None else f'{v["amplitude"]:.0f}'
        parts.append(f'<text x="{x+8}" y="44">corr {corr} amp {amp}</text>')
        reasons = ",".join(v["reasons"] + v.get("marginal_reasons", [])) or "-"
        parts.append(
            f'<text x="{x+8}" y="64">{_esc(reasons[:22])}</text>')
    parts.append("</svg>")
    return "".join(parts)
