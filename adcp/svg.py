"""Minimal deterministic SVG rendering: beam-quality grid and profile plot."""
from __future__ import annotations

import html
from typing import Any, Dict, List

_COLORS = {
    "four_beam_ok": "#2e8b57",
    "three_beam": "#d9a300",
    "rejected": "#b33a3a",
}


def quality_matrix(view: Dict[str, Any]) -> str:
    cells = view.get("cells", [])
    cell = 46
    w = 4 * cell + 40
    h = len(cells) * cell + 44
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
             f'height="{h}" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    parts.append('<text x="8" y="16" font-size="12">波束质量 ping '
                 f'{view.get("ping")} (每格带节点身份)</text>')
    for k in range(4):
        parts.append(f'<text x="{40 + k*cell + 12}" y="34" font-size="10" '
                     f'fill="#555">B{k}</text>')
    for ci, c in enumerate(cells):
        y = 40 + ci * cell
        parts.append(f'<text x="4" y="{y+26}" font-size="10">c{c["cell"]}'
                     f'</text>')
        q = {b["beam"]: b for b in c["beam_quality"]}
        for k in range(4):
            bq = q[k]
            fill = _COLORS[c["status"]] if bq["valid"] else "#b33a3a"
            uid = f'{c["quality_node"]}-b{k}'
            title = html.escape(uid + " " + ",".join(bq["reasons"]))
            parts.append(
                f'<rect data-id="{uid}" x="{40+k*cell+2}" y="{y+2}" '
                f'width="{cell-4}" height="{cell-4}" rx="4" fill="{fill}">'
                f'<title>{title}</title></rect>')
            if not bq["valid"]:
                parts.append(f'<text x="{40+k*cell+14}" y="{y+28}" '
                             f'font-size="14" fill="white">×</text>')
    parts.append("</svg>")
    return "".join(parts)


def profile_plot(profile: Dict[str, Any]) -> str:
    cells: List[dict] = profile.get("cells", [])
    w, h = 420, 240
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
             f'height="{h}" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    parts.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    parts.append('<text x="12" y="20" font-size="12">垂向剖面 earth-absolute '
                 '(m/s)</text>')
    if not cells:
        parts.append('<text x="12" y="60" fill="#b33a3a">无有效剖面</text>'
                     '</svg>')
        return "".join(parts)
    comps = [(0, "east", "#1f6fb2"), (1, "north", "#2e8b57"),
             (2, "up", "#8b5a2b")]
    x0, y0, pw, ph = 60.0, 40.0, 300.0, 170.0
    vmin = min(min(c["mean_velocity_earth_absolute"]) for c in cells)
    vmax = max(max(c["mean_velocity_earth_absolute"]) for c in cells)
    span = max(vmax - vmin, 1e-6)
    for gi in range(5):
        gx = x0 + gi * pw / 4
        parts.append(f'<line x1="{gx}" y1="{y0}" x2="{gx}" y2="{y0+ph}" '
                     f'stroke="#e3e3e3"/>')
    for ci, c in enumerate(cells):
        yy = y0 + ci * ph / max(len(cells) - 1, 1)
        parts.append(f'<text x="8" y="{yy+4}" font-size="10">c{c["cell"]}'
                     f'</text>')
    for idx, name, color in comps:
        pts = []
        for ci, c in enumerate(cells):
            val = c["mean_velocity_earth_absolute"][idx]
            xx = x0 + (val - vmin) / span * pw
            yy = y0 + ci * ph / max(len(cells) - 1, 1)
            pts.append(f"{xx},{yy}")
        uid = "profile-" + name
        parts.append(f'<polyline data-id="{uid}" points="{" ".join(pts)}" '
                     f'fill="none" stroke="{color}" stroke-width="2">'
                     f'<title>{html.escape(uid)}</title></polyline>')
        for li, label in enumerate(("east", "north", "up")):
            parts.append(f'<rect x="{280+li*46}" y="6" width="10" '
                         f'height="10" fill="{comps[li][2]}"/>')
            parts.append(f'<text x="{293+li*46}" y="15" font-size="9">'
                         f'{label}</text>')
    parts.append("</svg>")
    return "".join(parts)
