#!/usr/bin/env python3
"""Render the README charts as static SVG (light and dark variant each) from assets/chart-data.json.

  python3 scripts/make_readme_charts.py [--data assets/chart-data.json] [--out assets]

Stdlib only. Colors: two categorical hues (blue = sixi-asi / new, orange = upstream / before) plus a neutral gray
for other engines; values and labels always in text ink, never in the series color.
"""
import argparse
import json
import os
from xml.sax.saxutils import escape

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e4e3de",
                  blue="#2a78d6", orange="#eb6834", gray="#8f8d86"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#33322f",
                 blue="#3987e5", orange="#d95926", gray="#77756f"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 860


def svg_open(h, t, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" '
            f'font-family="{FONT}" role="img" aria-label="{escape(title)}">',
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>',
            f'<rect width="{W}" height="{h}" rx="8" fill="{t["surface"]}"/>']


def text(x, y, s, fill, size=13, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(s)}</text>')


def hbar(x, y, w, h, fill, r=4):
    """Bar anchored at x (square baseline end), rounded data end."""
    w = max(w, r)
    return (f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" fill="{fill}"/>')


def legend(x, y, items, t):
    out = []
    for label, color in items:
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(text(x + 18, y, label, t["text2"], 12))
        x += 30 + 6.6 * len(label)
    return out


def chart_engines(d, t):
    rows = sorted(d["rows"], key=lambda r: -r["tok_s"])
    left, right, top, bh, gap = 330, 70, 100, 18, 12
    h = top + len(rows) * (bh + gap) + 44
    vmax = max(r["tok_s"] for r in rows)
    ticks = list(range(0, int(vmax) + 11, 10))
    sx = (W - left - right) / ticks[-1]
    o = svg_open(h, t, d["title"], d["subtitle"])
    o.append(text(24, 30, d["title"], t["text"], 16, weight=600))
    o.append(text(24, 50, d["subtitle"], t["text2"], 12))
    o += legend(24, 74, [("sixi-asi", t["blue"]), ("other engines", t["gray"])], t)
    y0 = top - 8
    y1 = top + len(rows) * (bh + gap) - gap + 8
    for v in ticks:
        x = left + v * sx
        o.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(x, y1 + 18, str(v), t["muted"], 11, "middle"))
    o.append(text(left + (W - left - right) / 2, y1 + 36, "decode tokens per second (higher is better)", t["muted"], 11, "middle"))
    for i, r in enumerate(rows):
        y = top + i * (bh + gap)
        color = t["blue"] if r["group"] == "sixi" else t["gray"]
        o.append(text(left - 12, y + bh - 5, r["label"], t["text"] if r.get("bold") else t["text2"], 13, "end",
                      600 if r.get("bold") else 400))
        o.append(hbar(left, y, r["tok_s"] * sx, bh, color))
        o.append(text(left + r["tok_s"] * sx + 8, y + bh - 5, f'{r["tok_s"]:.1f}', t["text"], 13, weight=600))
    o.append("</svg>")
    return "\n".join(o)


def chart_verify(d, t):
    left, right, top, ph = 64, 170, 112, 250
    h = top + ph + 58
    xs = sorted({p[0] for s in d["series"] for p in s["points"]})
    xmax = max(xs)
    ymax = d["y_max"]
    sx = (W - left - right) / (xmax - 1)
    sy = ph / ymax
    X = lambda v: left + (v - 1) * sx
    Y = lambda v: top + ph - v * sy
    o = svg_open(h, t, d["title"], d["subtitle"])
    o.append(text(24, 30, d["title"], t["text"], 16, weight=600))
    o.append(text(24, 50, d["subtitle"], t["text2"], 12))
    for v in range(0, ymax + 1, 40):
        o.append(f'<line x1="{left}" y1="{Y(v):.1f}" x2="{W - right}" y2="{Y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(left - 8, Y(v) + 4, str(v), t["muted"], 11, "end"))
    for v in xs:
        o.append(text(X(v), top + ph + 18, str(v), t["muted"], 11, "middle"))
    o.append(text(left + (W - left - right) / 2, top + ph + 40, "rows in one forward pass (1 anchor + k draft tokens)", t["muted"], 11, "middle"))
    o.append(text(24, top - 12, "ms per forward", t["muted"], 11))
    fy = Y(d["floor_ms"])
    o.append(f'<line x1="{left}" y1="{fy:.1f}" x2="{W - right}" y2="{fy:.1f}" stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(text(W - right + 10, fy + 4, d["floor_label"], t["text2"], 12))
    colors = {"blue": t["blue"], "orange": t["orange"]}
    for s in d["series"]:
        c = colors[s["color"]]
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in s["points"])
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b in s["points"]:
            o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" fill="{c}" stroke="{t["surface"]}" stroke-width="2"/>')
        a, b = s["points"][-1]
        o.append(text(W - right + 10, Y(b) + 4 + s.get("label_dy", 0), s["label"], t["text"], 12, weight=600))
        for a, b, dy in s.get("callouts", []):
            o.append(text(X(a), Y(b) + dy, f"{b:.0f} ms", t["text"], 12, "middle", 600))
    o += legend(24, 74, [(s["legend"], colors[s["color"]]) for s in d["series"]], t)
    o.append("</svg>")
    return "\n".join(o)


def chart_longctx(d, t):
    left, right, top, bh, inner, gap = 150, 90, 100, 16, 4, 20
    rows = d["rows"]
    h = top + len(rows) * (2 * bh + inner + gap) + 40
    vmax = max(r["before_s"] for r in rows)
    step = 50
    ticks = list(range(0, int(vmax) + step, step))
    sx = (W - left - right) / ticks[-1]
    o = svg_open(h, t, d["title"], d["subtitle"])
    o.append(text(24, 30, d["title"], t["text"], 16, weight=600))
    o.append(text(24, 50, d["subtitle"], t["text2"], 12))
    o += legend(24, 74, [(d["before_label"], t["orange"]), (d["after_label"], t["blue"])], t)
    y0 = top - 8
    y1 = top + len(rows) * (2 * bh + inner + gap) - gap + 8
    for v in ticks:
        x = left + v * sx
        o.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(x, y1 + 18, f"{v} s", t["muted"], 11, "middle"))
    for i, r in enumerate(rows):
        y = top + i * (2 * bh + inner + gap)
        o.append(text(left - 12, y + bh + inner / 2 + 4, r["label"], t["text2"], 13, "end"))
        o.append(hbar(left, y, r["before_s"] * sx, bh, t["orange"]))
        o.append(text(left + r["before_s"] * sx + 8, y + bh - 3, f'{r["before_s"]:.0f} s', t["text"], 12, weight=600))
        o.append(hbar(left, y + bh + inner, r["after_s"] * sx, bh, t["blue"]))
        o.append(text(left + max(r["after_s"] * sx, 4) + 8, y + 2 * bh + inner - 3, f'{r["after_s"]:.1f} s', t["text"], 12, weight=600))
    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="assets/chart-data.json")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    d = json.load(open(a.data))
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("engines", chart_engines), ("verify-cost", chart_verify), ("long-context", chart_longctx)):
        for mode, t in THEMES.items():
            path = os.path.join(a.out, f"{name}-{mode}.svg")
            open(path, "w").write(fn(d[name], t) + "\n")
            print("wrote", path)


if __name__ == "__main__":
    main()
