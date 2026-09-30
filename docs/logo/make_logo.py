"""Build the capybara README logo and the legacy hamster icon assets.

The wheel is the looped span of layers -- the same wheel, run again and again -- and
its rim is built from blocks. More laps is more computation per token; the wheel does
not get any bigger, which is the point.

    python3 make_logo.py      # writes the SVGs next to this file
"""

import base64
import math
from pathlib import Path

ORANGE, ORANGE2, TEAL, DARK, BG = "#C25A28", "#E48A4E", "#2E6F78", "#16212B", "#F7F6F2"
FUR, FUR2, BELLY, PINK, PINK2, CHEEK = "#EBAA5C", "#D48A3C", "#FDEBD3", "#F29CA3", "#DE8088", "#F6A9AE"


def arc(cx, cy, rad, a0, a1):
    x0, y0 = cx + rad * math.cos(math.radians(a0)), cy + rad * math.sin(math.radians(a0))
    x1, y1 = cx + rad * math.cos(math.radians(a1)), cy + rad * math.sin(math.radians(a1))
    large = 1 if (a1 - a0) % 360 > 180 else 0
    return f"M{x0:.1f},{y0:.1f} A{rad},{rad} 0 {large} 1 {x1:.1f},{y1:.1f}"


def hamster(x, y, scale=1.0, tilt=-8):
    """A hamster facing right, mid-stride; (x, y) is the centre of its body."""
    return f'''
<g transform="translate({x},{y}) rotate({tilt}) scale({scale})">
  <defs><clipPath id="body"><ellipse cx="0" cy="0" rx="96" ry="74"/></clipPath></defs>
  <path d="M-30,48 L-38,60" stroke="{FUR2}" stroke-width="18" stroke-linecap="round"/>
  <ellipse cx="-40" cy="64" rx="13" ry="7.5" fill="{PINK2}"/>
  <path d="M38,48 L49,60" stroke="{FUR2}" stroke-width="16" stroke-linecap="round"/>
  <ellipse cx="52" cy="64" rx="12" ry="7" fill="{PINK2}"/>
  <circle cx="-94" cy="6" r="10" fill="{FUR2}"/>
  <ellipse cx="0" cy="0" rx="96" ry="74" fill="{FUR}"/>
  <g clip-path="url(#body)">
    <ellipse cx="6" cy="58" rx="84" ry="40" fill="{BELLY}"/>
    <path d="M-100,-10 C-90,-70 -20,-90 30,-78 C-20,-66 -60,-40 -76,4 Z" fill="{FUR2}" opacity="0.45"/>
  </g>
  <ellipse cx="62" cy="10" rx="30" ry="24" fill="{BELLY}"/>
  <circle cx="-14" cy="-66" r="21" fill="{FUR}"/><circle cx="-14" cy="-66" r="12" fill="{PINK}"/>
  <circle cx="30" cy="-72" r="19" fill="{FUR}"/><circle cx="30" cy="-72" r="10.5" fill="{PINK}"/>
  <circle cx="30" cy="-16" r="12.5" fill="{DARK}"/><circle cx="34.5" cy="-21" r="4.6" fill="#fff"/><circle cx="27" cy="-11" r="2" fill="#fff"/>
  <circle cx="72" cy="-12" r="11.5" fill="{DARK}"/><circle cx="76" cy="-16.5" r="4.2" fill="#fff"/><circle cx="69" cy="-7.5" r="1.8" fill="#fff"/>
  <ellipse cx="14" cy="10" rx="13" ry="8" fill="{CHEEK}" opacity="0.9"/>
  <ellipse cx="88" cy="14" rx="9" ry="7" fill="{CHEEK}" opacity="0.9"/>
  <ellipse cx="56" cy="4" rx="7" ry="5" fill="{PINK2}"/>
  <path d="M46,13 q5,6 10,0 q5,6 10,0" stroke="{DARK}" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M-54,42 L-68,54" stroke="{FUR}" stroke-width="20" stroke-linecap="round"/>
  <ellipse cx="-72" cy="58" rx="14" ry="8" fill="{PINK}"/>
  <path d="M58,40 L74,52" stroke="{FUR}" stroke-width="17" stroke-linecap="round"/>
  <ellipse cx="78" cy="56" rx="12.5" ry="7.5" fill="{PINK}"/>
</g>'''


def wheel(cx, cy, r, blocks=12, gap=3.0, width=26, teal=TEAL):
    rim = "".join(
        f'<path d="{arc(cx, cy, r, i * 360 / blocks + gap / 2 - 90, (i + 1) * 360 / blocks - gap / 2 - 90)}" '
        f'stroke="{ORANGE if i % 2 == 0 else ORANGE2}" stroke-width="{width}" fill="none"/>'
        for i in range(blocks))
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + (r - 12) * math.cos(math.radians(a)):.1f}" '
        f'y2="{cy + (r - 12) * math.sin(math.radians(a)):.1f}" stroke="{teal}" stroke-width="4" opacity="0.28"/>'
        for a in range(0, 360, 45))
    return spokes, rim


def rotation_arrow(cx, cy, rad, a0, a1, width=11, teal=TEAL):
    """A curved arrow going round the wheel, clockwise from a0 to a1 (degrees)."""
    tip = math.radians(a1)
    tx, ty = cx + rad * math.cos(tip), cy + rad * math.sin(tip)
    # the arrowhead points along the tangent, clockwise
    dx, dy = -math.sin(tip), math.cos(tip)
    nx, ny = math.cos(tip), math.sin(tip)
    h, w = 22, 14
    p1 = (tx + dx * h, ty + dy * h)
    p2 = (tx + nx * w, ty + ny * w)
    p3 = (tx - nx * w, ty - ny * w)
    return (f'<path d="{arc(cx, cy, rad, a0, a1)}" stroke="{teal}" stroke-width="{width}" fill="none" stroke-linecap="round"/>'
            f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" fill="{teal}"/>')


def icon(size=512, background=True, teal=TEAL, bg=BG, inner=False):
    """The square mark. inner=True returns just the drawing, for nesting in a lockup."""
    cx, cy, r = 256, 238, 180
    spokes, rim = wheel(cx, cy, r, teal=teal)
    motion = "".join(
        f'<path d="M{x0},{y} L{x1},{y}" stroke="{teal}" stroke-width="7" stroke-linecap="round" opacity="{op}"/>'
        for x0, x1, y, op in ((122, 152, 292, 0.5), (114, 148, 318, 0.35), (124, 150, 344, 0.22)))
    back = f'<rect width="512" height="512" rx="104" fill="{bg}"/>' if background else ""
    drawing = f'''
  {back}
  <ellipse cx="256" cy="480" rx="140" ry="12" fill="{DARK}" opacity="0.08"/>
  <path d="M{cx},{cy} L190,478 M{cx},{cy} L322,478" stroke="{teal}" stroke-width="10" stroke-linecap="round" opacity="0.8"/>
  {spokes}
  <circle cx="{cx}" cy="{cy}" r="13" fill="{teal}"/><circle cx="{cx}" cy="{cy}" r="5" fill="{bg}"/>
  {rim}
  {rotation_arrow(cx, cy, r + 30, -58, 2, width=9, teal=teal)}
  {motion}
  {hamster(252, 327, 0.92, tilt=-4)}'''
    if inner:
        return drawing
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 512 512">{drawing}\n</svg>'


def text_path(text, size, x, y, weight="bold"):
    """Outline text in DejaVu Sans (bundled with matplotlib, freely redistributable)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath, TextToPath

    prop = FontProperties(family="DejaVu Sans", weight=weight)
    d = []
    for verts, code in TextPath((0, 0), text, size=size, prop=prop).iter_segments():
        pts = [(x + verts[i], y - verts[i + 1]) for i in range(0, len(verts), 2)]
        if code == Path.MOVETO:
            d.append("M%.1f,%.1f" % pts[0])
        elif code == Path.LINETO:
            d.append("L%.1f,%.1f" % pts[0])
        elif code == Path.CURVE3:
            d.append("Q%.1f,%.1f %.1f,%.1f" % (*pts[0], *pts[1]))
        elif code == Path.CURVE4:
            d.append("C%.1f,%.1f %.1f,%.1f %.1f,%.1f" % (*pts[0], *pts[1], *pts[2]))
        elif code == Path.CLOSEPOLY:
            d.append("Z")
    advance = TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0] * size / prop.get_size_in_points()
    return " ".join(d), advance


def lockup(dark=False):
    """Approved capybara plus the original outlined wordmark, for the README."""
    teal = "#7FC4CC" if dark else TEAL
    muted = "#AAB4BC" if dark else "#5A6570"
    x0, base = 330, 158
    d_open, w_open = text_path("open-", 132, x0, base)
    d_loop, w_loop = text_path("loopify", 132, x0 + w_open, base)
    d_tag, w_tag = text_path("looped language models, in the open", 40, x0 + 6, base + 76, weight="normal")
    width = int(x0 + max(w_open + w_loop, w_tag) + 40)
    image_data = base64.b64encode(Path(__file__).with_name("capybara-wheel-v1.png").read_bytes()).decode("ascii")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="300" viewBox="0 0 {width} 300" role="img" aria-labelledby="logo-title">
  <title id="logo-title">open-loopify: a capybara running in a wheel</title>
  <image x="10" y="10" width="280" height="280" xlink:href="data:image/png;base64,{image_data}"/>
  <path d="{d_open}" fill="{teal}"/>
  <path d="{d_loop}" fill="{ORANGE if not dark else "#EE9A62"}"/>
  <path d="{d_tag}" fill="{muted}"/>
</svg>'''


if __name__ == "__main__":
    open("open-loopify-icon.svg", "w").write(icon())
    open("open-loopify-icon-transparent.svg", "w").write(icon(background=False))
    open("open-loopify-logo.svg", "w").write(lockup())
    open("open-loopify-logo-dark.svg", "w").write(lockup(dark=True))
    print("wrote open-loopify-icon.svg, -icon-transparent.svg, -logo.svg, -logo-dark.svg")
