import io
import os
import re
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from PIL import Image, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(HERE, "Poppins-Bold.ttf")
FONT_URL = "https://fonts.gstatic.com/s/poppins/v24/pxiByp8kv8JHgFVrLCz7V1s.ttf"

NAVY = tuple(c / 255 for c in (5, 28, 63))
CORAL = tuple(c / 255 for c in (253, 61, 57))
X_HEIGHT = 76.0
TARGET_SPAN = 629.0
TARGET_WV_GAP = 9.0
OUT_SCALE = 10
UNIT = 100 / 72


def ensure_font():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)


def glyph_patch(text, x, y, fp, fs_pt, sx, color):
    p = TextPath((0, 0), s=text, size=fs_pt, prop=fp)
    t = Affine2D().scale(sx * UNIT, -UNIT).translate(x, y)
    return PathPatch(t.transform_path(p), color=color, lw=0)


def render(fp, fs_pt, sx, x_veri, x_org, y_base, w, h, dpi):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(glyph_patch("straw", x_org, y_base, fp, fs_pt, sx, NAVY))
    ax.add_patch(glyph_patch("veri", x_org + x_veri, y_base, fp, fs_pt, sx,
                             CORAL))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    return fig, buf


def min_row_gap(img, px_per_unit):
    a = np.array(img).astype(int)
    navy = (a[:, :, 3] > 128) & (a[:, :, 2] - a[:, :, 0] > 15)
    coral = (a[:, :, 3] > 128) & (a[:, :, 0] - a[:, :, 1] > 60)
    gaps = [np.where(coral[y])[0].min() - np.where(navy[y])[0].max()
            for y in range(a.shape[0]) if navy[y].any() and coral[y].any()]
    return min(gaps) / px_per_unit


def save_svg(fig, path, w, h):
    fig.savefig(path, transparent=True)
    with open(path) as f:
        svg = f.read()
    svg = re.sub(r'viewBox="[^"]*"', f'viewBox="0 0 {w:.2f} {h:.2f}"', svg, count=1)
    svg = re.sub(r'width="[^"]*"', f'width="{w:.2f}"', svg, count=1)
    svg = re.sub(r'height="[^"]*"', f'height="{h:.2f}"', svg, count=1)
    with open(path, "w") as f:
        f.write(svg)


def main():
    ensure_font()
    matplotlib.rcParams["svg.fonttype"] = "path"
    fp = FontProperties(fname=FONT_PATH)
    f1000 = ImageFont.truetype(FONT_PATH, size=1000)
    _, y0, _, y1 = f1000.getbbox("x")
    fs_pt = X_HEIGHT / ((y1 - y0) / 1000.0) * 0.72

    e_s, e_v = (TextPath((0, 0), s=t, size=fs_pt, prop=fp).get_extents()
                for t in ("straw", "veri"))
    h = (max(e.ymax for e in (e_s, e_v))
         - min(e.ymin for e in (e_s, e_v))) * UNIT
    y_base = max(e.ymax for e in (e_s, e_v)) * UNIT
    c_gap = (e_v.x0 - e_s.x1) * UNIT
    c_span = (e_v.x1 - e_s.x0) * UNIT

    def solve(t_gap):
        sx = (TARGET_SPAN - t_gap) / (c_span - c_gap)
        return sx, t_gap - c_gap * sx

    def probe(sx, x_veri):
        fig, buf = render(fp, fs_pt, sx, x_veri, -e_s.x0 * UNIT * sx,
                          y_base, TARGET_SPAN, h, 150)
        plt.close(fig)
        buf.seek(0)
        return min_row_gap(Image.open(buf).convert("RGBA"), 1.5)

    x_lo = solve(TARGET_WV_GAP)[1]
    x_hi = x_lo + 20.0
    for _ in range(12):
        x_mid = (x_lo + x_hi) / 2
        if probe((TARGET_SPAN - x_mid) / c_span, x_mid) < TARGET_WV_GAP:
            x_lo = x_mid
        else:
            x_hi = x_mid
    x_veri = (x_lo + x_hi) / 2
    sx = (TARGET_SPAN - x_veri) / c_span

    fig, buf = render(fp, fs_pt, sx, x_veri, -e_s.x0 * UNIT * sx, y_base,
                      TARGET_SPAN, h, 100 * OUT_SCALE)
    save_svg(fig, os.path.join(HERE, "strawveri_wordmark.svg"), TARGET_SPAN, h)
    fig.savefig(os.path.join(HERE, "strawveri_wordmark.png"),
                dpi=100 * OUT_SCALE, transparent=True)
    plt.close(fig)

    buf.seek(0)
    gap = min_row_gap(Image.open(buf).convert("RGBA"), OUT_SCALE)
    print(f"span {TARGET_SPAN:.0f}, w-v min aralık {gap:.1f} birim -> "
          f"strawveri_wordmark.png ({int(TARGET_SPAN * OUT_SCALE)}x"
          f"{int(h * OUT_SCALE)}), .svg")


if __name__ == "__main__":
    main()
