import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

R_SCALE = 1.12
W, H = 290, 320
MARGIN = 20
OUT_SCALE = 10
SUPERSAMPLE = 1.2

DOTS = [
    ("gövde", 158.0,  92.8, 10.63, 253,  67,  66),
    ("gövde", 107.8,  93.1, 10.58, 254,  67,  67),
    ("gövde",  81.1,  93.5, 10.14, 255,  65,  67),
    ("gövde", 184.8,  93.5, 10.54, 254,  64,  68),
    ("gövde",  54.2,  94.0, 10.12, 254,  64,  68),
    ("gövde", 211.8,  94.0, 10.48, 254,  64,  67),
    ("gövde", 132.2,  94.4,  7.52, 253,  70,  71),
    ("gövde", 234.9, 100.9,  7.04, 252,  68,  71),
    ("gövde",  30.5, 101.0,  7.13, 254,  66,  70),
    ("gövde", 132.1, 117.4,  8.66, 254,  65,  70),
    ("gövde", 158.2, 120.3, 11.41, 254,  63,  67),
    ("gövde", 106.9, 120.6, 11.33, 254,  62,  66),
    ("gövde", 186.8, 121.9, 11.38, 254,  62,  66),
    ("gövde",  78.3, 122.0, 11.60, 254,  62,  66),
    ("gövde",  48.7, 122.4, 11.77, 254,  61,  65),
    ("gövde", 215.2, 122.6, 11.58, 253,  62,  67),
    ("gövde", 244.0, 123.9, 10.19, 253,  64,  68),
    ("gövde",  21.4, 124.0,  9.99, 254,  62,  68),
    ("gövde", 133.0, 145.7, 12.10, 254,  62,  66),
    ("gövde", 163.8, 149.9, 13.34, 254,  62,  64),
    ("gövde", 102.0, 150.6, 13.39, 254,  62,  64),
    ("gövde", 196.3, 151.2, 12.36, 253,  62,  65),
    ("gövde",  70.2, 151.6, 12.43, 254,  63,  66),
    ("gövde", 225.6, 153.2, 11.16, 254,  62,  66),
    ("gövde",  40.5, 153.3, 11.51, 255,  60,  64),
    ("gövde",  13.4, 154.6, 10.06, 254,  64,  68),
    ("gövde", 252.6, 155.1,  9.77, 254,  64,  68),
    ("gövde", 117.9, 181.3, 14.01, 254,  66,  62),
    ("gövde", 151.9, 182.0, 13.27, 254,  65,  63),
    ("gövde", 242.0, 182.1, 10.75, 254,  66,  67),
    ("gövde",  22.8, 182.3,  9.89, 254,  65,  67),
    ("gövde", 183.5, 182.4, 12.23, 254,  64,  64),
    ("gövde",  84.4, 182.4, 12.75, 255,  65,  62),
    ("gövde", 213.2, 182.5, 11.54, 254,  64,  66),
    ("gövde",  53.5, 182.8, 12.20, 254,  63,  64),
    ("gövde", 231.9, 210.8,  9.82, 253,  69,  66),
    ("gövde", 200.1, 212.0, 12.40, 254,  69,  63),
    ("gövde",  33.5, 212.1,  9.96, 255,  64,  65),
    ("gövde", 168.3, 212.1, 13.10, 254,  68,  61),
    ("gövde", 133.0, 212.6, 13.73, 253,  71,  61),
    ("gövde",  65.7, 212.7, 12.31, 254,  68,  63),
    ("gövde",  98.6, 212.7, 13.02, 254,  69,  61),
    ("gövde", 217.6, 239.0, 10.40, 253,  73,  65),
    ("gövde",  47.6, 240.1,  9.90, 254,  69,  64),
    ("gövde", 185.8, 241.2, 12.27, 254,  74,  59),
    ("gövde",  79.5, 241.4, 11.73, 255,  72,  61),
    ("gövde", 150.8, 241.7, 12.82, 254,  76,  59),
    ("gövde", 114.2, 241.8, 12.33, 254,  76,  59),
    ("gövde", 200.8, 265.4,  8.89, 254,  78,  63),
    ("gövde",  64.5, 266.6,  8.26, 254,  80,  62),
    ("gövde",  96.8, 267.6, 11.49, 255,  79,  58),
    ("gövde", 168.1, 268.0, 11.73, 255,  78,  58),
    ("gövde", 133.6, 268.7, 12.68, 254,  81,  57),
    ("gövde",  84.8, 288.1,  6.59, 255,  87,  63),
    ("gövde", 181.0, 288.5,  6.55, 254,  85,  62),
    ("gövde", 112.0, 292.6,  9.51, 254,  87,  58),
    ("gövde", 153.1, 293.4, 10.33, 255,  85,  58),
    ("gövde", 132.8, 309.1,  6.75, 254,  91,  62),
    ("yaprak", 133.1,  15.9, 12.64,  86, 192,  84),
    ("yaprak", 103.8,  40.2, 11.81,  86, 194,  83),
    ("yaprak",  64.4,  40.2, 11.93,  87, 193,  84),
    ("yaprak", 161.4,  40.3, 11.97,  86, 194,  83),
    ("yaprak", 199.6,  40.4, 11.74,  88, 194,  85),
    ("yaprak",  84.0,  59.5,  9.33,  86, 194,  83),
    ("yaprak", 181.3,  61.0,  9.10,  86, 194,  83),
    ("yaprak", 132.7,  65.1, 14.83,  89, 195,  85),
    ("yaprak", 159.4,  68.5,  7.66,  87, 194,  83),
    ("yaprak", 105.0,  69.3,  7.05,  85, 195,  83),
]


def render_png(path):
    xy = np.array([[d[1], d[2]] for d in DOTS])
    rad = np.array([d[3] for d in DOTS]) * R_SCALE
    col = np.array([[d[4], d[5], d[6]] for d in DOTS]) / 255.0

    dpi = int(100 * OUT_SCALE * SUPERSAMPLE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter(xy[:, 0], xy[:, 1], s=np.pi * (rad * 0.72) ** 2,
               c=col, edgecolors="none")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)

    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    im = im.resize((W * OUT_SCALE, H * OUT_SCALE), Image.LANCZOS)
    im = im.crop(im.getchannel("A").getbbox())
    side = max(im.size) + 2 * MARGIN * OUT_SCALE
    canvas = Image.new("RGBA", (side, side))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2), im)
    canvas.save(path)
    return side


def write_svg(path):
    groups = {"gövde": [], "yaprak": []}
    for part, x, y, r, R, G, B in DOTS:
        groups[part].append(
            f'    <circle cx="{x}" cy="{y}" r="{r * R_SCALE:.2f}" '
            f'fill="#{R:02x}{G:02x}{B:02x}"/>')
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    for part, circles in groups.items():
        lines.append(f'  <g id="{part}">')
        lines.extend(circles)
        lines.append("  </g>")
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    side = render_png("strawveri_icon.png")
    write_svg("strawveri_icon.svg")
    with open("strawveri_icon_dots.csv", "w") as f:
        f.write("parca,x,y,yaricap,renk_r,renk_g,renk_b\n")
        for d in DOTS:
            f.write(",".join(str(v) for v in d) + "\n")
    print(f"{len(DOTS)} nokta -> strawveri_icon.png "
          f"({side}x{side}), .svg, _dots.csv")


if __name__ == "__main__":
    main()
