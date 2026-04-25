from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

ICON_SIZE = 1024
FAVICON_SIZE = 256
SPLASH_SIZE = 1024
ADAPTIVE_SIZE = 1024

BG_TOP = (7, 10, 24)
BG_BOTTOM = (23, 36, 84)
INDIGO = (92, 107, 255)
CYAN = (72, 216, 255)
VIOLET = (144, 96, 255)
LILAC = (218, 227, 255)
WHITE = (245, 249, 255)


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        lerp(c1[0], c2[0], t),
        lerp(c1[1], c2[1], t),
        lerp(c1[2], c2[2], t),
    )


def vertical_gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    px = img.load()
    for y in range(size):
        color = blend(top, bottom, y / max(size - 1, 1))
        for x in range(size):
            px[x, y] = (*color, 255)
    return img


def draw_glow(base: Image.Image, center: tuple[float, float], radius: float, color: tuple[int, int, int], alpha: int) -> None:
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    x, y = center
    d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=radius * 0.18))
    base.alpha_composite(glow)


def create_canvas(size: int) -> Image.Image:
    canvas = vertical_gradient(size, BG_TOP, BG_BOTTOM)
    draw_glow(canvas, (size * 0.5, size * 0.42), size * 0.28, INDIGO, 130)
    draw_glow(canvas, (size * 0.68, size * 0.68), size * 0.18, CYAN, 86)
    draw_glow(canvas, (size * 0.26, size * 0.74), size * 0.14, VIOLET, 72)
    return canvas


def add_mesh(base: Image.Image) -> None:
    size = base.size[0]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    step = size // 12
    for offset in range(-size // 2, size, step):
        d.line(
            ((offset, 0), (offset + size * 0.45, size)),
            fill=(255, 255, 255, 16),
            width=max(1, size // 256),
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=size * 0.004))
    base.alpha_composite(overlay)


def add_ring(base: Image.Image) -> None:
    size = base.size[0]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    ring_box = (
        size * 0.17,
        size * 0.17,
        size * 0.83,
        size * 0.83,
    )
    d.arc(ring_box, start=18, end=342, fill=(*LILAC, 175), width=int(size * 0.032))
    d.arc(
        (
            size * 0.22,
            size * 0.22,
            size * 0.78,
            size * 0.78,
        ),
        start=210,
        end=26,
        fill=(*CYAN, 188),
        width=int(size * 0.016),
    )

    cx = cy = size * 0.5
    orbit_r = size * 0.33
    node_r = size * 0.024
    points = []
    for angle in (-46, -4, 36, 82, 132, 196, 248, 308):
        rad = math.radians(angle)
        x = cx + orbit_r * math.cos(rad)
        y = cy + orbit_r * math.sin(rad)
        points.append((x, y))

    links = ((0, 1), (1, 2), (3, 4), (5, 6), (6, 7))
    for a, b in links:
        d.line((points[a], points[b]), fill=(*WHITE, 120), width=int(size * 0.008))

    for idx, (x, y) in enumerate(points):
        fill = CYAN if idx % 2 == 0 else LILAC
        d.ellipse((x - node_r, y - node_r, x + node_r, y + node_r), fill=(*fill, 255))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=size * 0.002))
    base.alpha_composite(overlay)


def add_core_mark(base: Image.Image) -> None:
    size = base.size[0]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    mark_box = (
        size * 0.28,
        size * 0.28,
        size * 0.72,
        size * 0.72,
    )
    inner_box = (
        size * 0.34,
        size * 0.34,
        size * 0.66,
        size * 0.66,
    )
    stroke = int(size * 0.092)

    d.arc(mark_box, start=48, end=320, fill=(*WHITE, 255), width=stroke)
    d.arc(inner_box, start=56, end=325, fill=(*CYAN, 255), width=int(size * 0.03))

    d.line(
        ((size * 0.59, size * 0.36), (size * 0.59, size * 0.62)),
        fill=(*WHITE, 255),
        width=int(size * 0.034),
    )
    d.line(
        ((size * 0.59, size * 0.62), (size * 0.68, size * 0.69)),
        fill=(*WHITE, 220),
        width=int(size * 0.03),
    )

    d.ellipse(
        (size * 0.445, size * 0.445, size * 0.555, size * 0.555),
        fill=(*INDIGO, 230),
    )
    d.ellipse(
        (size * 0.47, size * 0.47, size * 0.53, size * 0.53),
        fill=(*WHITE, 240),
    )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=size * 0.0015))
    base.alpha_composite(overlay)


def add_shell_highlights(base: Image.Image) -> None:
    size = base.size[0]
    gloss = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(gloss)
    d.rounded_rectangle(
        (size * 0.08, size * 0.08, size * 0.92, size * 0.92),
        radius=size * 0.24,
        outline=(255, 255, 255, 24),
        width=int(size * 0.008),
    )
    d.arc(
        (size * 0.18, size * 0.08, size * 0.82, size * 0.56),
        start=200,
        end=332,
        fill=(255, 255, 255, 36),
        width=int(size * 0.014),
    )
    gloss = gloss.filter(ImageFilter.GaussianBlur(radius=size * 0.004))
    base.alpha_composite(gloss)


def rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size, size), radius=radius, fill=255)
    return mask


def render_icon(size: int, corner_radius: float) -> Image.Image:
    canvas = create_canvas(size)
    add_mesh(canvas)
    add_ring(canvas)
    add_core_mark(canvas)
    add_shell_highlights(canvas)
    canvas.putalpha(rounded_mask(size, corner_radius))
    return canvas


def render_adaptive_foreground(size: int) -> Image.Image:
    fg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    halo = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d_halo = ImageDraw.Draw(halo)
    cx = cy = size * 0.5
    r = size * 0.26
    d_halo.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*INDIGO, 156))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=size * 0.06))
    fg.alpha_composite(halo)

    d = ImageDraw.Draw(fg)
    d.arc((size * 0.22, size * 0.22, size * 0.78, size * 0.78), start=26, end=334, fill=(*WHITE, 255), width=int(size * 0.086))
    d.arc((size * 0.285, size * 0.285, size * 0.715, size * 0.715), start=42, end=330, fill=(*CYAN, 255), width=int(size * 0.028))
    d.line(((size * 0.605, size * 0.315), (size * 0.605, size * 0.59)), fill=(*WHITE, 255), width=int(size * 0.03))
    d.line(((size * 0.605, size * 0.59), (size * 0.69, size * 0.655)), fill=(*WHITE, 230), width=int(size * 0.028))

    orbit_r = size * 0.32
    node_r = size * 0.022
    for angle in (-40, 40, 124, 222, 304):
        rad = math.radians(angle)
        x = cx + orbit_r * math.cos(rad)
        y = cy + orbit_r * math.sin(rad)
        d.ellipse((x - node_r, y - node_r, x + node_r, y + node_r), fill=(*LILAC, 248))

    return fg


def save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    icon = render_icon(ICON_SIZE, corner_radius=ICON_SIZE * 0.23)
    splash = render_icon(SPLASH_SIZE, corner_radius=SPLASH_SIZE * 0.18)
    favicon = render_icon(FAVICON_SIZE, corner_radius=FAVICON_SIZE * 0.22)
    adaptive = render_adaptive_foreground(ADAPTIVE_SIZE)

    save_png(icon, ASSETS / "icon.png")
    save_png(splash, ASSETS / "splash-icon.png")
    save_png(favicon, ASSETS / "favicon.png")
    save_png(adaptive, ASSETS / "adaptive-icon.png")

    print("Generated Cortex Lab icon assets:")
    for name in ("icon.png", "splash-icon.png", "favicon.png", "adaptive-icon.png"):
        print(f" - {ASSETS / name}")


if __name__ == "__main__":
    main()
