#!/usr/bin/env python3
"""
make_figure3.py — Composite summary figure from three existing plots.

Panel A: valid-enriched rates per model (stats/valid_rate_by_model.png)
Panel B: FDR overlap per model        (stats/fdr_overlap.png)
Panel C: hallmark recovery per model  (stats/hallmark_truth.png)

Writes: manuscript/figures/figure3_summary_panels.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

STATS = Path("stats")
OUT = Path("manuscript/figures/figure3_summary_panels.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("A", STATS / "valid_rate_by_model.png",
     "Valid-enriched rate per model (mean ± SD across runs)"),
    ("B", STATS / "fdr_overlap.png",
     "Mean overlap with FDR top-10 union per model"),
    ("C", STATS / "hallmark_truth.png",
     "MSigDB Hallmark direct recovery per model"),
]


def load_panel(path, target_width):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = target_width / w
    new_size = (target_width, int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def main():
    missing = [p for _, p, _ in PANELS if not p.exists()]
    if missing:
        for m in missing:
            print(f"[MISSING] {m}")
        raise SystemExit("Missing input panels. Run stats.py first.")

    target_width = 1400
    panels = [load_panel(p, target_width) for _, p, _ in PANELS]

    # Header space for panel labels
    header_h = 60
    padding = 20
    total_h = sum(p.size[1] + header_h + padding for p in panels)

    canvas = Image.new("RGB", (target_width, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    y = 0
    for (label, _, caption), img in zip(PANELS, panels):
        draw.text((padding, y + 15), f"Panel {label}", fill="#1e40af", font=font)
        y += header_h
        canvas.paste(img, (0, y))
        y += img.size[1] + padding

    canvas.save(OUT, dpi=(200, 200))
    print(f"[INFO] wrote {OUT} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
