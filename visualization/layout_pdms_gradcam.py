"""Lay out existing PDMS-DARTS Grad-CAM outputs as a 2x3 paper figure.

The red rectangles are intentionally manual, visual discriminative-region
markers.  They are not ROI, lesion, detection, or segmentation annotations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.gradcam_utils import resize_rgb


DATASETS = ("PathMNIST", "LC25000", "CRC5000")
# Coordinates are in each saved original image's native pixel coordinate system.
# They mark visually distinct tissue structures for illustration only.
VISUAL_REGION_BOXES = {
    "PathMNIST": [(9, 2, 14, 12), (17, 8, 24, 16), (3, 16, 10, 23)],
    "LC25000": [(15, 28, 65, 95), (104, 13, 150, 65), (165, 47, 211, 105),
                (91, 126, 157, 185), (40, 159, 92, 214)],
    "CRC5000": [(6, 6, 31, 30), (48, 16, 72, 39), (94, 10, 121, 37),
                (112, 53, 143, 78), (18, 80, 46, 110), (62, 93, 91, 124)],
}


def font(size):
    for candidate in (
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def with_visual_boxes(image, boxes, native_size=False):
    """Draw manual illustrative boxes without asserting any ground truth ROI."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    thickness = 1 if native_size else 3
    for box in boxes:
        draw.rectangle(box, outline=(220, 0, 0), width=thickness)
    return marked


def text_center(draw, xy, text, image_font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=image_font)
    draw.text((xy[0] - (right - left) // 2, xy[1] - (bottom - top) // 2), text, fill="black", font=image_font)


def make_figure(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell, gap, label_width, header_height = 420, 16, 205, 104
    canvas = Image.new(
        "RGB", (label_width + 3 * cell + 2 * gap, header_height + 2 * cell + gap), "white"
    )
    draw = ImageDraw.Draw(canvas)
    header_font, row_font = font(29), font(27)

    for column, dataset in enumerate(DATASETS):
        text_center(draw, (label_width + column * (cell + gap) + cell // 2, 44), dataset, header_font)
    text_center(draw, (label_width // 2, header_height + cell // 2), "Original", row_font)
    text_center(draw, (label_width // 2, header_height + cell + gap + cell // 2), "PDMS-DARTS", row_font)

    for column, dataset in enumerate(DATASETS):
        dataset_dir = input_dir / dataset
        original_path = dataset_dir / "original.png"
        cam_path = dataset_dir / "PDMS_DARTS_gradcam.png"
        if not original_path.is_file() or not cam_path.is_file():
            raise FileNotFoundError("Missing original or Grad-CAM output in {}".format(dataset_dir))
        original = Image.open(original_path).convert("RGB")
        marked_native = with_visual_boxes(original, VISUAL_REGION_BOXES[dataset], native_size=True)
        marked_native.save(dataset_dir / "original_visual_regions.png")
        # Draw after enlargement so every dataset has the same thin red-box
        # style as the reference paper, including 28x28 PathMNIST images.
        enlarged_original = Image.fromarray(resize_rgb(np.asarray(original), (cell, cell)))
        scale_x, scale_y = cell / original.width, cell / original.height
        scaled_boxes = [tuple(int(round(value * (scale_x if offset % 2 == 0 else scale_y)))
                              for offset, value in enumerate(box))
                        for box in VISUAL_REGION_BOXES[dataset]]
        marked_original = with_visual_boxes(enlarged_original, scaled_boxes)
        cam = Image.open(cam_path).convert("RGB")
        x = label_width + column * (cell + gap)
        canvas.paste(marked_original, (x, header_height))
        canvas.paste(Image.fromarray(resize_rgb(np.asarray(cam), (cell, cell))), (x, header_height + cell + gap))

    stem = "GradCAM_PDMS_DARTS_2x3_visual_regions"
    png300, png600, pdf = (output_dir / (stem + suffix) for suffix in ("_300dpi.png", "_600dpi.png", ".pdf"))
    canvas.save(png300, dpi=(300, 300))
    canvas.save(png600, dpi=(600, 600))
    canvas.save(pdf, "PDF", resolution=600.0)
    return png300, png600, pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the PDMS-DARTS 2x3 Grad-CAM layout with manual visual boxes.")
    parser.add_argument("--input_dir", default="gradcam_results/PDMS_DARTS")
    parser.add_argument("--output_dir", default="gradcam_results/PDMS_DARTS")
    args = parser.parse_args()
    for path in make_figure(args.input_dir, args.output_dir):
        print(path)
