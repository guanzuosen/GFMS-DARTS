"""Generate real Grad-CAM figures for the project's PDMS-DARTS checkpoints.

PDMS-DARTS denotes the implemented ``normal_space=pdc`` plus fixed ``dids``
configuration.  This script never loads a DARTS model or a search supernet.
For each dataset it deterministically selects the first correctly predicted
test item, then uses that item's ground-truth class as the Grad-CAM target.
"""

from __future__ import print_function

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.compare_gradcam import (  # noqa: E402
    DATASETS,
    _font,
    build_model,
    dataset_bundle,
    last_spatial_layer,
    predict,
)
from visualization.gradcam_utils import GradCAM, overlay_cam, resize_rgb, tensor_to_display_image  # noqa: E402


def select_first_correct(test_data, model, device):
    """Use a reproducible non-CAM selection rule: lowest correct test index."""
    for index in range(len(test_data)):
        image, label = test_data[index]
        label = int(np.asarray(label).reshape(-1)[0])
        prediction, probability = predict(model, image, device)
        if prediction == label:
            return index, image, label, prediction, probability
    raise RuntimeError("No correctly predicted test sample exists for PDMS-DARTS.")


def save_rgb(path, image):
    Image.fromarray(image, mode="RGB").save(str(path))


def compose_figure(rows, output_dir):
    cell = 420
    label_width = 205
    header_height = 88
    canvas = Image.new("RGB", (label_width + 2 * cell, header_height + 3 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(31)
    row_font = _font(30)
    for column, name in enumerate(("Original", "PDMS-DARTS")):
        x = label_width + column * cell
        box = draw.textbbox((0, 0), name, font=title_font)
        draw.text((x + (cell - (box[2] - box[0])) // 2, 25), name, fill="black", font=title_font)
    for row_index, row in enumerate(rows):
        y = header_height + row_index * cell
        box = draw.textbbox((0, 0), row["dataset"], font=row_font)
        draw.text(
            ((label_width - (box[2] - box[0])) // 2, y + (cell - (box[3] - box[1])) // 2),
            row["dataset"], fill="black", font=row_font,
        )
        for column, image in enumerate((row["original"], row["pdms"])):
            canvas.paste(Image.fromarray(resize_rgb(image, (cell, cell))), (label_width + column * cell, y))
    png300 = output_dir / "GradCAM_PDMS_DARTS_300dpi.png"
    png600 = output_dir / "GradCAM_PDMS_DARTS_600dpi.png"
    pdf = output_dir / "GradCAM_PDMS_DARTS.pdf"
    canvas.save(str(png300), dpi=(300, 300))
    canvas.save(str(png600), dpi=(600, 600))
    canvas.save(str(pdf), "PDF", resolution=600.0)
    return png300, png600, pdf


def run(args):
    if torch.cuda.is_available() and args.gpu >= 0:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError("Invalid GPU {}.".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    checkpoint_args = {
        "PathMNIST": args.pathmnist_checkpoint,
        "LC25000": args.lc25000_checkpoint,
        "CRC5000": args.crc5000_checkpoint,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, records = [], []

    for dataset in DATASETS:
        checkpoint_path = Path(checkpoint_args[dataset]).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError("{} checkpoint does not exist: {}".format(dataset, checkpoint_path))
        test_data, num_classes, mean, std = dataset_bundle(dataset, args.crc_seed)
        model, _ = build_model(checkpoint_path, "PDMS-DARTS", num_classes, device)
        target_layer, target_layer_name = last_spatial_layer(model)
        index, image, label, prediction, probability = select_first_correct(test_data, model, device)
        original = tensor_to_display_image(image, mean, std)
        with GradCAM(model, target_layer) as calculator:
            cam = calculator(image.unsqueeze(0).to(device), label, original.shape[:2])
        heatmap = overlay_cam(original, cam, alpha=args.alpha)

        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(dataset_dir / "original.png", original)
        save_rgb(dataset_dir / "PDMS_DARTS_gradcam.png", heatmap)
        np.save(str(dataset_dir / "PDMS_DARTS_cam.npy"), cam)
        if not np.allclose(np.load(str(dataset_dir / "PDMS_DARTS_cam.npy")), cam):
            raise RuntimeError("CAM read-back check failed for {}.".format(dataset))

        record = {
            "dataset": dataset,
            "sample_index": index,
            "ground_truth_label": label,
            "PDMS_DARTS_prediction": prediction,
            "PDMS_DARTS_probability": "{:.8f}".format(probability),
            "checkpoint": str(checkpoint_path),
            "target_layer": target_layer_name,
            "target_class": label,
            "cmap": "jet",
            "alpha": args.alpha,
        }
        records.append(record)
        rows.append({"dataset": dataset, "original": original, "pdms": heatmap})
        print("{}: sample={}, label={}, prediction={}, target_layer={}".format(
            dataset, index, label, prediction, target_layer_name,
        ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    png300, png600, pdf = compose_figure(rows, output_dir)
    print("Saved 300 dpi PNG: {}".format(png300))
    print("Saved 600 dpi PNG: {}".format(png600))
    print("Saved PDF: {}".format(pdf))


def parse_args():
    parser = argparse.ArgumentParser(description="PDMS-DARTS-only Grad-CAM for PathMNIST, LC25000 and CRC5000.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--crc_seed", type=int, default=0)
    parser.add_argument("--output_dir", default="gradcam_results/PDMS_DARTS")
    parser.add_argument("--pathmnist_checkpoint", required=True)
    parser.add_argument("--lc25000_checkpoint", required=True)
    parser.add_argument("--crc5000_checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not 0.0 <= arguments.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1].")
    run(arguments)
