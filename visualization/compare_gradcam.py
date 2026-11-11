"""Create a fair DARTS-versus-PDMS-DARTS Grad-CAM comparison figure.

This script only accepts final discrete-model checkpoints.  It deliberately
rejects search-supernet ``weights.pt`` files and refuses to use random weights.
With no explicit checkpoint arguments it searches the current project for
metadata-bearing final checkpoints and reports every missing required model.
"""

from __future__ import print_function

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataSets import CRC5000, LC25000, DataSet  # noqa: E402
from model.net_cell import MSTF_NAS_NETWORK  # noqa: E402
from visualization.gradcam_utils import GradCAM, overlay_cam, resize_rgb, tensor_to_display_image  # noqa: E402


DATASETS = ("PathMNIST", "LC25000", "CRC5000")
MODEL_NAMES = ("DARTS", "PDMS-DARTS")
CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)
CRC_MEAN = (0.5, 0.5, 0.5)
CRC_STD = (0.5, 0.5, 0.5)


def _font(size):
    candidates = (
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def _checkpoint_metadata(path):
    """Read only lightweight checkpoint metadata needed for compatibility."""
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}, _checkpoint_state_dict(checkpoint)
    state = _checkpoint_state_dict(checkpoint)
    metadata = {key: value for key, value in checkpoint.items() if key not in ("model_state_dict", "state_dict", "model")}
    return metadata, state


def _contains_dids(state):
    # Pooling branches have no state tensors, so their names do not appear in
    # a checkpoint.  The scale/type gate parameters and the shared projection
    # are the reliable serialized DIDS signature.
    return any(
        ".downsample.avg_scale_gate." in key
        or ".downsample.max_scale_gate." in key
        or ".downsample.type_gate." in key
        for key in state
    )


def _normal_space_from_state(state, fallback):
    """The LC transfer code enforces PDC+DIDS; metadata may predate labels."""
    if fallback:
        return fallback
    if _contains_dids(state):
        return "pdc"
    return None


def _downsample_from_state(state, fallback):
    if fallback:
        return fallback
    if _contains_dids(state):
        return "dids"
    return None


def discover_checkpoints(root):
    """Discover metadata-bearing *final* checkpoints; never returns supernets."""
    discovered = {dataset: {name: [] for name in MODEL_NAMES} for dataset in DATASETS}
    suffixes = {".pt", ".pth", ".ckpt", ".pkl"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        # Search-supernet files are not final retrained discrete models.
        if path.name == "weights.pt" or any(part.startswith("search-") for part in path.parts):
            continue
        try:
            metadata, state = _checkpoint_metadata(path)
        except Exception:
            continue
        dataset = metadata.get("dataset")
        path_lower = str(path).lower()
        if dataset not in DATASETS:
            if "lc25000" in path_lower:
                dataset = "LC25000"
            else:
                continue
        normal_space = _normal_space_from_state(state, metadata.get("normal_space"))
        downsample = _downsample_from_state(state, metadata.get("downsample"))
        if normal_space == "normal" and downsample == "reduction":
            label = "DARTS"
        elif normal_space == "pdc" and downsample == "dids":
            label = "PDMS-DARTS"
        else:
            continue
        discovered[dataset][label].append(path.resolve())
    return discovered


def resolve_checkpoints(args):
    discovered = discover_checkpoints(REPO_ROOT)
    result = {}
    missing = []
    for dataset in DATASETS:
        explicit = {
            "DARTS": getattr(args, "{}_darts_checkpoint".format(dataset.lower().replace("-", ""))),
            "PDMS-DARTS": getattr(args, "{}_pdms_checkpoint".format(dataset.lower().replace("-", ""))),
        }
        result[dataset] = {}
        for model_name in MODEL_NAMES:
            value = explicit[model_name]
            if value:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    missing.append("{} {} checkpoint does not exist: {}".format(dataset, model_name, path))
                    continue
            else:
                candidates = discovered[dataset][model_name]
                path = candidates[-1] if candidates else None
            if path is None:
                missing.append("{} {} final discrete checkpoint is missing".format(dataset, model_name))
            else:
                result[dataset][model_name] = path
    if missing:
        lines = ["Grad-CAM was not run because required final checkpoints are missing:"]
        lines.extend("- " + item for item in missing)
        lines.append("Search-supernet weights.pt files were intentionally excluded.")
        raise FileNotFoundError("\n".join(lines))
    return result


def dataset_bundle(dataset, crc_seed):
    if dataset == "PathMNIST":
        _, test_data, num_classes = DataSet[dataset](size=28)
        return test_data, num_classes, CIFAR_MEAN, CIFAR_STD
    if dataset == "LC25000":
        _, _, test_data, num_classes, _ = LC25000(size=224)
        return test_data, num_classes, CIFAR_MEAN, CIFAR_STD
    if dataset == "CRC5000":
        _, test_data, num_classes = CRC5000(size=150, purpose="train", seed=crc_seed)
        return test_data, num_classes, CRC_MEAN, CRC_STD
    raise ValueError("Unsupported dataset: {}".format(dataset))


def build_model(path, label, num_classes, device):
    metadata, state = _checkpoint_metadata(path)
    genotype = metadata.get("genotype")
    if not isinstance(genotype, (list, tuple)):
        raise ValueError("{} does not record its discrete genotype.".format(path))
    normal_space = metadata.get("normal_space")
    downsample = metadata.get("downsample")
    if label == "PDMS-DARTS":
        normal_space = normal_space or "pdc"
        downsample = downsample or "dids"
    elif label == "DARTS":
        normal_space = normal_space or "normal"
        downsample = downsample or "reduction"
    init_channels = int(metadata.get("init_channels", 128))
    num_blocks = int(metadata.get("num_blocks", 3))
    model = MSTF_NAS_NETWORK(
        num_class=num_classes,
        genotype=list(genotype),
        n_n=5,
        num_block=num_blocks,
        init_c=init_channels,
        normal_space=normal_space,
        downsample=downsample,
        n_r=4,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch for {}. Missing: {}; unexpected: {}".format(
                path, missing, unexpected
            )
        )
    model.eval()
    return model, {"normal_space": normal_space, "downsample": downsample, "metadata": metadata}


def predict(model, image, device):
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = F.softmax(logits, dim=1)[0]
        probability, prediction = probabilities.max(dim=0)
    return int(prediction.item()), float(probability.item())


def select_sample(test_data, darts, pdms, device, mean, std):
    """Deterministically choose a jointly-correct, non-flat test image.

    Ranking uses prediction confidence and image variance only; CAMs are never
    calculated during sample selection.
    """
    candidates = []
    for index in range(len(test_data)):
        image, label = test_data[index]
        label = int(np.asarray(label).reshape(-1)[0])
        darts_pred, darts_prob = predict(darts, image, device)
        pdms_pred, pdms_prob = predict(pdms, image, device)
        if darts_pred != label or pdms_pred != label:
            continue
        display = tensor_to_display_image(image, mean, std)
        texture = float(display.astype(np.float32).std() / 255.0)
        # Fixed rule: confidence dominates; texture only resolves whether a
        # confidently classified but visually near-uniform tile is preferred.
        score = 0.9 * (darts_prob + pdms_prob) / 2.0 + 0.1 * texture
        candidates.append((score, -index, image, label, darts_pred, darts_prob, pdms_pred, pdms_prob))
    if not candidates:
        raise RuntimeError("No jointly correct test sample exists for the two requested models.")
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, negative_index, image, label, darts_pred, darts_prob, pdms_pred, pdms_prob = candidates[0]
    return {
        "index": -negative_index,
        "image": image,
        "label": label,
        "darts_prediction": darts_pred,
        "darts_probability": darts_prob,
        "pdms_prediction": pdms_pred,
        "pdms_probability": pdms_prob,
    }


def last_spatial_layer(model):
    if not hasattr(model, "main_stream") or len(model.main_stream) == 0:
        raise RuntimeError("Model has no main_stream blocks for target-layer selection.")
    index = len(model.main_stream) - 1
    layer = model.main_stream[index]
    return layer, "main_stream.{} (final Block output before global_pool)".format(index)


def save_rgb(path, image):
    Image.fromarray(image, mode="RGB").save(str(path))


def compose_figure(rows, output_dir):
    cell = 420
    label_width = 205
    header_height = 88
    canvas = Image.new("RGB", (label_width + 3 * cell, header_height + 3 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(31)
    row_font = _font(30)
    columns = ("Original", "DARTS", "PDMS-DARTS")
    for column, name in enumerate(columns):
        box_left = label_width + column * cell
        text_box = draw.textbbox((0, 0), name, font=title_font)
        draw.text((box_left + (cell - (text_box[2] - text_box[0])) // 2, 25), name, fill="black", font=title_font)
    for row_index, row in enumerate(rows):
        y = header_height + row_index * cell
        label = row["dataset"]
        text_box = draw.textbbox((0, 0), label, font=row_font)
        draw.text(((label_width - (text_box[2] - text_box[0])) // 2, y + (cell - (text_box[3] - text_box[1])) // 2), label, fill="black", font=row_font)
        images = (row["original"], row["darts"], row["pdms"])
        for column, image in enumerate(images):
            rendered = resize_rgb(image, (cell, cell))
            canvas.paste(Image.fromarray(rendered), (label_width + column * cell, y))
    png300 = output_dir / "GradCAM_DARTS_vs_PDMS_DARTS_300dpi.png"
    png600 = output_dir / "GradCAM_DARTS_vs_PDMS_DARTS_600dpi.png"
    pdf = output_dir / "GradCAM_DARTS_vs_PDMS_DARTS.pdf"
    canvas.save(str(png300), dpi=(300, 300))
    canvas.save(str(png600), dpi=(600, 600))
    canvas.save(str(pdf), "PDF", resolution=600.0)
    return png300, png600, pdf


def run(args):
    random.seed(args.selection_seed)
    np.random.seed(args.selection_seed)
    torch.manual_seed(args.selection_seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError("Invalid GPU {}.".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    checkpoints = resolve_checkpoints(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    figure_rows = []

    for dataset in DATASETS:
        test_data, num_classes, mean, std = dataset_bundle(dataset, args.crc_seed)
        darts_path = checkpoints[dataset]["DARTS"]
        pdms_path = checkpoints[dataset]["PDMS-DARTS"]
        darts, darts_info = build_model(darts_path, "DARTS", num_classes, device)
        pdms, pdms_info = build_model(pdms_path, "PDMS-DARTS", num_classes, device)
        darts_layer, darts_layer_name = last_spatial_layer(darts)
        pdms_layer, pdms_layer_name = last_spatial_layer(pdms)
        print("{} DARTS checkpoint = {}".format(dataset, darts_path))
        print("{} PDMS-DARTS checkpoint = {}".format(dataset, pdms_path))
        print("DARTS target layer = {}".format(darts_layer_name))
        print("PDMS-DARTS target layer = {}".format(pdms_layer_name))

        selected = select_sample(test_data, darts, pdms, device, mean, std)
        image = selected["image"]
        original = tensor_to_display_image(image, mean, std)
        target_class = int(selected["label"])
        with GradCAM(darts, darts_layer) as darts_cam_calculator:
            darts_cam = darts_cam_calculator(image.unsqueeze(0).to(device), target_class, original.shape[:2])
        with GradCAM(pdms, pdms_layer) as pdms_cam_calculator:
            pdms_cam = pdms_cam_calculator(image.unsqueeze(0).to(device), target_class, original.shape[:2])
        darts_overlay = overlay_cam(original, darts_cam, alpha=args.alpha)
        pdms_overlay = overlay_cam(original, pdms_cam, alpha=args.alpha)
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(dataset_dir / "original.png", original)
        save_rgb(dataset_dir / "DARTS_gradcam.png", darts_overlay)
        save_rgb(dataset_dir / "PDMS_DARTS_gradcam.png", pdms_overlay)
        np.save(str(dataset_dir / "DARTS_cam.npy"), darts_cam)
        np.save(str(dataset_dir / "PDMS_DARTS_cam.npy"), pdms_cam)
        # Read-back confirms the raw arrays can be used to reproduce overlays.
        if not np.allclose(np.load(str(dataset_dir / "DARTS_cam.npy")), darts_cam):
            raise RuntimeError("DARTS CAM read-back check failed for {}.".format(dataset))
        if not np.allclose(np.load(str(dataset_dir / "PDMS_DARTS_cam.npy")), pdms_cam):
            raise RuntimeError("PDMS-DARTS CAM read-back check failed for {}.".format(dataset))

        sample_rows.append({
            "dataset": dataset,
            "sample_index": selected["index"],
            "ground_truth_label": target_class,
            "DARTS_prediction": selected["darts_prediction"],
            "DARTS_probability": "{:.8f}".format(selected["darts_probability"]),
            "PDMS_DARTS_prediction": selected["pdms_prediction"],
            "PDMS_DARTS_probability": "{:.8f}".format(selected["pdms_probability"]),
            "DARTS_checkpoint": str(darts_path),
            "PDMS_DARTS_checkpoint": str(pdms_path),
            "DARTS_target_layer": darts_layer_name,
            "PDMS_DARTS_target_layer": pdms_layer_name,
            "target_class": target_class,
            "cmap": "jet",
            "alpha": args.alpha,
        })
        figure_rows.append({"dataset": dataset, "original": original, "darts": darts_overlay, "pdms": pdms_overlay})
        del darts, pdms
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = list(sample_rows[0].keys())
    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample_rows)
    png300, png600, pdf = compose_figure(figure_rows, output_dir)
    print("Saved 3x3 Grad-CAM comparison: {}".format(png600))
    print("Saved PDF: {}".format(pdf))


def parse_args():
    parser = argparse.ArgumentParser(description="Fair DARTS versus PDMS-DARTS Grad-CAM comparison.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--output_dir", default="gradcam_results")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--selection_seed", type=int, default=0)
    parser.add_argument("--crc_seed", type=int, default=0)
    for dataset in DATASETS:
        prefix = dataset.lower().replace("-", "")
        parser.add_argument("--{}_darts_checkpoint".format(prefix))
        parser.add_argument("--{}_pdms_checkpoint".format(prefix))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if tuple(arguments.datasets) != DATASETS or tuple(arguments.models) != MODEL_NAMES:
        raise SystemExit("This paper figure is intentionally fixed to all three datasets and both models.")
    if not 0.0 <= arguments.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0,1].")
    run(arguments)
