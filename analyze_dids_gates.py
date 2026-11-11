"""Post-hoc DIDS dynamic-gate analysis for an already trained discrete model.

This script never calls ``Train.py`` or ``Search.py``.  It loads a supplied
checkpoint, performs exactly one pass over the test split, and records the
Softmax gate weights produced by every DIDS stage independently.

Examples
--------
Collect one trained model::

    python analyze_dids_gates.py \
        --dataset PathMNIST --gpu 1 \
        --checkpoint /path/to/pathmnist_final.pt \
        --genotype_file /path/to/genotype.txt \
        --output_dir dids_gate_analysis/PathMNIST

Create one representative-dataset heatmap after collecting four datasets::

    python analyze_dids_gates.py --plot_dirs \
        dids_gate_analysis/PathMNIST \
        dids_gate_analysis/PneumoniaMNIST \
        dids_gate_analysis/RetinaMNIST \
        dids_gate_analysis/BreastMNIST \
        --plot_output dids_gate_analysis/representative_dids_gates
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont

from DataSets import DataSet
from model.net_cell import MSTF_NAS_NETWORK
from model.operators import DIDS


GATE_COLUMNS = (
    "lambda_avg_2", "lambda_avg_3", "lambda_avg_4",
    "lambda_max_2", "lambda_max_3", "lambda_max_4",
    "gamma_avg", "gamma_max",
)
SCALE_LABELS = ("Avg-2", "Avg-3", "Avg-4", "Max-2", "Max-3", "Max-4")
TYPE_LABELS = ("Avg", "Max")


def parse_genotype(raw: str | None, genotype_file: str | None, checkpoint: object,
                   n_nodes: int, num_blocks: int) -> List[int]:
    """Read one-Block or full-network genotype and expand one-Block input."""
    if raw is not None and genotype_file is not None:
        raise ValueError("Use only one of --genotype and --genotype_file.")
    if genotype_file is not None:
        raw = Path(genotype_file).read_text(encoding="utf-8").strip()
    if raw is None and isinstance(checkpoint, dict) and "genotype" in checkpoint:
        raw = checkpoint["genotype"]
    if raw is None:
        raise ValueError("Provide --genotype/--genotype_file, or use a checkpoint containing 'genotype'.")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Genotype must be a Python-style list of integers.") from exc
    if not isinstance(raw, (list, tuple)) or not all(isinstance(value, int) for value in raw):
        raise ValueError("Genotype must be a list of integers.")

    one_block_length = 2 * n_nodes
    values = list(raw)
    if len(values) == one_block_length:
        return values * num_blocks
    if len(values) == one_block_length * num_blocks:
        return values
    raise ValueError(
        f"Expected {one_block_length} values for one DIDS Block or "
        f"{one_block_length * num_blocks} values for the full model, received {len(values)}."
    )


def checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Accept raw state_dict or common checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Checkpoint does not contain a valid model state_dict.")
    state = checkpoint
    if any(key.startswith("module.") for key in state):
        state = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }
    return state


def named_dids_modules(model: torch.nn.Module) -> List[Tuple[str, DIDS]]:
    stages = [(name, module) for name, module in model.named_modules() if isinstance(module, DIDS)]
    if not stages:
        raise RuntimeError("The loaded model contains no DIDS module; use a --downsample dids checkpoint.")
    return stages


def validate_gate_weights(gates: Dict[str, torch.Tensor], expected_batch: int, stage_name: str,
                          atol: float = 1e-5) -> Dict[str, float]:
    """Validate shapes and Softmax sums for one DIDS stage/batch."""
    expected_widths = {"avg_scale": 3, "max_scale": 3, "type": 2}
    errors = {}
    for name, width in expected_widths.items():
        values = gates.get(name)
        if values is None or tuple(values.shape) != (expected_batch, width):
            received = None if values is None else tuple(values.shape)
            raise RuntimeError(
                f"{stage_name}: expected {name} shape {(expected_batch, width)}, received {received}."
            )
        maximum_error = float((values.sum(dim=1) - 1.0).abs().max().item())
        if maximum_error > atol:
            raise RuntimeError(
                f"{stage_name}: {name} Softmax sums are invalid; max absolute error={maximum_error:.3e}."
            )
        errors[name] = maximum_error
    return errors


def metric_row(values: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(GATE_COLUMNS, values.tolist())}


def write_stage_outputs(output_dir: Path, stage_index: int, stage_name: str,
                        sample_indices: np.ndarray, labels: np.ndarray,
                        avg_scale: np.ndarray, max_scale: np.ndarray, type_gate: np.ndarray,
                        sum_errors: Dict[str, float]) -> Dict[str, object]:
    """Save raw per-sample data and separate overall/per-class statistics."""
    stage_tag = f"stage_{stage_index:02d}"
    all_weights = np.concatenate((avg_scale, max_scale, type_gate), axis=1)
    stage_ids = np.full(labels.shape[0], stage_index, dtype=np.int64)

    np.savez_compressed(
        output_dir / f"{stage_tag}_gate_weights.npz",
        sample_index=sample_indices,
        label=labels,
        stage=stage_ids,
        avg_scale=avg_scale,
        max_scale=max_scale,
        type_gate=type_gate,
    )
    with (output_dir / f"{stage_tag}_gate_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_index", "label", "stage", *GATE_COLUMNS))
        for index, label, stage, weights in zip(sample_indices, labels, stage_ids, all_weights):
            writer.writerow((int(index), int(label), int(stage), *[f"{value:.10f}" for value in weights]))

    overall = metric_row(all_weights.mean(axis=0))
    overall.update({
        "stage": stage_index,
        "stage_name": stage_name,
        "num_samples": int(labels.size),
        "avg_scale_softmax_max_abs_error": sum_errors["avg_scale"],
        "max_scale_softmax_max_abs_error": sum_errors["max_scale"],
        "type_gate_softmax_max_abs_error": sum_errors["type"],
    })
    with (output_dir / f"{stage_tag}_overall_mean.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall))
        writer.writeheader()
        writer.writerow(overall)

    class_rows = []
    for label in np.unique(labels):
        class_weights = all_weights[labels == label]
        row = metric_row(class_weights.mean(axis=0))
        row.update({"label": int(label), "num_samples": int(class_weights.shape[0])})
        class_rows.append(row)
    with (output_dir / f"{stage_tag}_class_mean.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("label", "num_samples", *GATE_COLUMNS))
        writer.writeheader()
        writer.writerows(class_rows)

    return {"overall": overall, "class_means": class_rows}


def collect_gates(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.dataset not in DataSet:
        raise ValueError(
            f"{args.dataset} is not supported by this MedMNIST collector. "
            "Use a dedicated dataset loader before analysing CRC5000/LC25000."
        )
    if args.downsample != "dids":
        raise ValueError("DIDS gate analysis requires --downsample dids.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    genotype = parse_genotype(args.genotype, args.genotype_file, checkpoint, args.n_nodes, args.num_blocks)
    _, test_data, num_class = DataSet[args.dataset](size=28)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
        pin_memory=torch.cuda.is_available(), num_workers=args.num_workers,
    )

    if torch.cuda.is_available() and args.gpu >= 0:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu must be in [0, {torch.cuda.device_count() - 1}], received {args.gpu}.")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model = MSTF_NAS_NETWORK(
        num_class=num_class, genotype=genotype, n_n=args.n_nodes, num_block=args.num_blocks,
        init_c=args.init_channels, normal_space=args.normal_space, downsample=args.downsample,
        n_r=args.n_reduce_nodes,
    ).to(device)
    state_dict = checkpoint_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the requested discrete model. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}."
        )
    model.eval()
    dids_stages = named_dids_modules(model)
    for _, module in dids_stages:
        module.set_gate_recording(True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    buffers = {
        name: {"sample_index": [], "label": [], "avg_scale": [], "max_scale": [], "type": [],
               "sum_errors": {"avg_scale": 0.0, "max_scale": 0.0, "type": 0.0}}
        for name, _ in dids_stages
    }
    next_sample_index = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long()
            _ = model(images)
            batch_size = labels.numel()
            sample_indices = np.arange(next_sample_index, next_sample_index + batch_size, dtype=np.int64)
            next_sample_index += batch_size
            labels_np = labels.numpy().astype(np.int64, copy=False)

            for stage_name, module in dids_stages:
                gates = module.get_last_gate_weights()
                if gates is None:
                    raise RuntimeError(f"{stage_name} did not record gate weights.")
                errors = validate_gate_weights(gates, batch_size, stage_name)
                stage_buffer = buffers[stage_name]
                stage_buffer["sample_index"].append(sample_indices)
                stage_buffer["label"].append(labels_np)
                for name in ("avg_scale", "max_scale", "type"):
                    stage_buffer[name].append(gates[name].cpu().numpy().copy())
                    stage_buffer["sum_errors"][name] = max(stage_buffer["sum_errors"][name], errors[name])

    summary = {
        "dataset": args.dataset,
        "checkpoint": str(checkpoint_path),
        "genotype": genotype,
        "normal_space": args.normal_space,
        "downsample": args.downsample,
        "num_classes": num_class,
        "num_test_samples": next_sample_index,
        "stages": [],
    }
    for stage_index, (stage_name, _) in enumerate(dids_stages, start=1):
        stage_buffer = buffers[stage_name]
        stage_summary = write_stage_outputs(
            output_dir, stage_index, stage_name,
            np.concatenate(stage_buffer["sample_index"]),
            np.concatenate(stage_buffer["label"]),
            np.concatenate(stage_buffer["avg_scale"]),
            np.concatenate(stage_buffer["max_scale"]),
            np.concatenate(stage_buffer["type"]),
            stage_buffer["sum_errors"],
        )
        stage_summary["stage_index"] = stage_index
        stage_summary["stage_name"] = stage_name
        summary["stages"].append(stage_summary)
        overall = stage_summary["overall"]
        print(
            f"{args.dataset} | stage {stage_index} ({stage_name}) | n={overall['num_samples']} | "
            f"Avg Scale sum max error={overall['avg_scale_softmax_max_abs_error']:.3e}, "
            f"Max Scale sum max error={overall['max_scale_softmax_max_abs_error']:.3e}, "
            f"Type sum max error={overall['type_gate_softmax_max_abs_error']:.3e}"
        )
    with (output_dir / "dids_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved DIDS per-sample gates and stage statistics: {output_dir}")
    return output_dir


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _heat_color(value: float) -> Tuple[int, int, int]:
    """A dependency-free perceptually ordered blue-to-red heatmap colour."""
    value = min(max(float(value), 0.0), 1.0)
    if value < 0.5:
        ratio = value * 2.0
        return (int(30 + 180 * ratio), int(90 + 120 * ratio), int(210 - 70 * ratio))
    ratio = (value - 0.5) * 2.0
    return (int(210 + 35 * ratio), int(210 - 150 * ratio), int(140 - 100 * ratio))


def _draw_heatmap(draw: ImageDraw.ImageDraw, origin: Tuple[int, int], values: np.ndarray,
                  row_labels: Sequence[str], column_labels: Sequence[str], title: str,
                  cell_width: int, cell_height: int, fonts: Dict[str, object]) -> None:
    x0, y0 = origin
    label_width = max(220, max((draw.textbbox((0, 0), label, font=fonts["row"])[2] for label in row_labels), default=0) + 24)
    draw.text((x0 + label_width, y0), title, fill="black", font=fonts["title"])
    top = y0 + 48
    for column, label in enumerate(column_labels):
        x = x0 + label_width + column * cell_width
        bbox = draw.textbbox((0, 0), label, font=fonts["header"])
        draw.text((x + (cell_width - (bbox[2] - bbox[0])) / 2, top), label, fill="black", font=fonts["header"])
    top += 34
    for row, (label, row_values) in enumerate(zip(row_labels, values)):
        y = top + row * cell_height
        bbox = draw.textbbox((0, 0), label, font=fonts["row"])
        draw.text((x0 + label_width - 12 - (bbox[2] - bbox[0]), y + (cell_height - (bbox[3] - bbox[1])) / 2),
                  label, fill="black", font=fonts["row"])
        for column, value in enumerate(row_values):
            x = x0 + label_width + column * cell_width
            color = _heat_color(float(value))
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=color, outline="white", width=2)
            text = f"{value:.3f}"
            text_box = draw.textbbox((0, 0), text, font=fonts["value"])
            text_color = "white" if float(value) < 0.35 or float(value) > 0.72 else "black"
            draw.text(
                (x + (cell_width - (text_box[2] - text_box[0])) / 2,
                 y + (cell_height - (text_box[3] - text_box[1])) / 2),
                text, fill=text_color, font=fonts["value"],
            )


def plot_gate_summaries(args: argparse.Namespace) -> Path:
    summaries = []
    for directory in args.plot_dirs:
        path = Path(directory).expanduser().resolve() / "dids_gate_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing analysis summary: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows, row_labels = [], []
    for summary in summaries:
        for stage in summary["stages"]:
            if args.plot_level == "dataset":
                source_rows = [("", stage["overall"])]
            else:
                source_rows = [(f" class {row['label']}", row) for row in stage["class_means"]]
            for suffix, values in source_rows:
                rows.append([values[column] for column in GATE_COLUMNS])
                row_labels.append(f"{summary['dataset']} | S{stage['stage_index']}{suffix}")
    if not rows:
        raise RuntimeError("No DIDS gate rows were found in the provided summaries.")

    values = np.asarray(rows, dtype=float)
    row_count = values.shape[0]
    cell_width, cell_height = 132, 48
    left_width = max(250, max(len(label) for label in row_labels) * 10 + 24)
    scale_width = left_width + cell_width * len(SCALE_LABELS) + 32
    type_width = left_width + cell_width * len(TYPE_LABELS) + 32
    figure_width = scale_width + type_width + 80
    figure_height = max(360, 118 + row_count * cell_height + 42)
    image = Image.new("RGB", (figure_width, figure_height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {"title": _font(24), "header": _font(16), "row": _font(15), "value": _font(15)}
    draw.text((28, 18), f"DIDS Gate Weights ({args.plot_level}-mean, test set)", fill="black", font=_font(28))
    _draw_heatmap(draw, (28, 66), values[:, :6], row_labels, SCALE_LABELS, "Scale Gate", cell_width, cell_height, fonts)
    _draw_heatmap(draw, (scale_width + 44, 66), values[:, 6:], row_labels, TYPE_LABELS, "Type Gate", cell_width, cell_height, fonts)

    output_base = Path(args.plot_output).expanduser().resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    image.save(png_path, dpi=(600, 600))
    image.save(pdf_path, "PDF", resolution=600.0)
    metadata_path = output_base.with_name(output_base.name + "_data.json")
    metadata_path.write_text(json.dumps({
        "plot_level": args.plot_level,
        "row_labels": row_labels,
        "columns": list(GATE_COLUMNS),
        "values": values.tolist(),
        "source_directories": [str(Path(directory).resolve()) for directory in args.plot_dirs],
    }, indent=2), encoding="utf-8")
    print(f"Saved DIDS heatmap PNG (600 dpi): {png_path}")
    print(f"Saved DIDS heatmap PDF: {pdf_path}")
    return png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect or visualize DIDS dynamic gate weights without training.")
    parser.add_argument("--plot_dirs", nargs="+", help="Existing analysis output directories; enables plot-only mode.")
    parser.add_argument("--plot_output", default="dids_gate_analysis/dids_gate_weights",
                        help="Output base path for plot-only mode, without extension.")
    parser.add_argument("--plot_level", choices=("dataset", "class"), default="dataset")
    parser.add_argument("--dataset", help="MedMNIST dataset name for collection mode.")
    parser.add_argument("--checkpoint", help="Trained discrete-model checkpoint for collection mode.")
    parser.add_argument("--genotype", help="One-Block or full-network genotype list.")
    parser.add_argument("--genotype_file", help="Path to genotype.txt for collection mode.")
    parser.add_argument("--output_dir", default="dids_gate_analysis/output", help="Collection output directory.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID; pass -1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=128, help="Test inference batch size.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--normal_space", choices=("pdc", "normal"), default="pdc")
    parser.add_argument("--downsample", choices=("dids", "reduction"), default="dids")
    parser.add_argument("--n_nodes", type=int, default=5)
    parser.add_argument("--n_reduce_nodes", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--init_channels", type=int, default=128)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.plot_dirs:
        plot_gate_summaries(cli_args)
    else:
        if not cli_args.dataset or not cli_args.checkpoint:
            raise SystemExit("Collection mode requires --dataset and --checkpoint.")
        if cli_args.batch_size <= 0:
            raise SystemExit("--batch_size must be positive.")
        collect_gates(cli_args)
