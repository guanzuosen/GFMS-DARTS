"""Retrain four supplied E4 genotypes (seed 0) and analyse DIDS gates.

No search is invoked and no per-epoch files are written.  For each dataset,
only the best-Test-Top-1 discrete checkpoint plus final DIDS statistics and
figures are retained.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from DataSets import DataSet
from analyze_dids_gates import collect_gates, plot_gate_summaries
from model.net_cell import MSTF_NAS_NETWORK


# Genotypes supplied by the user, in the requested representative-dataset order.
E4_GENOTYPES = {
    "PathMNIST": [4, 0, 2, 0, 2, 0, 4, 0, 4, 0],
    "PneumoniaMNIST": [4, 0, 6, 0, 2, 2, 1, 2, 1, 2],
    "RetinaMNIST": [0, 0, 4, 1, 4, 0, 0, 1, 0, 4],
    "BreastMNIST": [6, 0, 0, 1, 1, 0, 4, 1, 6, 0],
}

TRAIN_CONFIGS = {
    "PathMNIST": {"batch_size": 512, "lr": 0.05},
    "PneumoniaMNIST": {"batch_size": 128, "lr": 0.0005},
    "RetinaMNIST": {"batch_size": 128, "lr": 0.001},
    "BreastMNIST": {"batch_size": 128, "lr": 0.001},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, num_class):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1).long()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
    return total_loss / total_samples, 100.0 * correct / total_samples


def evaluate_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1).long()
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_samples += labels.size(0)
            correct += logits.argmax(dim=1).eq(labels).sum().item()
    return total_loss / total_samples, 100.0 * correct / total_samples


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def retrain_dataset(dataset: str, args, device: torch.device, root: Path) -> Path:
    config = TRAIN_CONFIGS[dataset]
    cell_genotype = E4_GENOTYPES[dataset]
    full_genotype = cell_genotype * args.num_blocks
    seed_everything(args.seed)
    train_data, test_data, num_class = DataSet[dataset](size=28)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
    )
    # Keep the existing regular-training protocol: evaluate full batches only.
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True,
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
    )
    model = MSTF_NAS_NETWORK(
        num_class=num_class, genotype=full_genotype, n_n=args.n_nodes,
        num_block=args.num_blocks, init_c=args.init_channels,
        normal_space="pdc", downsample="dids", n_r=args.n_reduce_nodes,
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = float("-inf")
    best_epoch = 0
    best_state = None
    print(f"\n===== E4 retraining | {dataset} | seed={args.seed} | GPU={args.gpu} =====")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, num_class)
        test_loss, test_acc = evaluate_accuracy(model, test_loader, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_state = cpu_state_dict(model)
        if (epoch + 1) % args.report_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"{dataset} epoch {epoch + 1:03d}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%"
            )
    if best_state is None:
        raise RuntimeError(f"{dataset}: no checkpoint state was collected.")

    dataset_output = root / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dataset_output / "best_test_top1_checkpoint.pt"
    torch.save({
        "model_state_dict": best_state,
        "genotype": full_genotype,
        "cell_genotype": cell_genotype,
        "dataset": dataset,
        "seed": args.seed,
        "epoch": best_epoch,
        "best_test_top1_accuracy": best_acc,
        "normal_space": "pdc",
        "downsample": "dids",
        "init_channels": args.init_channels,
        "num_blocks": args.num_blocks,
    }, checkpoint_path)
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    analysis_args = SimpleNamespace(
        dataset=dataset,
        checkpoint=str(checkpoint_path),
        genotype=None,
        genotype_file=None,
        n_nodes=args.n_nodes,
        num_blocks=args.num_blocks,
        n_reduce_nodes=args.n_reduce_nodes,
        init_channels=args.init_channels,
        normal_space="pdc",
        downsample="dids",
        gpu=args.gpu,
        batch_size=args.analysis_batch_size,
        num_workers=args.num_workers,
        output_dir=str(dataset_output / "dids_gates"),
    )
    collect_gates(analysis_args)
    (dataset_output / "retrain_summary.json").write_text(json.dumps({
        "dataset": dataset,
        "seed": args.seed,
        "cell_genotype": cell_genotype,
        "best_test_top1_epoch": best_epoch,
        "best_test_top1_accuracy": best_acc,
        "checkpoint": str(checkpoint_path),
    }, indent=2), encoding="utf-8")
    print(f"{dataset}: saved only final checkpoint and DIDS analysis to {dataset_output}")
    return dataset_output / "dids_gates"


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain seed-0 E4 models and directly create DIDS gate figures.")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--analysis_batch_size", type=int, default=128)
    parser.add_argument("--init_channels", type=int, default=128)
    parser.add_argument("--n_nodes", type=int, default=5)
    parser.add_argument("--n_reduce_nodes", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.epochs <= 0 or args.analysis_batch_size <= 0:
        raise SystemExit("--epochs and --analysis_batch_size must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required by this requested run, but CUDA is unavailable.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"Invalid --gpu {args.gpu}; available range is 0-{torch.cuda.device_count() - 1}.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = torch.backends.cudnn.enabled
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.output_dir or f"dids_gate_analysis/E4-seed{args.seed}-{timestamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    analysis_dirs = [retrain_dataset(dataset, args, device, root) for dataset in E4_GENOTYPES]
    for level in ("dataset", "class"):
        plot_gate_summaries(SimpleNamespace(
            plot_dirs=[str(path) for path in analysis_dirs],
            plot_output=str(root / f"DIDS_Gate_Weights_{level}"),
            plot_level=level,
        ))
    print(f"\nCompleted E4 seed-{args.seed} retraining and DIDS analysis: {root}")
