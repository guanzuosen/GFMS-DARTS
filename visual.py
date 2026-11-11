"""Visualize a searched Normal Cell genotype.

The project encodes one Normal Cell as::

    [op_index, input_state_index] * n_nodes

For example, ``[2, 0, 4, 0, 2, 1, 4, 0, 4, 0]`` represents five
operation nodes.  State ``h[0]`` is the Cell input; the output is the mean
fusion (``concat`` in the original architecture diagram) of all states that
are not used as the input of another node.

Examples
--------
    python visual.py --genotype "[2, 0, 4, 0, 2, 1, 4, 0, 4, 0]" \\
        --normal_space pdc --output figures/path_cell.png

    python visual.py --genotype_file search-PathMNIST-pdc-dids-*/genotype.txt \\
        --output path_cell.svg
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from model.operators import MSTF_NAS_SPACE_DARTS, MSTF_NAS_SPACE_PDC


OP_LABELS = {
    "none": "None / Zero",
    "max_pool_3x3": "MaxPool 3×3",
    "avg_pool_3x3": "AvgPool 3×3",
    "skip_connect": "Skip",
    "sep_conv_3x3": "Sep 3×3",
    "sep_conv_5x5": "Sep 5×5",
    "dil_conv_3x3": "Dil 3×3",
    "dil_conv_5x5": "Dil 5×5",
    "pdc_sep_conv_3x3": "PDC3",
    "pdc_sep_conv_5x5": "PDC5",
    "conv_1x1": "Conv 1×1",
    "asym_conv_7x7": "Asym 7×1 → 1×7",
}


def _normal_space(name: str) -> Sequence[str]:
    spaces = {"normal": MSTF_NAS_SPACE_DARTS, "pdc": MSTF_NAS_SPACE_PDC}
    try:
        return spaces[name]
    except KeyError as exc:
        raise ValueError("normal_space must be 'normal' or 'pdc'.") from exc


def _parse_genotype(genotype: Sequence[int] | str | Path) -> list[int]:
    """Accept an integer list, its Python-literal text, or a genotype file."""
    if isinstance(genotype, Path):
        text = genotype.read_text(encoding="utf-8").strip()
        genotype = text
    elif isinstance(genotype, str):
        possible_path = Path(genotype)
        if possible_path.is_file():
            genotype = possible_path.read_text(encoding="utf-8").strip()

    if isinstance(genotype, str):
        try:
            genotype = ast.literal_eval(genotype)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("genotype must be a Python-style integer list.") from exc

    if not isinstance(genotype, (list, tuple)) or not all(isinstance(x, int) for x in genotype):
        raise ValueError("genotype must be a list (or tuple) of integers.")
    if not genotype or len(genotype) % 2:
        raise ValueError("Normal Cell genotype must contain a non-zero, even number of values.")
    return list(genotype)


def _dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def genotype_to_dot(genotype: Sequence[int] | str | Path, normal_space: str = "pdc",
                    title: str = "") -> str:
    """Return Graphviz DOT source for one project Normal Cell genotype.

    The output follows the actual :class:`model.net_cell.Normal_Cell` forward
    logic: each searched pair creates one state and unreferenced states are
    merged at the Cell output.
    """
    cell = _parse_genotype(genotype)
    operations = _normal_space(normal_space)
    n_nodes = len(cell) // 2
    op_indices, input_indices = cell[0::2], cell[1::2]

    for node, (op_index, input_index) in enumerate(zip(op_indices, input_indices), start=1):
        if not 0 <= op_index < len(operations):
            raise ValueError(
                f"Node {node}: operation index {op_index} is invalid for {normal_space} "
                f"space (valid range: 0-{len(operations) - 1})."
            )
        # Node i is generated after states h[0] ... h[i-1] exist.
        if not 0 <= input_index < node:
            raise ValueError(
                f"Node {node}: input index {input_index} is invalid; expected 0-{node - 1}."
            )

    used_as_input = set(input_indices)
    output_states = [state for state in range(n_nodes + 1) if state not in used_as_input]
    if not output_states:
        raise RuntimeError("The genotype has no terminal state to fuse.")

    lines = [
        "digraph NormalCell {",
        "  rankdir=LR;",
        "  graph [bgcolor=white, pad=0.18, nodesep=0.40, ranksep=0.80, "
        "labelloc=t, label=" + _dot_quote(title) + ", fontsize=18, fontname=\"Times-Roman\"];",
        "  node [shape=rect, style=filled, fontname=\"Times-Roman\", fontsize=14, "
        "margin=\"0.10,0.06\", color=\"#263238\", penwidth=1.2];",
        "  edge [fontname=\"Times-Roman\", fontsize=13, color=\"#424242\", penwidth=1.0, arrowsize=0.65];",
        "  h0 [label=\"h[i]\", fillcolor=\"#B7D7A8\"];",
    ]

    for node in range(1, n_nodes + 1):
        lines.append(
            f"  h{node} [label=\"node\", fillcolor=\"#A9D5E8\"];"
        )

    lines.extend([
        "  fuse [label=\"Fuse\", fillcolor=\"#EFB3C6\"];",
        "  output [label=\"h[i+1]\", fillcolor=\"#FFF2A8\"];",
    ])

    for node, (op_index, input_index) in enumerate(zip(op_indices, input_indices), start=1):
        op_name = operations[op_index]
        label = OP_LABELS.get(op_name, op_name)
        lines.append(f"  h{input_index} -> h{node} [label={_dot_quote(label)}];")

    for state in output_states:
        lines.append(f"  h{state} -> fuse;")
    lines.append("  fuse -> output;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def visualize_genotype(genotype: Sequence[int] | str | Path, output_path: str | Path,
                       normal_space: str = "pdc", title: str = "",
                       keep_dot: bool = True) -> Path:
    """Render one Normal Cell genotype to ``output_path`` with Graphviz.

    ``output_path`` suffix determines the format, e.g. ``.png``, ``.pdf`` or
    ``.svg``.  A same-name ``.dot`` source file is retained by default.
    Returns the rendered image path.
    """
    output = Path(output_path).expanduser()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    dot_file = output.with_suffix(".dot")
    dot_file.write_text(genotype_to_dot(genotype, normal_space=normal_space, title=title), encoding="utf-8")

    dot_binary = shutil.which("dot")
    if dot_binary is None:
        raise RuntimeError("Graphviz executable 'dot' was not found on PATH.")
    try:
        subprocess.run(
            [dot_binary, f"-T{output.suffix[1:]}", str(dot_file), "-o", str(output)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Graphviz failed to render {dot_file}: {exc.stderr.strip()}") from exc

    if not keep_dot:
        dot_file.unlink()
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one GDAR-MedMNIST Normal Cell genotype.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--genotype", help="Genotype, for example '[2,0, 4,0, 2,1, 4,0, 4,0]'.")
    source.add_argument("--genotype_file", type=Path, help="Path to a genotype.txt file.")
    parser.add_argument("--normal_space", choices=["normal", "pdc"], default="pdc")
    parser.add_argument("--output", type=Path, default=Path("normal_cell.png"))
    parser.add_argument("--title", default="", help="Optional title shown above the Cell diagram.")
    parser.add_argument("--remove_dot", action="store_true", help="Do not retain the generated DOT source file.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    source: Iterable[int] | str | Path = args.genotype_file if args.genotype_file else args.genotype
    result = visualize_genotype(
        source, args.output, normal_space=args.normal_space, title=args.title,
        keep_dot=not args.remove_dot,
    )
    print(f"Saved Cell visualization: {result.resolve()}")
