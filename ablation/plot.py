#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


NumberList = List[float]
MetricData = Dict[str, Dict[str, NumberList]]


# =====================
# Edit the data directly here.
# =====================
EPOCHS: List[int] = [20, 30, 40, 50]

METRICS: MetricData = {
    "BLEU": {
        "global-only": [0.824, 0.837, 0.842, 0.852],
        "MolFrag": [0.831, 0.844, 0.856, 0.860],
    },
    "EXACT": {
        "global-only": [0.208, 0.249, 0.275, 0.289],
        "MolFrag": [0.217, 0.285, 0.308, 0.324],
    },
    "LEVENSHTEIN": {
        "global-only": [19.612, 18.191, 17.462, 15.912],
        "MolFrag": [18.662, 16.7, 15.338, 14.945],
    },
    "MACCS FTS": {
        "global-only": [0.881, 0.891, 0.897, 0.899],
        "MolFrag": [0.884, 0.899, 0.905, 0.908],
    },
    "RDK FTS": {
        "global-only": [0.778, 0.795, 0.806, 0.811],
        "MolFrag": [0.784, 0.810, 0.825, 0.828],
    },
    "Morgan FTS": {
        "global-only": [0.707, 0.732, 0.743, 0.752],
        "MolFrag": [0.715, 0.747, 0.763, 0.770],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a metric across epochs, comparing two methods per figure. "
            "By default, the data defined in this script is used."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a JSON file. If omitted, the built-in EPOCHS/METRICS data is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "pdf", "svg"],
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI (default: 200).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("The top-level JSON value must be an object.")
    return data


def normalize_data(data: dict) -> Tuple[List[int], MetricData]:
    """
    Support two JSON formats:

    1) Recommended format:
    {
      "epochs": [1, 2, 3, ...],
      "metrics": {
        "acc": {
          "method_a": [0.1, 0.2, 0.3],
          "method_b": [0.15, 0.25, 0.35]
        },
        "f1": {...}
      }
    }

    2) Simplified format (epochs are inferred automatically):
    {
      "acc": {
        "method_a": [0.1, 0.2, 0.3],
        "method_b": [0.15, 0.25, 0.35]
      },
      "f1": {...}
    }
    """
    if "metrics" in data:
        metrics = data["metrics"]
        raw_epochs = data.get("epochs")
    else:
        metrics = data
        raw_epochs = None

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("No valid metrics data was found.")

    first_metric = next(iter(metrics.values()))
    if not isinstance(first_metric, dict):
        raise ValueError("Each metric must map method names to lists of values.")

    if len(first_metric) != 2:
        raise ValueError("Each plot must contain exactly two methods.")

    first_method_values = next(iter(first_metric.values()))
    if not isinstance(first_method_values, list) or not first_method_values:
        raise ValueError("The values for each method must be a non-empty list.")

    n_points = len(first_method_values)
    if raw_epochs is None:
        epochs = list(range(1, n_points + 1))
    else:
        if not isinstance(raw_epochs, list):
            raise ValueError("epochs must be a list.")
        if len(raw_epochs) != n_points:
            raise ValueError("The length of epochs must match the number of metric data points.")
        epochs = raw_epochs

    normalized: MetricData = {}
    for metric_name, method_dict in metrics.items():
        if not isinstance(method_dict, dict):
            raise ValueError(f"The value of metric {metric_name} must be an object.")
        if len(method_dict) != 2:
            raise ValueError(f"Metric {metric_name} must contain exactly two methods.")

        normalized[metric_name] = {}
        for method_name, values in method_dict.items():
            if not isinstance(values, list):
                raise ValueError(f"{metric_name}/{method_name} must be a list.")
            if len(values) != len(epochs):
                raise ValueError(
                    f"The length of {metric_name}/{method_name} ({len(values)}) does not "
                    f"match the number of epochs ({len(epochs)})."
                )
            normalized[metric_name][method_name] = values

    return epochs, normalized


def normalize_inline_data(epochs: List[int], metrics: MetricData) -> Tuple[List[int], MetricData]:
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("EPOCHS must be a non-empty list.")

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("METRICS must be a non-empty dictionary.")

    normalized: MetricData = {}
    for metric_name, method_dict in metrics.items():
        if not isinstance(method_dict, dict):
            raise ValueError(f"The value of metric {metric_name} must be an object.")
        if len(method_dict) != 2:
            raise ValueError(f"Metric {metric_name} must contain exactly two methods.")

        normalized[metric_name] = {}
        for method_name, values in method_dict.items():
            if not isinstance(values, list):
                raise ValueError(f"{metric_name}/{method_name} must be a list.")
            if len(values) != len(epochs):
                raise ValueError(
                    f"The length of {metric_name}/{method_name} ({len(values)}) does not "
                    f"match the number of epochs ({len(epochs)})."
                )
            normalized[metric_name][method_name] = values

    return epochs, normalized


def plot_metric(
    epochs: List[int],
    metric_name: str,
    method_values: Dict[str, NumberList],
    output_path: Path,
    dpi: int,
) -> None:
    method_names = list(method_values.keys())
    y1 = method_values[method_names[0]]
    y2 = method_values[method_names[1]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, y1, marker="o", linewidth=2, label=method_names[0])
    plt.plot(epochs, y2, marker="s", linewidth=2, label=method_names[1])
    plt.xticks(epochs, [str(epoch) for epoch in epochs])
    if len(epochs) > 1:
        avg_step = (max(epochs) - min(epochs)) / (len(epochs) - 1)
        x_padding = avg_step * 0.35
    else:
        x_padding = 1.0
    plt.xlim(min(epochs) - x_padding, max(epochs) + x_padding)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    # plt.title(f"{metric_name} vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.input is not None:
        data = load_json(args.input)
        epochs, metrics = normalize_data(data)
    else:
        epochs, metrics = normalize_inline_data(EPOCHS, METRICS)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, method_dict in metrics.items():
        safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
        out_file = args.output_dir / f"{safe_metric_name}.{args.format}"
        plot_metric(epochs, metric_name, method_dict, out_file, args.dpi)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
