#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot import EPOCHS, METRICS


NumberList = List[float]
MetricData = Dict[str, Dict[str, NumberList]]
TARGET_METRICS = ["MACCS FTS", "RDK FTS", "Morgan FTS"]
CHINESE_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MACCS FTS, RDK FTS, and Morgan FTS on the same axes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/similarity_metrics_combined.png"),
        help="Output image path (default: plots/similarity_metrics_combined.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI (default: 220).",
    )
    return parser.parse_args()


def validate_inputs(epochs: List[int], metrics: MetricData) -> None:
    if not epochs:
        raise ValueError("EPOCHS must not be empty.")

    for metric_name in TARGET_METRICS:
        if metric_name not in metrics:
            raise KeyError(f"Metric not found: {metric_name}")

        method_dict = metrics[metric_name]
        if not isinstance(method_dict, dict) or len(method_dict) != 2:
            raise ValueError(f"{metric_name} must contain exactly two methods.")

        for method_name, values in method_dict.items():
            if len(values) != len(epochs):
                raise ValueError(
                    f"The length of {metric_name}/{method_name} ({len(values)}) does not "
                    f"match the length of EPOCHS ({len(epochs)})."
                )


def add_x_axis_style(ax: plt.Axes, epochs: List[int]) -> None:
    ax.set_xticks(epochs)
    ax.set_xticklabels([str(epoch) for epoch in epochs])

    if len(epochs) > 1:
        avg_step = (max(epochs) - min(epochs)) / (len(epochs) - 1)
        x_padding = avg_step * 0.35
    else:
        x_padding = 1.0

    ax.set_xlim(min(epochs) - x_padding, max(epochs) + x_padding)


def plot_combined(epochs: List[int], metrics: MetricData, output: Path, dpi: int) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib import font_manager

    latin_font = font_manager.FontProperties(family="DejaVu Sans")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.6))
    metric_colors = {
        "MACCS FTS": "tab:blue",
        "RDK FTS": "tab:orange",
        "Morgan FTS": "tab:green",
    }
    method_label_map = {
        "global-only": "w/o all",
        "MolFrag": "MoGL",
        "w/o all": "w/o all",
        "Our Method": "MoGL",
    }
    method_linestyles = {
        "w/o all": "--",
        "MoGL": "-",
    }

    for metric_name in TARGET_METRICS:
        method_dict = metrics[metric_name]
        method_names = list(method_dict.keys())

        for raw_method_name in method_names:
            display_method_name = method_label_map.get(raw_method_name, raw_method_name)
            ax.plot(
                epochs,
                method_dict[raw_method_name],
                color=metric_colors.get(metric_name, "tab:blue"),
                marker="o",
                linestyle=method_linestyles.get(display_method_name, "-"),
                linewidth=2,
            )

    add_x_axis_style(ax, epochs)
    ax.set_xlabel("Epoch", fontproperties=latin_font)
    ax.set_ylabel("Score", fontproperties=latin_font)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontproperties(latin_font)
    ax.grid(True, linestyle="--", alpha=0.35)

    metric_handles = [
        Line2D([0], [0], color=metric_colors[name], linewidth=2.5, label=name)
        for name in TARGET_METRICS
    ]
    method_handles = [
        Line2D([0], [0], color="black", linestyle=style, linewidth=2.5, label=name)
        for name, style in method_linestyles.items()
    ]

    metric_legend = fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.34, 0.045),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        handlelength=2.8,
    )

    method_legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.82, 0.045),
        ncol=2,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        handlelength=2.8,
    )

    for text in metric_legend.get_texts():
        text.set_fontproperties(latin_font)

    zh_font = (
        font_manager.FontProperties(fname=str(CHINESE_FONT_PATH))
        if CHINESE_FONT_PATH.exists()
        else None
    )
    for text in method_legend.get_texts():
        if any(ord(ch) > 127 for ch in text.get_text()) and zh_font is not None:
            text.set_fontproperties(zh_font)
        else:
            text.set_fontproperties(latin_font)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_inputs(EPOCHS, METRICS)
    plot_combined(EPOCHS, METRICS, args.output, args.dpi)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
