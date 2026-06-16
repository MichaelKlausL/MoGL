#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


NumberList = List[float]
MetricData = Dict[str, Dict[str, NumberList]]


# =====================
# 直接在这里改数据即可
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
        description="画出某个指标在不同 epoch 下的折线图（每张图两个方法，默认使用代码内置数据）。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="可选：JSON 文件路径。不传则使用代码内置的 EPOCHS/METRICS。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="输出图片目录，默认 plots。",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "pdf", "svg"],
        help="图片格式，默认 png。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="图片 DPI，默认 200。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象。")
    return data


def normalize_data(data: dict) -> Tuple[List[int], MetricData]:
    """
    支持两种 JSON 格式：

    1) 推荐格式：
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

    2) 简化格式（自动推断 epoch）：
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
        raise ValueError("未找到有效的 metrics 数据。")

    first_metric = next(iter(metrics.values()))
    if not isinstance(first_metric, dict):
        raise ValueError("每个指标下应为方法到数值列表的映射。")

    if len(first_metric) != 2:
        raise ValueError("每张图必须且只能有两个方法。")

    first_method_values = next(iter(first_metric.values()))
    if not isinstance(first_method_values, list) or not first_method_values:
        raise ValueError("方法对应的数据必须是非空列表。")

    n_points = len(first_method_values)
    if raw_epochs is None:
        epochs = list(range(1, n_points + 1))
    else:
        if not isinstance(raw_epochs, list):
            raise ValueError("epochs 必须是列表。")
        if len(raw_epochs) != n_points:
            raise ValueError("epochs 长度必须与指标曲线长度一致。")
        epochs = raw_epochs

    normalized: MetricData = {}
    for metric_name, method_dict in metrics.items():
        if not isinstance(method_dict, dict):
            raise ValueError(f"指标 {metric_name} 的值应为对象。")
        if len(method_dict) != 2:
            raise ValueError(f"指标 {metric_name} 下方法数量不是 2。")

        normalized[metric_name] = {}
        for method_name, values in method_dict.items():
            if not isinstance(values, list):
                raise ValueError(f"{metric_name}/{method_name} 必须是列表。")
            if len(values) != len(epochs):
                raise ValueError(
                    f"{metric_name}/{method_name} 长度({len(values)})与 epoch 数({len(epochs)})不一致。"
                )
            normalized[metric_name][method_name] = values

    return epochs, normalized


def normalize_inline_data(epochs: List[int], metrics: MetricData) -> Tuple[List[int], MetricData]:
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("EPOCHS 必须是非空列表。")

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("METRICS 必须是非空字典。")

    normalized: MetricData = {}
    for metric_name, method_dict in metrics.items():
        if not isinstance(method_dict, dict):
            raise ValueError(f"指标 {metric_name} 的值应为对象。")
        if len(method_dict) != 2:
            raise ValueError(f"指标 {metric_name} 下方法数量不是 2。")

        normalized[metric_name] = {}
        for method_name, values in method_dict.items():
            if not isinstance(values, list):
                raise ValueError(f"{metric_name}/{method_name} 必须是列表。")
            if len(values) != len(epochs):
                raise ValueError(
                    f"{metric_name}/{method_name} 长度({len(values)})与 epoch 数({len(epochs)})不一致。"
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
        print(f"已保存: {out_file}")


if __name__ == "__main__":
    main()
