import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import ChEBI_20_data_Dataset, MolFrag_Dataset, PubChem_Dataset


def build_parser(config):
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for different checkpoints (token/s and sample/min)."
    )
    parser.add_argument("--config", type=str, default="_yamls/Eval_Atomas.yaml")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Optional; defaults to resume_from_checkpoint from the config when omitted.",
    )
    parser.add_argument("--method_names", type=str, nargs="*", default=None)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.get("dataset", "ChEBI-20_data")),
        choices=["pubchemstm", "ChEBI-20_data", "MolFrag"],
    )
    parser.add_argument("--split", type=str, default="filtered")
    parser.add_argument("--test_split", type=str, default=str(config.get("test_split", "test")))
    parser.add_argument("--batch_size", type=int, default=int(config.get("batch_size", 1)))
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--project", type=str, default="Atomas")
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--version", type=str, default=str(config.get("version", "speed-benchmark")))
    parser.add_argument("--model_size", type=str, default=str(config.get("model_size", "base")))
    parser.add_argument("--queue_size", type=int, default=int(config.get("queue_size", 13200)))
    parser.add_argument(
        "--task",
        type=str,
        default=str(config.get("task", "genmol")),
        choices=["genmol", "gentext"],
    )
    parser.add_argument("--max_lenth", type=int, default=int(config.get("max_lenth", 512)))
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tsclosswt", type=float, default=float(config.get("tsclosswt", 1.0)))
    parser.add_argument("--lmlosswt", type=float, default=float(config.get("lmlosswt", 10.0)))
    parser.add_argument("--wtilosswt", type=float, default=float(config.get("wtilosswt", 1.0)))
    parser.add_argument(
        "--missing_fragment_loss_wt",
        type=float,
        default=float(config.get("missing_fragment_loss_wt", 1.0)),
    )
    parser.add_argument(
        "--frag2mol_loss_wt",
        type=float,
        default=float(config.get("frag2mol_loss_wt", 1.0)),
    )
    parser.add_argument(
        "--keyword_loss_wt",
        type=float,
        default=float(config.get("keyword_loss_wt", 1.0)),
    )
    parser.add_argument("--textencoder", type=str, default="molt5")
    parser.add_argument("--encode_text_lr", type=float, default=float(config.get("encode_text_lr", 1e-4)))
    parser.add_argument("--encode_smiles_lr", type=float, default=float(config.get("encode_smiles_lr", 1e-4)))
    parser.add_argument("--molt5_lr", type=float, default=float(config.get("molt5_lr", 2e-4)))
    parser.add_argument("--text_lr_scale", type=float, default=float(config.get("text_lr_scale", 0.1)))
    parser.add_argument("--smiles_lr_scale", type=float, default=float(config.get("smiles_lr_scale", 0.1)))
    parser.add_argument("--decay", type=float, default=float(config.get("decay", 0.0)))
    parser.add_argument("--precision", default=config.get("precision", "bf16"))
    parser.add_argument("--temp_dir", type=str, default="./output_data/speed_benchmark")

    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--generate_max_length", type=int, default=512)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--token_scope",
        type=str,
        default="output",
        choices=["output", "total"],
        help="output: count generated tokens only; total: count input and output tokens.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="")

    return parser


def build_test_loader(args):
    if args.dataset == "ChEBI-20_data":
        test_data = ChEBI_20_data_Dataset(args.data_dir, args.dataset, args.test_split)
    elif args.dataset == "pubchemstm":
        test_data = PubChem_Dataset(args.data_dir, "pubchemstm", args.split)
    elif args.dataset == "MolFrag":
        test_data = MolFrag_Dataset(args.data_dir, args.dataset, args.test_split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return test_loader


def load_checkpoint_state_dict(checkpoint_path):
    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_inputs_from_batch(batch, task):
    if task == "genmol":
        return batch["description"]
    return batch["smiles"]


def count_non_pad_tokens(token_ids, pad_token_id):
    if pad_token_id is None:
        return int(token_ids.numel())
    return int((token_ids != pad_token_id).sum().item())


@torch.no_grad()
def benchmark_checkpoint(model, dataloader, args, device):
    model.eval()
    model.to(device)

    pad_token_id = model.tokenizer.pad_token_id
    measured_batches = 0
    measured_samples = 0
    total_tokens = 0
    total_elapsed = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_batches > 0 and measured_batches >= args.max_batches:
            break
        if args.max_samples > 0 and measured_samples >= args.max_samples:
            break

        inputs = build_inputs_from_batch(batch, args.task)
        if args.max_samples > 0:
            remain = args.max_samples - measured_samples
            if remain <= 0:
                break
            if len(inputs) > remain:
                inputs = inputs[:remain]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_t = time.perf_counter()

        input_ids = model.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_lenth,
        ).input_ids.to(device)

        outputs = model.molt5_m.generate(
            input_ids,
            num_beams=args.num_beams,
            max_length=args.generate_max_length,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_t

        output_tokens = count_non_pad_tokens(outputs, pad_token_id)
        if args.token_scope == "total":
            input_tokens = count_non_pad_tokens(input_ids, pad_token_id)
            batch_tokens = output_tokens + input_tokens
        else:
            batch_tokens = output_tokens

        if batch_idx >= args.warmup_batches:
            current_batch_samples = len(inputs)
            total_elapsed += elapsed
            total_tokens += batch_tokens
            measured_batches += 1
            measured_samples += current_batch_samples

    if measured_batches == 0:
        raise RuntimeError(
            "No measured batches. Please reduce --warmup_batches or increase dataset size/--max_batches."
        )

    token_per_sec = total_tokens / max(total_elapsed, 1e-12)
    sample_per_min = measured_samples / max(total_elapsed, 1e-12) * 60.0

    return {
        "measured_batches": measured_batches,
        "measured_samples": measured_samples,
        "elapsed_sec": total_elapsed,
        "total_tokens": total_tokens,
        "token_per_sec": token_per_sec,
        "sample_per_min": sample_per_min,
    }


def infer_method_names(checkpoints, method_names):
    if method_names is not None and len(method_names) > 0:
        if len(method_names) != len(checkpoints):
            raise ValueError("The number of --method_names must match the number of --checkpoints.")
        return method_names

    names = []
    for ckpt in checkpoints:
        ckpt_path = Path(ckpt)
        parent = ckpt_path.parent.name
        stem = ckpt_path.stem
        names.append(f"{parent}/{stem}")
    return names


def resolve_checkpoints(args, config):
    if args.checkpoints and len(args.checkpoints) > 0:
        return args.checkpoints

    resume = config.get("resume_from_checkpoint", None)
    if isinstance(resume, str) and resume.strip():
        return [resume.strip()]
    if isinstance(resume, (list, tuple)) and len(resume) > 0:
        cks = [str(x).strip() for x in resume if str(x).strip()]
        if cks:
            return cks

    raise ValueError(
        "No --checkpoints were provided, and the config does not contain a usable "
        "resume_from_checkpoint value."
    )


def print_result_table(results):
    print("\n===== Inference Speed Benchmark =====")
    print(
        f"{'method':40s} {'token/s':>12s} {'sample/min':>12s} "
        f"{'samples':>10s} {'tokens':>12s} {'sec':>10s}"
    )
    for item in results:
        print(
            f"{item['method'][:40]:40s} "
            f"{item['token_per_sec']:12.2f} "
            f"{item['sample_per_min']:12.2f} "
            f"{item['measured_samples']:10d} "
            f"{item['total_tokens']:12d} "
            f"{item['elapsed_sec']:10.2f}"
        )


def main():
    parser_boot = argparse.ArgumentParser(add_help=False)
    parser_boot.add_argument("--config", type=str, default="_yamls/Eval_Atomas.yaml")
    boot_args, _ = parser_boot.parse_known_args()

    with open(boot_args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    parser = build_parser(config)
    args = parser.parse_args()
    checkpoints = resolve_checkpoints(args, config)

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataloader = build_test_loader(args)
    method_names = infer_method_names(checkpoints, args.method_names)
    from models.atomas import Atomas

    results = []
    for method_name, ckpt in zip(method_names, checkpoints):
        print(f"\n[Benchmark] method={method_name}")
        print(f"[Benchmark] checkpoint={ckpt}")

        model = Atomas(args=args)
        state_dict = load_checkpoint_state_dict(ckpt)
        model.load_state_dict(state_dict, strict=True)

        speed = benchmark_checkpoint(model, dataloader, args, device)
        speed["method"] = method_name
        speed["checkpoint"] = ckpt
        results.append(speed)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_result_table(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
