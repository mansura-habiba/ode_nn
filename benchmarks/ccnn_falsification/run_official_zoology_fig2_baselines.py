"""Run *original Zoology ICLR-2024 Figure-2 model implementations* from a pinned clone.

This is an execution adapter, not a rewritten attention/Hyena/Based/Mamba
architecture. One model and one input condition per invocation. The original
Zoology trainer/data loader/optimizer/learning-rate scheduler are used.

The paper's full model comparison is very costly. On Colab, run narrower
controlled slices, record any changes to batch/data/epochs explicitly, and
never call a reduced run a reproduction of the full published results.

Official config: https://github.com/HazyResearch/zoology/blob/
1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/
paper_configs/iclr24_zoology_figure2/configs.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

UPSTREAM_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
OFFICIAL_CONFIG = Path("zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py")
MIXERS = ("attention", "hyena", "based", "mamba")
CONDITIONS = {64: 4, 128: 8, 256: 16}
ORIGINAL_LRS = (1e-4, 4.641588833612782e-4, 2.154434690031882e-3, 1e-2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoology-root", type=Path, required=True,
                   help="Directory containing cloned, pinned HazyResearch/zoology")
    p.add_argument("--model", choices=MIXERS, default="attention")
    p.add_argument("--sequence-length", type=int, choices=tuple(CONDITIONS), default=64)
    p.add_argument("--width", type=int, choices=(64, 128, 256, 512), default=128)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=None,
                   help="Default: 64 (published) or 2 with --pilot")
    p.add_argument("--learning-rate", default="all",
                   help="'all' => original four-rate search; or select one numeric LR")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Colab safety setting (NOT original paper batch size); same across comparators")
    p.add_argument("--pilot", action="store_true",
                   help="2,048 train/256 test examples and 2 epochs by default: debugging only")
    p.add_argument("--output", type=Path, default=Path("ccnn_zoology_baseline_results"))
    p.add_argument("--allow-unpinned", action="store_true",
                   help="Only for debugging; refuses a source revision mismatch otherwise")
    return p.parse_args(argv)


def upstream_commit(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def select_original_configs(configs, kind, length, width, lr):
    prefix = f"{kind}-seqlen{length}-dmodel{width}-lr"
    suffix = f"-kv{CONDITIONS[length]}"
    chosen = [cfg for cfg in configs
              if str(cfg.run_id).startswith(prefix)
              and str(cfg.run_id).endswith(suffix)
              and (lr == "all" or abs(float(cfg.learning_rate) - float(lr)) < 1e-12)]
    if not chosen:
        raise ValueError(f"No ORIGINAL ICLR24 config for {kind} L={length} d={width} LR={lr}.")
    # Mamba at width 512 is present in the official config, but may need special GPU extensions.
    return sorted(chosen, key=lambda cfg: float(cfg.learning_rate))


def main(argv=None):
    args = parse_args(argv)
    root = args.zoology_root.expanduser().resolve()
    config_path = root / OFFICIAL_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected original Zoology paper config: {config_path}")
    commit = upstream_commit(root)
    if commit != UPSTREAM_SHA and not args.allow_unpinned:
        raise RuntimeError(f"Zoology HEAD {commit} != expected {UPSTREAM_SHA}; checkout exact commit.")

    import torch
    import pandas as pd

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    orig_module = runpy.run_path(str(config_path))
    selected = select_original_configs(orig_module["configs"], args.model,
                                       args.sequence_length, args.width, args.learning_rate)

    # The native logger writes to W&B. This adapter only replaces logging
    # transport with CSV; native Zoology model, trainer and metric computation
    # are NOT modified.
    module = importlib.import_module("zoology.train")
    originals = module.WandbLogger
    active_run = {"id": None, "path": None}

    class MetricsCSV:
        def __init__(self, config):
            self.run_id = config.run_id
            self.rows = []
            self.path = out / f"{self.run_id}.csv"
            active_run["id"] = self.run_id
            active_run["path"] = str(self.path)

        def log_config(self, config):
            pass

        def log_model(self, model, config):
            pass

        def log(self, metrics):
            if "valid/accuracy" in metrics:
                self.rows.append({
                    "run_id": self.run_id,
                    "epoch": metrics.get("epoch"),
                    "valid_accuracy": metrics.get("valid/accuracy"),
                    "valid_loss": metrics.get("valid/loss"),
                })
                # Save after each validation epoch: interrupted Colab runs
                # still leave provenance and partial validation scores.
                pd.DataFrame(self.rows).to_csv(self.path, index=False)

        def finish(self):
            pass

    module.WandbLogger = MetricsCSV
    try:
        for selected_cfg in selected:
            cfg = copy.deepcopy(selected_cfg)
            cfg.seed = args.seed
            cfg.data.seed = args.seed
            cfg.data.cache_dir = str(cache_dir)
            cfg.data.batch_size = (args.batch_size, args.batch_size)
            cfg.max_epochs = args.epochs or (2 if args.pilot else 64)
            if args.pilot:
                cfg.data.train_configs[0].num_examples = 2048
                cfg.data.test_configs[0].num_examples = 256
                cfg.data.cache_dir = None

            cfg.run_id = f"{cfg.run_id}-seed{args.seed}-{'pilot' if args.pilot else 'control'}"
            report = {
                "status": "NOT_STARTED",
                "zoology_revision": commit,
                "paper_config_path": str(OFFICIAL_CONFIG),
                "original_model_name": args.model,
                "sequence_length": args.sequence_length,
                "pairs": CONDITIONS[args.sequence_length],
                "width": args.width,
                "learning_rate": float(cfg.learning_rate),
                "seed": args.seed,
                "epochs_cap": cfg.max_epochs,
                "batch_size": args.batch_size,
                "num_train": cfg.data.train_configs[0].num_examples,
                "num_test": cfg.data.test_configs[0].num_examples,
                "pilot": bool(args.pilot),
                "is_full_original_paper_reproduction": False,
                "change_from_original": (
                    "Original model and input/label code, BUT smaller batch and/or sample/epoch "
                    "conditions, one model width and one sequence configuration only. "
                    "Original Figure-2 model sweep not reproduced."
                ),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "CPU"),
                "python": platform.python_version(),
                "model_config": cfg.model.model_dump(mode="json"),
            }
            meta = out / f"{cfg.run_id}.manifest.json"
            meta.write_text(json.dumps(report, indent=2))
            print("RUNNING_NATIVE_ZOOLOGY", cfg.run_id,
                  "GPU", report["gpu_name"], "outputs", str(out), flush=True)
            start = time.perf_counter()
            try:
                module.train(cfg)
                report["status"] = "COMPLETED"
            except Exception as ex:
                report["status"] = "FAILED"
                report["error"] = f"{type(ex).__name__}: {ex}"
                raise
            finally:
                report["seconds"] = time.perf_counter()-start
                report["validation_csv"] = active_run["path"]
                if active_run["path"] and Path(active_run["path"]).exists():
                    vals = pd.read_csv(active_run["path"])
                    report["epochs_completed"] = len(vals)
                    if not vals.empty:
                        report["best_reported_valid_accuracy"] = float(vals.valid_accuracy.max())
                        report["last_reported_valid_accuracy"] = float(vals.valid_accuracy.iloc[-1])
                # IMPORTANT: Zoology's original harness evaluates its
                # configured TEST segment at every epoch. Any max-selected LR
                # score is therefore a reproduction-style descriptive score,
                # NOT an independently locked-test scientific comparison.
                report["selection_warning"] = (
                    "Original Zoology trainer reports its test_configs as valid; "
                    "use a separately locked test for our new CCNN comparison."
                )
                meta.write_text(json.dumps(report, indent=2))
                print("SAVED", str(meta), "status", report["status"], flush=True)
    finally:
        module.WandbLogger = originals


if __name__ == "__main__":
    main()
