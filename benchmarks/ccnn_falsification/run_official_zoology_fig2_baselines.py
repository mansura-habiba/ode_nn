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
    p.add_argument("--snapshot-dir", type=Path, required=True,
                   help="Frozen train/val/cal/test tensor directory emitted by CCNN paper notebook; "
                        "refuse to run with independently regenerated data")
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
                   help="Two epochs on the same frozen benchmark snapshots: engineering check only")
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


def read_frozen_snapshot(snapshot_dir, seed, split, length, pairs):
    """Return exact MQAR inputs/labels from one frozen CCNN snapshot, checking SHA."""
    import torch
    root = Path(snapshot_dir).expanduser().resolve()
    manifest_path = root / f"seed{seed}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing shared benchmark manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest["seed"]) != int(seed) or manifest["schema"] != "frozen_mqar_v1":
        raise ValueError("Wrong frozen benchmark seed or schema")
    if manifest["upstream_commit"] != UPSTREAM_SHA:
        raise ValueError("Frozen generator revision does not match pinned Zoology model")
    matches = [x for x in manifest["records"]
               if x["split"] == split and x["length"] == length and x["pairs"] == pairs]
    if len(matches) != 1:
        raise ValueError(f"Exactly one {split} L={length} K={pairs} snapshot required")
    rec = matches[0]
    # Read by filename relative to the supplied snapshot directory, not by a
    # Colab absolute path persisted in the manifest.
    path = root / rec["filename"]
    obj = torch.load(path, map_location="cpu", weights_only=True)
    x = obj["inputs"].contiguous().long()
    y = obj["labels"].contiguous().long()
    actual = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    if actual != rec["sha256"] or x.shape != y.shape:
        raise ValueError(f"Frozen tensor changed or shape mismatch: {path}")
    if x.shape != (rec["num_examples"], length):
        raise ValueError("Sample count mismatch to frozen manifest")
    valid = y != -100
    if not valid.any() or not ((y[valid] >= 4096) & (y[valid] < 8192)).all():
        raise ValueError("The stored data do not obey the original MQAR label contract")
    return x, y, actual


def frozen_dataloader(snapshot_dir, seed, split, length, pairs, batch_size):
    """Adapt shared frozen tensors to Zoology's original batch-triple interface."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    x, y, digest = read_frozen_snapshot(snapshot_dir, seed, split, length, pairs)

    class Batches(Dataset):
        def __len__(self):
            return (len(x) + batch_size - 1) // batch_size

        def __getitem__(self, index):
            sl = slice(index * batch_size, (index + 1) * batch_size)
            a, b = x[sl], y[sl]
            metadata = [{"num_kv_pairs": pairs, "input_seq_len": length}] * len(a)
            return a, b, metadata

    return DataLoader(Batches(), batch_size=None, num_workers=0), digest


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
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    # The CCNN notebook's frozen manifest is the sole admissible data source.
    # Dataset generation is not performed inside the published-model runner.
    split_proofs = {}
    for split in ("train", "val", "cal", "test"):
        _, _, split_proofs[split] = read_frozen_snapshot(
            snapshot_dir, args.seed, split, args.sequence_length,
            CONDITIONS[args.sequence_length])
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
    original_prepare_data = module.prepare_data
    original_validation = module.Trainer.test
    active_run = {"id": None, "path": None,
                  "best_validation_loss": float("inf"), "best_epoch": None,
                  "checkpoint": None}
    completed = []

    def select_validation_checkpoint(trainer, epoch_idx):
        # Native Zoology test() receives our frozen VALIDATION loader, not TEST.
        # Intercept its validation result to store the best checkpoint.
        result = original_validation(trainer, epoch_idx)
        metric = float(result["valid/loss"])
        if metric < active_run["best_validation_loss"]:
            active_run["best_validation_loss"] = metric
            active_run["best_epoch"] = int(epoch_idx + 1)
            checkpoint = out / f"{active_run['id']}.best_val_state.pt"
            import torch
            torch.save({k: v.detach().cpu() for k,v in trainer.model.state_dict().items()},checkpoint)
            active_run["checkpoint"] = str(checkpoint)
        return result

    def prepare_shared_data(_config):
        train_dl, train_hash = frozen_dataloader(
            snapshot_dir, args.seed, "train", args.sequence_length,
            CONDITIONS[args.sequence_length], args.batch_size)
        val_dl, val_hash = frozen_dataloader(
            snapshot_dir, args.seed, "val", args.sequence_length,
            CONDITIONS[args.sequence_length], args.batch_size)
        assert train_hash == split_proofs["train"]
        assert val_hash == split_proofs["val"]
        # IMPORTANT: native Zoology's Trainer calls this second loader 'test'
        # at every epoch. We supply ONLY the separate validation split; the
        # actual locked TEST is never exposed during training or LR selection.
        return train_dl, val_dl

    class MetricsCSV:
        def __init__(self, config):
            self.run_id = config.run_id
            self.rows = []
            self.path = out / f"{self.run_id}.csv"
            active_run["id"] = self.run_id
            active_run["path"] = str(self.path)
            active_run["best_validation_loss"] = float("inf")
            active_run["best_epoch"] = None
            active_run["checkpoint"] = None

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
    module.prepare_data = prepare_shared_data
    module.Trainer.test = select_validation_checkpoint
    try:
        for selected_cfg in selected:
            cfg = copy.deepcopy(selected_cfg)
            cfg.seed = args.seed
            cfg.data.seed = args.seed
            cfg.data.cache_dir = None
            cfg.data.batch_size = (args.batch_size, args.batch_size)
            cfg.max_epochs = args.epochs or (2 if args.pilot else 64)
            # These metadata counts reflect the frozen files, not a newly
            # generated source. The second loader is exclusively VALIDATION.
            cfg.data.train_configs[0].num_examples = read_frozen_snapshot(
                snapshot_dir,args.seed,"train",args.sequence_length,CONDITIONS[args.sequence_length])[0].shape[0]
            cfg.data.test_configs[0].num_examples = read_frozen_snapshot(
                snapshot_dir,args.seed,"val",args.sequence_length,CONDITIONS[args.sequence_length])[0].shape[0]

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
                "num_validation": cfg.data.test_configs[0].num_examples,
                "num_locked_test": read_frozen_snapshot(snapshot_dir,args.seed,
                    "test",args.sequence_length,CONDITIONS[args.sequence_length])[0].shape[0],
                "frozen_split_sha256":split_proofs,
                "uses_identical_ccnn_snapshot":True,
                "pilot": bool(args.pilot),
                "is_full_original_paper_reproduction": False,
                "change_from_original": (
                    "Original published model weights/architecture and training code, "
                    "but CCNN notebook frozen datasets and its own held-out validation "
                    "split replace original Zoology data loader. Batch size, epoch "
                    "limit, and condition range can differ from the paper."
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
                    "Native Zoology trainer is now fed frozen validation only; "
                    "locked TEST is reserved and was not evaluated or used for "
                    "epoch/LR selection in this baseline-runner phase."
                )
                report["locked_test_evaluated"] = False
                report["best_validation_loss"] = (
                    active_run["best_validation_loss"]
                    if active_run["checkpoint"] else None
                )
                report["best_validation_epoch"] = active_run["best_epoch"]
                report["validation_checkpoint"] = active_run["checkpoint"]
                meta.write_text(json.dumps(report, indent=2))
                if report["status"] == "COMPLETED" and active_run["checkpoint"]:
                    completed.append({"config":copy.deepcopy(cfg),
                                      "manifest": str(meta),
                                      "checkpoint": active_run["checkpoint"],
                                      "validation_loss": active_run["best_validation_loss"]})
                print("SAVED", str(meta), "status", report["status"], flush=True)
    finally:
        module.WandbLogger = originals
        module.prepare_data = original_prepare_data
        module.Trainer.test = original_validation

    # Select the BEST LR/epoch by validation loss alone. Only then open
    # the locked test split, exactly once for the selected candidate.
    if not completed:
        raise RuntimeError("No completed validation-selected Zoology checkpoint")
    winner = min(completed, key=lambda record: record["validation_loss"])
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native = module.LanguageModel(winner["config"].model).to(torch_device)
    native.load_state_dict(torch.load(winner["checkpoint"], map_location="cpu",
                                      weights_only=True))
    native.eval()
    test_dl, test_hash = frozen_dataloader(
        snapshot_dir, args.seed, "test", args.sequence_length,
        CONDITIONS[args.sequence_length], args.batch_size)
    assert test_hash == split_proofs["test"]
    import torch.nn.functional as F
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for token_ids, target_ids, _ in test_dl:
            token_ids = token_ids.to(torch_device)
            target_ids = target_ids.to(torch_device)
            label_mask = target_ids != -100
            logits = native(token_ids)[label_mask]
            labels = target_ids[label_mask]
            correct += int((logits.argmax(-1) == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(F.cross_entropy(
                logits.float(), labels, reduction="sum").item())
    if total == 0:
        raise RuntimeError("Locked MQAR test has no supervised queries")
    report_path = Path(winner["manifest"])
    report = json.loads(report_path.read_text())
    report["locked_test_evaluated"] = True
    report["locked_test_accuracy"] = correct / total
    report["locked_test_nll"] = loss_sum / total
    report["locked_test_supervised_queries"] = total
    report["selected_by"] = "minimum validation loss across learning rates and epochs"
    report["selected_validation_loss"] = winner["validation_loss"]
    report["selected_lr"] = float(winner["config"].learning_rate)
    report["selected_checkpoint"] = winner["checkpoint"]
    report_path.write_text(json.dumps(report, indent=2))
    summary = {k: report[k] for k in (
        "original_model_name", "sequence_length", "pairs", "width", "seed",
        "selected_lr", "selected_validation_loss",
        "locked_test_accuracy", "locked_test_nll", "locked_test_supervised_queries"
    )}
    (out / f"locked_test_{args.model}_L{args.sequence_length}_d{args.width}_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2))
    print("VALIDATION_SELECTED_LOCKED_TEST", json.dumps(summary, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
