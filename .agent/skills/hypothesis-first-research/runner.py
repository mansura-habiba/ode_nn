#!/usr/bin/env python3
"""Bounded, stdlib-only empirical experiment runner (Python 3.10+).

Usage: python runner.py protocol.json --ledger results.jsonl
All registered methods must invoke the SAME sealed evaluator via {evaluator}.
Evaluator must emit a JSON metrics object as its final stdout line.
No auto-tuning, reruns, or scientific verdicts.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def verify(spec, root):
    ensure(spec.get("version") == 1, "version must equal 1")
    ensure(spec.get("approved") is True, "human-approved protocol required")
    ensure(spec.get("stage") in ("pilot", "main"), "stage must be pilot or main")
    ensure(spec.get("hypothesis_id") in ("H1", "H2", "H3", "H4"), "hypothesis_id must be H1-H4")
    ensure(bool(spec.get("anchor")), "published protocol anchor required")
    ensure(isinstance(spec.get("git_commit"), str) and len(spec["git_commit"]) >= 7, "git_commit required")
    ensure(bool(spec.get("primary_metric")), "primary_metric required")
    ensure(spec.get("direction") in ("min", "max"), "direction must be min or max")
    ensure(type(spec.get("minimum_effect")) in (int, float), "minimum_effect required")
    budget = spec.get("budget_seconds")
    ensure(type(budget) is int and 1 <= budget <= 86400, "budget_seconds must be 1..86400")
    seeds = spec.get("seeds")
    ensure(isinstance(seeds, list) and bool(seeds) and all(type(s) is int and s >= 0 for s in seeds)
           and len(seeds) == len(set(seeds)), "unique nonnegative integer seeds required")
    splits = spec.get("split_ids", {})
    ensure(all(isinstance(splits.get(k), list) and splits[k] for k in ("train", "validation", "test")),
           "split_ids must contain train, validation and test IDs")
    groups = [set(str(x) for x in splits[k]) for k in ("train", "validation", "test")]
    ensure(all(len(splits[k]) == len(groups[i]) for i, k in enumerate(("train", "validation", "test"))),
           "duplicate split IDs")
    ensure(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
           "train/validation/test overlap")
    evaluator = (root / spec.get("evaluator", "")).resolve()
    ensure(evaluator.is_file() and evaluator.is_relative_to(root.resolve()), "local evaluator required")
    ensure(sha(evaluator) == spec.get("evaluator_sha256"), "evaluator SHA-256 mismatch")
    files = spec.get("data_files")
    ensure(isinstance(files, dict) and bool(files), "data_files hashes required")
    for name, expected in files.items():
        path = (root / name).resolve()
        ensure(path.is_file() and path.is_relative_to(root.resolve()) and sha(path) == expected,
               "data SHA-256 mismatch: " + name)
    methods = spec.get("methods")
    ensure(isinstance(methods, list) and len(methods) >= 2, "at least two methods required")
    names = [m.get("name") for m in methods]
    ensure(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names),
           "unique named methods required")
    ensure(any(m.get("baseline") is True for m in methods), "baseline required")
    for m in methods:
        cmd = m.get("command")
        ensure(isinstance(cmd, list) and bool(cmd) and all(isinstance(x, str) and x for x in cmd),
               "commands must be argv arrays")
        ensure(any("{seed}" in x for x in cmd), "method command must include {seed}")
        ensure(any("{evaluator}" in x for x in cmd), "method command must invoke {evaluator}")
        ensure(isinstance(m.get("config_sha256"), str) and len(m["config_sha256"]) == 64,
               "config_sha256 required")
    return evaluator


def metrics_from_stdout(stdout, metric):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            data = value.get("metrics", value)
            ensure(isinstance(data, dict) and type(data.get(metric)) in (int, float),
                   "missing numeric primary metric")
            return data
    raise ValueError("evaluator must print a JSON object with metrics")


def run(specfile, ledger):
    root = specfile.resolve().parent
    spec = json.loads(specfile.read_text(encoding="utf-8"))
    evaluator = verify(spec, root)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ledger = ledger.resolve()
    ensure(ledger not in (specfile.resolve(), evaluator), "ledger cannot overwrite spec or evaluator")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ensure(json.loads(line).get("fingerprint") != fingerprint,
                       "protocol already attempted in ledger; no silent reruns")
    with ledger.open("a", encoding="utf-8") as log:
        for method in spec["methods"]:
            for seed in spec["seeds"]:
                verify(spec, root)
                command = [x.replace("{seed}", str(seed)).replace("{evaluator}", str(evaluator))
                           for x in method["command"]]
                env = dict(os.environ, PYTHONHASHSEED=str(seed), RESEARCH_SEED=str(seed))
                start = time.monotonic()
                status, metrics, error = "CRASH", {}, None
                try:
                    proc = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True,
                                          timeout=spec["budget_seconds"], check=False)
                    if proc.returncode:
                        error = (proc.stderr or proc.stdout)[-1200:]
                    else:
                        metrics = metrics_from_stdout(proc.stdout, spec["primary_metric"])
                        status = "PASS"
                except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    error = str(exc)[-1200:]
                try:
                    verify(spec, root)
                except ValueError as exc:
                    status, error = "FAIL", "sealed inputs changed: " + str(exc)
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                    "stage": spec["stage"], "hypothesis_id": spec["hypothesis_id"],
                    "anchor": spec["anchor"], "git_commit": spec["git_commit"],
                    "method": method["name"], "baseline": bool(method.get("baseline")),
                    "config_sha256": method["config_sha256"], "seed": seed, "status": status,
                    "primary_metric": spec["primary_metric"], "metrics": metrics if status == "PASS" else {},
                    "duration_seconds": round(time.monotonic() - start, 3), "error": error,
                    "python": sys.version.split()[0], "platform": platform.platform()
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(method["name"], "seed", seed, status)
                if status != "PASS":
                    print("Stopped: inspect ledger before proposing any rerun", file=sys.stderr)
                    return 1
    print("Registered runs completed. Scientific adjudication belongs to P5, not this runner.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        sys.exit(run(args.protocol, args.ledger))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("Blocked:", exc, file=sys.stderr)
        sys.exit(2)
