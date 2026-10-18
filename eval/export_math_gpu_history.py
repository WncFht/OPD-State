"""Export W&B's sampled whole-GPU telemetry for physical GPU indices 0-3.

Training/validation boundaries are estimates from W&B log timestamps and the
already audited phase durations. The event API samples data, so maxima here
are observed sampled maxima, not guaranteed continuous device peaks.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plots", Path(__file__).with_name("plot_math_experiment.py"))
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)
FIELDS = ("memoryAllocatedBytes", "memoryAllocated", "gpu", "powerWatts")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/results/resources/wandb-gpu")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    import wandb

    api = wandb.Api(timeout=40)
    report = {
        "exporter_sha256": sha256(Path(__file__)),
        "wandb_version": wandb.__version__,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_samples": args.samples,
        "scope": "W&B sampled system stream, GPU indices 0-3 only; phase boundaries approximate; not continuous peaks",
        "runs": {},
    }
    for arm, directory in PLOTS.RUNS.items():
        local = ROOT / "docs/experiment/results" / directory
        summary = json.loads((local / "summary.json").read_text())
        assert summary["validation_steps"] == [50]
        parts = urlparse(summary["wandb_url"]).path.strip("/").split("/")
        run = api.run("/".join((parts[0], parts[1], parts[-1])))
        assert run.state == "finished"
        timestamps = {
            int(row["training/global_step"]): float(row["_timestamp"])
            for row in run.scan_history(keys=["training/global_step", "_timestamp"], page_size=1000)
        }
        assert set(timestamps) == set(range(1, 51))
        with (local / "training-1-50.csv").open() as stream:
            training = list(csv.DictReader(stream))
        train_start = timestamps[1] - float(training[0]["timing_s/step"])
        validation_end = timestamps[50]
        validation_start = validation_end - float(training[-1]["timing_s/testing"])
        assert train_start < validation_start < validation_end
        events = run.history(samples=args.samples, stream="system", pandas=False)
        rows = []
        for event in events:
            selected = {
                f"gpu{g}.{key}": event[f"system.gpu.{g}.{key}"]
                for g in range(4)
                for key in FIELDS
                if f"system.gpu.{g}.{key}" in event
            }
            if not selected:
                continue
            timestamp = float(event["_timestamp"])
            phase = (
                "initialization"
                if timestamp < train_start
                else "training"
                if timestamp < validation_start
                else "validation"
                if timestamp <= validation_end
                else "after_validation"
            )
            rows.append({"timestamp": timestamp, "runtime": event.get("_runtime"), "phase_hint": phase, **selected})
        assert rows, (arm, "No GPU telemetry")
        stats = {}
        for phase in ("all", "initialization", "training", "validation", "after_validation"):
            subset = [row for row in rows if phase == "all" or row["phase_hint"] == phase]
            per_gpu = {}
            for g in range(4):
                memory = [
                    float(row[f"gpu{g}.memoryAllocatedBytes"]) / 2**30
                    for row in subset
                    if row.get(f"gpu{g}.memoryAllocatedBytes") is not None
                ]
                utilization = [float(row[f"gpu{g}.gpu"]) for row in subset if row.get(f"gpu{g}.gpu") is not None]
                assert all(math.isfinite(v) and 0 <= v <= 100 for v in memory + utilization)
                if memory:
                    per_gpu[str(g)] = {
                        "memory_samples": len(memory),
                        "max_used_GiB": max(memory),
                        "mean_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
                    }
            stats[phase] = per_gpu
        output = args.output / parts[-1]
        output.mkdir(parents=True, exist_ok=True)
        files = {}
        for start in range(0, len(rows), 900):
            path = output / f"gpu-samples-{start + 1}-{min(start + 900, len(rows))}.csv"
            with path.open("w", newline="") as stream:
                fields = ["timestamp", "runtime", "phase_hint", *(f"gpu{g}.{k}" for g in range(4) for k in FIELDS)]
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows[start : start + 900])
            files[str(path.relative_to(args.output))] = sha256(path)
        report["runs"][arm] = {
            "id": parts[-1],
            "url": summary["wandb_url"],
            "system_events_returned": len(events),
            "gpu_events_retained": len(rows),
            "first_gpu_timestamp": min(r["timestamp"] for r in rows),
            "last_gpu_timestamp": max(r["timestamp"] for r in rows),
            "phase_boundaries": {
                "training_start_estimate": train_start,
                "validation_start_estimate": validation_start,
                "validation_end_estimate": validation_end,
            },
            "local_training_csv_sha256": sha256(local / "training-1-50.csv"),
            "sampled_gpu_statistics": stats,
            "outputs_sha256": files,
        }
        print(
            json.dumps({"source": arm, "events": len(events), "gpu_events": len(rows), "sampled_memory": stats}),
            flush=True,
        )
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
