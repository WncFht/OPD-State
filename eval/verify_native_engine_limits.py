"""Match actual native worker scheduler limits to their train/eval engines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from native_devices import matches_device


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, devices, *, training_sequences, evaluation_sequences):
    directory = Path(directory)
    if set(devices) != set(range(4)):
        raise ValueError("Engine-limit acceptance requires the physical GPU 0–3 inventory")
    candidates = list(directory.glob("checkpoints/**/native-trajectories/manifest.json"))
    if len(candidates) != 1:
        raise ValueError("Expected one native engine/trajectory manifest")
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text())
    batch_path = manifest_path.with_name("batches.jsonl")
    batches = [json.loads(line) for line in batch_path.read_text().splitlines()]
    engines = manifest["engines"]
    if "eval:current_student" not in engines or not any(not b["validate"] for b in batches):
        raise ValueError("Engine-limit acceptance needs both training and pure-student evaluation")
    observed, counts = defaultdict(set), defaultdict(int)
    fields = {
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "max_model_len": "context_limit",
    }
    for batch in batches:
        receipts = batch.get("generation_execution", [])
        if not receipts:
            raise ValueError("Actual engine scheduler receipts are missing")
        for receipt in receipts:
            slot, rank = receipt["slot"], receipt["replica_rank"]
            evaluation = slot == "eval:current_student"
            if evaluation != bool(batch["validate"]) or slot not in engines or rank not in devices:
                raise ValueError("An engine executed in the wrong stage, slot or physical replica")
            expected = evaluation_sequences if evaluation else training_sequences
            if engines[slot]["max_num_seqs"] != expected:
                raise ValueError("Requested engine concurrency differs from the acceptance setting")
            if not matches_device(receipt["device_uuid"], devices[rank]):
                raise ValueError("Actual scheduler receipt came from an unexpected physical GPU")
            for actual, declared in fields.items():
                value = receipt.get(actual)
                if type(value) is not int or value < 1 or value != engines[slot][declared]:
                    raise ValueError("Actual worker scheduler limit differs from its engine declaration: " + actual)
            observed[slot].add(rank)
            counts[slot] += 1
    if observed["eval:current_student"] != set(devices):
        raise ValueError("Pure-student validation did not retain scheduler receipts from every GPU")
    if set(observed) != set(engines):
        raise ValueError("A declared engine was never exercised")
    return {
        "scope": "Actual worker scheduler/config receipts; no kernel timing or source-quality claim",
        "passed": True,
        "training_sequences": training_sequences,
        "evaluation_sequences": evaluation_sequences,
        "observed_ranks": {slot: sorted(ranks) for slot, ranks in observed.items()},
        "expected_devices": devices,
        "scheduler_receipt_counts": dict(counts),
        "engines": engines,
        "prepared_data_manifest_sha256": manifest.get("prepared_data_manifest_sha256"),
        "manifest_sha256": digest(manifest_path),
        "batches_sha256": digest(batch_path),
        "verifier_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--training-sequences", type=int, required=True)
    parser.add_argument("--evaluation-sequences", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = {int(key): value for key, value in json.loads(args.devices.read_text()).items()}
    report = verify(
        args.run, devices, training_sequences=args.training_sequences, evaluation_sequences=args.evaluation_sequences
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "engines": len(report["engines"])}))


if __name__ == "__main__":
    main()
