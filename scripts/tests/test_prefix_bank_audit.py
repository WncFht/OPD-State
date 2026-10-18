"""Reject re-signed bank records with wrong prefixes, sources or worker work."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import audit_prefix_bank as AUDIT


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root):
    questions = [
        {"source": name, "uid": f"{name}:{i}", "prompt_ids": [1, 10 + b * 2 + i]}
        for b, name in enumerate(("MATH-500", "AMC23", "AIME25"))
        for i in range(2)
    ]
    prepared = {
        "schema": "fixed-prefix-input-v1",
        "tokenizer_signature": "toy-tokenizer",
        "vocab_size": 32,
        "sampling": {"temperature": 1, "top_p": 1, "top_k": -1},
        "response_cap": 2,
        "per_benchmark": 2,
        "positions_per_response": 8,
        "selection_seed": 42,
        "eos_token_id": 2,
        "models": {
            role: {"path": f"/{role}", "fingerprint": letter * 64}
            for role, letter in zip(AUDIT.ROLES, ("a", "b"), strict=True)
        },
        "questions": questions,
    }
    save(root / "original-input.json", prepared)
    save(root / "input.json", prepared)
    source_path = root / "source/verl/toy.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("pass\n")
    source = {
        "base_commit": "a" * 40,
        "source_kind": "git_archive",
        "files": {"verl/toy.py": AUDIT.digest(source_path)},
    }
    save(root / "source-manifest.json", source)
    devices = {
        str(gpu): {
            key: f"GPU-card{gpu}" for key in ("nvml_uuid", "cuda_uuid", "cuda_uuid_v2")
        }
        for gpu in range(4)
    }
    save(root / "device-identities.json", devices)
    initialized, records, generation = [], [], {}
    for role, state in AUDIT.ROLES.items():
        raw = []
        for question in questions:
            seed = int.from_bytes(
                hashlib.sha256(
                    repr((42, ("bank-positions", question["uid"], role))).encode()
                ).digest()[:4],
                "big",
            )
            model = prepared["models"][role]
            segment = {
                "identities": [[role, model["path"], model["fingerprint"]]],
                "start": 0,
                "end": 2,
                "teacher_weight": None,
                "finish_reason": "stop",
            }
            tokens = [3 if role == "initial_student" else 4, 2]
            record = {
                "state_source": state,
                "uid": question["uid"],
                "source": question["source"],
                "sample": 0,
                "prompt_ids": question["prompt_ids"],
                "response_ids": tokens,
                "termination": "eos",
                "native_segment": segment,
                "position_seed": seed,
                "positions": [0, 1],
                "weights": [1, 1],
                "strata": [
                    {"position": 0, "begin": 0, "end": 1},
                    {"position": 1, "begin": 1, "end": 2},
                ],
            }
            records.append(record)
            raw.append(
                {
                    "request": {
                        "prompt_id": question["uid"],
                        "prompt_ids": question["prompt_ids"],
                        "visit": 0,
                        "sample": 0,
                    },
                    "token_ids": tokens,
                    "termination": "eos",
                    "segments": [segment],
                }
            )
        (root / f"{role}-trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw)
        )
        calls = []
        for gpu in range(4):
            initialized.append(
                {
                    "slot": role,
                    "physical_gpu": gpu,
                    "expected": devices[str(gpu)],
                    "observed": {"device_uuid": f"GPU-card{gpu}"},
                }
            )
            ids = [f"{role}:{i}" for i in range(len(questions)) if i % 4 == gpu]
            calls.append(
                {
                    "slot": role,
                    "requests": len(ids),
                    "request_ids": ids,
                    "replica_rank": gpu,
                    "device_uuid": f"GPU-card{gpu}",
                }
            )
        execution = {
            "execution": calls,
            "frozen_verifications": [
                {
                    "slot": role,
                    "frozen_digests": {role: prepared["models"][role]["fingerprint"]},
                }
                for _ in range(4)
            ],
            "timing": {"seconds": 1.0},
        }
        generation[role] = execution
        save(root / f"{role}-execution.json", execution)
    (root / "engine-initialization.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in initialized)
    )
    bank = {
        key: prepared[key]
        for key in (
            "tokenizer_signature",
            "vocab_size",
            "sampling",
            "response_cap",
            "per_benchmark",
        )
    }
    bank.update(
        schema="fixed-prefix-bank-v1",
        input_sha256=AUDIT.digest(root / "input.json"),
        question_ids_sha256=hashlib.sha256(
            json.dumps([q["uid"] for q in questions], separators=(",", ":")).encode()
        ).hexdigest(),
        initial_student_fingerprint="a" * 64,
        teacher_fingerprint="b" * 64,
        records=records,
        generation_provenance={
            "source_manifest_sha256": AUDIT.digest(root / "source-manifest.json"),
            "physical_gpu_identities": devices,
            "generation": generation,
            "driver_dependencies_sha256": {},
            "graph": False,
            "cold_seconds": {},
        },
    )
    save(root / "prefix-bank.json", bank)
    return bank


class PrefixBankAuditTests(unittest.TestCase):
    def test_paired_positions_are_not_all_distinct_token_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            result = AUDIT.audit(root, root / "original-input.json", per_benchmark=2)
            self.assertEqual(result["responses"], 12)
            self.assertEqual(result["selected_positions"], 24)
            self.assertEqual(result["distinct_hashed_token_prefixes"], 18)

    def test_changed_semantics_cannot_pass_by_rewriting_matching_receipts(self):
        for problem in (
            "position",
            "eos",
            "gpu",
            "frozen",
            "rpc",
            "model",
            "missing",
            "input",
            "source",
        ):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                bank = fixture(root)
                if problem == "position":
                    bank["records"][0]["weights"][0] = 2
                elif problem in {"eos", "model"}:
                    path = root / "initial_student-trajectories.jsonl"
                    raw = [json.loads(line) for line in path.read_text().splitlines()]
                    if problem == "eos":
                        raw[0]["token_ids"] = [2, 2]
                        bank["records"][0]["response_ids"] = [2, 2]
                    else:
                        raw[0]["segments"][0]["identities"][0][2] = "b" * 64
                        bank["records"][0]["native_segment"] = raw[0]["segments"][0]
                    path.write_text("".join(json.dumps(row) + "\n" for row in raw))
                elif problem == "gpu":
                    path = root / "engine-initialization.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["observed"]["device_uuid"] = "GPU-foreign"
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                elif problem in {"frozen", "rpc"}:
                    execution = bank["generation_provenance"]["generation"][
                        "initial_student"
                    ]
                    if problem == "frozen":
                        execution["frozen_verifications"][0]["frozen_digests"][
                            "initial_student"
                        ] = "f" * 64
                    else:
                        execution["execution"][1]["request_ids"][0] = execution[
                            "execution"
                        ][0]["request_ids"][0]
                    save(root / "initial_student-execution.json", execution)
                elif problem == "missing":
                    bank["records"].pop()
                elif problem == "input":
                    save(root / "original-input.json", {"changed": True})
                else:
                    (root / "source/verl/toy.py").write_text("changed\n")
                save(root / "prefix-bank.json", bank)
                with self.assertRaises(ValueError):
                    AUDIT.audit(root, root / "original-input.json", per_benchmark=2)


if __name__ == "__main__":
    unittest.main()
