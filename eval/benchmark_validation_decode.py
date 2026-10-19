"""Compare padded and effective-token decoding on preserved validation IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("rows must be positive")
    import torch
    from transformers import AutoTokenizer

    from opd_state.integrations.verl.validation_data import decode_effective_tokens

    if torch.cuda.is_initialized():
        raise ValueError("Decode benchmark must remain CPU-only")
    records = []
    digest = hashlib.sha256()
    with args.records.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if row["state_source"] == "on_validation_states":
                records.append(row)
    selected = sorted(random.Random(42).sample(range(len(records)), min(args.rows, len(records))))
    records = [records[index] for index in selected]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    output_ids = torch.full((len(records), 31744), tokenizer.pad_token_id, dtype=torch.long)
    input_ids = torch.full((len(records), 1024), tokenizer.pad_token_id, dtype=torch.long)
    output_mask, input_mask = (
        torch.zeros_like(output_ids, dtype=torch.bool),
        torch.zeros_like(input_ids, dtype=torch.bool),
    )
    for index, row in enumerate(records):
        p, r = len(row["prompt_ids"]), len(row["response_ids"])
        if p > 1024 or r > 31744:
            raise ValueError("Response workload exceeds the declared original protocol")
        output_ids[index, :r] = torch.tensor(row["response_ids"])
        output_mask[index, :r] = True
        input_ids[index, -p:] = torch.tensor(row["prompt_ids"])
        input_mask[index, -p:] = True

    def decode(effective):
        if effective:
            return (
                decode_effective_tokens(tokenizer, input_ids, input_mask),
                decode_effective_tokens(tokenizer, output_ids, output_mask),
            )
        return tuple(
            [tokenizer.decode(row, skip_special_tokens=True) for row in ids] for ids in (input_ids, output_ids)
        )

    tokenizer.decode(records[0]["response_ids"], skip_special_tokens=True)
    timings, reference = [], None
    for effective in (False, True, True, False):
        tick = time.perf_counter()
        result = decode(effective)
        seconds = time.perf_counter() - tick
        checksum = hashlib.sha256(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if reference is None:
            reference = checksum
        if checksum != reference:
            raise ValueError("Effective decoding changed the produced text")
        timings.append({"effective": effective, "seconds": seconds, "text_sha256": checksum})
    receipt = {
        "scope": "CPU tokenizer decoding only; original padded decode versus masked effective positions, ABBA order",
        "passed": True,
        "rows": len(records),
        "source_data_sha256": digest.hexdigest(),
        "selected_response_indices": selected,
        "measurements": timings,
        "effective_response_positions": int(output_mask.sum()),
        "padded_response_positions": output_ids.numel(),
        "effective_prompt_positions": int(input_mask.sum()),
        "padded_prompt_positions": input_ids.numel(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "helper_sha256": hashlib.sha256(Path(decode_effective_tokens.__code__.co_filename).read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "rows": len(records), "timings": [r["seconds"] for r in timings]}))


if __name__ == "__main__":
    main()
