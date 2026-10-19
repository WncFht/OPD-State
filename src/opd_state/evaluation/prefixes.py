"""Prefix coverage, stratified positions and collective work ordering."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixSample:
    position: int
    begin: int
    end: int

    @property
    def weight(self):
        return self.end - self.begin


def balanced_record_indices(records, world_size):
    """Put similar causal-forward lengths in the same collective round.

    Input is already DP-padded. Each rank receives the same number of rows;
    alternating the within-round rank assignment avoids always assigning the
    largest row to rank zero. Callers must invert this permutation before
    removing DP padding or checking original response identities.
    """
    if (
        type(world_size) is not int
        or world_size < 1
        or len(records) == 0
        or len(records) % world_size
    ):
        raise ValueError(
            "KL balancing requires nonempty padded records and a positive world size"
        )
    if world_size == 1:
        return tuple(range(len(records)))

    def cost(index):
        record = records[index]
        positions = record.get("positions")
        last = positions[-1] if positions else len(record["response_ids"]) - 1
        projected = len(positions) if positions else len(record["response_ids"])
        return -(len(record["prompt_ids"]) + last), -projected, index

    ordered = sorted(range(len(records)), key=cost)
    ranks = [[] for _ in range(world_size)]
    for round_id, begin in enumerate(range(0, len(records), world_size)):
        for rank in range(world_size):
            offset = rank if round_id % 2 == 0 else world_size - rank - 1
            ranks[rank].append(ordered[begin + offset])
    return tuple(index for rank in ranks for index in rank)


def sample_prefix_positions(length, *, max_positions=8, seed=0):
    """Always include action zero, then sample uniformly within disjoint strata.

    Positions index response actions, not input-token positions. Weighting by
    stratum size makes the estimated sum unbiased conditional on the trajectory.
    """
    if type(length) is not int or length < 1:
        raise ValueError("KL prefix sampling requires a nonempty effective response")
    if type(max_positions) is not int or max_positions < 2:
        raise ValueError("KL prefix sampling needs at least two available positions")
    if type(seed) is not int or seed < 0:
        raise ValueError("KL prefix sampling requires a nonnegative integer seed")
    if length <= max_positions:
        return tuple(PrefixSample(i, i, i + 1) for i in range(length))
    generator = random.Random(seed)
    strata = max_positions - 1
    result = [PrefixSample(0, 0, 1)]
    for i in range(strata):
        begin = 1 + (length - 1) * i // strata
        end = 1 + (length - 1) * (i + 1) // strata
        result.append(PrefixSample(generator.randrange(begin, end), begin, end))
    return tuple(result)


def estimate_trajectory_statistics(samples, values, *, length):
    """Estimate per-response sums and means; reject incomplete stratum coverage."""
    if type(length) is not int or length < 1 or not samples or not values:
        raise ValueError(
            "KL estimation requires positions, statistics and a valid effective length"
        )
    cursor = 0
    for sample in samples:
        if (
            any(type(v) is not int for v in (sample.position, sample.begin, sample.end))
            or sample.begin != cursor
            or not sample.begin <= sample.position < sample.end <= length
        ):
            raise ValueError(
                "KL sample strata must exactly partition the effective response"
            )
        cursor = sample.end
    if cursor != length:
        raise ValueError("KL sample strata do not cover the entire effective response")
    result = {"response_length": length, "evaluated_positions": len(samples)}
    for name, entries in values.items():
        if len(entries) != len(samples) or not all(
            math.isfinite(float(v)) for v in entries
        ):
            raise ValueError(
                "KL statistics must be finite and align with the sampled positions"
            )
        total = math.fsum(
            sample.weight * float(value)
            for sample, value in zip(samples, entries, strict=True)
        )
        result[name + "/sum_est"] = total
        result[name + "/token_mean_est"] = total / length
    return result
