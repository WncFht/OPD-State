"""Strict decoding for project-owned contracts, without a second verl schema."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints


def decode(expected, value, path="configuration"):
    """Decode only declared fields and reject implicit boolean/numeric casts."""
    origin, args = get_origin(expected), get_args(expected)
    if expected is Any:
        raise TypeError(f"{path}: unrestricted scientific fields are not supported")
    if origin in (types.UnionType, Union):
        for choice in args:
            try:
                return decode(choice, value, path)
            except (TypeError, ValueError):
                pass
        raise TypeError(f"{path}: value does not match {expected}")
    if is_dataclass(expected):
        if type(value) is not dict:
            raise TypeError(f"{path}: expected an object")
        unknown = set(value) - {item.name for item in fields(expected)}
        if unknown:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        hints = get_type_hints(expected)
        return expected(
            **{k: decode(hints[k], v, f"{path}.{k}") for k, v in value.items()}
        )
    if origin is list:
        if type(value) is not list:
            raise TypeError(f"{path}: expected a list")
        return [decode(args[0], item, f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is dict:
        if type(value) is not dict:
            raise TypeError(f"{path}: expected a mapping")
        return {
            decode(args[0], k, f"{path}.key"): decode(args[1], v, f"{path}.{k}")
            for k, v in value.items()
        }
    if expected is float:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise TypeError(f"{path}: expected a finite number")
        return float(value)
    if type(value) is not expected:
        raise TypeError(
            f"{path}: expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def positive(value, path):
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")


@dataclass(frozen=True)
class Objective:
    """Algorithm identity; optimizer and backend loss fields retain native names."""

    schema_version: int = 1
    kind: str = "sampled_opd"

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported objective schema version")
        if self.kind not in {"sampled_opd", "unbiased_topk_tail"}:
            raise ValueError("Unknown OPD objective")
