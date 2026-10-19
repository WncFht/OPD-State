"""Suite schedules and benchmark contracts, independent of backend workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from opd_state.core.config import decode, positive


@dataclass(frozen=True)
class Trigger:
    initial: bool = False
    every_outer_steps: int | None = None
    training_end: bool = True


@dataclass(frozen=True)
class KL:
    enabled: bool = False
    balance_records: bool = True
    forward_batch_size: int = 1
    position_chunk_size: int = 128
    vocab_chunk_size: int = 8192
    fixed_prefix_bank: str | None = None


@dataclass(frozen=True)
class Diagnostics:
    kl: KL = field(default_factory=KL)


@dataclass(frozen=True)
class Suite:
    execution: str = "live"
    trigger: Trigger = field(default_factory=Trigger)
    benchmarks: list[str] = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


@dataclass(frozen=True)
class Data:
    paths: list[str] = field(default_factory=list)
    expected_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Sampling:
    samples_per_case: int = 16
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    do_sample: bool = True


@dataclass(frozen=True)
class Budget:
    prompt_tokens: int = 1024
    response_tokens: int = 31744
    context_tokens: int = 32768


@dataclass(frozen=True)
class Benchmark:
    task: str = "math"
    data: Data = field(default_factory=Data)
    sampling: Sampling = field(default_factory=Sampling)
    budget: Budget = field(default_factory=Budget)


@dataclass(frozen=True)
class Evaluation:
    schema_version: int = 1
    suites: dict[str, Suite] = field(default_factory=dict)
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)


def parse_evaluation(value):
    config = decode(Evaluation, value, "evaluation")
    if config.schema_version != 1:
        raise ValueError("Unsupported evaluation schema version")
    for name, benchmark in config.benchmarks.items():
        if benchmark.task != "math":
            raise ValueError(f"No registered evaluator for {benchmark.task}")
        if not benchmark.data.paths or any(not p for p in benchmark.data.paths):
            raise ValueError(f"Benchmark {name} requires explicit input paths")
        for count in benchmark.data.expected_counts.values():
            positive(count, "expected benchmark population")
        budget, sampling = benchmark.budget, benchmark.sampling
        for key, number in asdict(budget).items():
            positive(number, f"benchmark.{name}.budget.{key}")
        if budget.context_tokens < budget.prompt_tokens + budget.response_tokens:
            raise ValueError("Evaluation context must cover its full response budget")
        positive(sampling.samples_per_case, "samples_per_case")
        if sampling.temperature <= 0 or not 0 < sampling.top_p <= 1:
            raise ValueError("Evaluation sampling needs positive temperature and top-p")
        if sampling.top_k < -1 or sampling.top_k == 0:
            raise ValueError("Use top_k=-1 or a positive evaluation cutoff")
    for name, suite in config.suites.items():
        if suite.execution != "live":
            raise ValueError(
                "Only live evaluation has been integrated with this backend"
            )
        if not suite.benchmarks or len(set(suite.benchmarks)) != len(suite.benchmarks):
            raise ValueError(f"Suite {name} requires distinct benchmarks")
        if set(suite.benchmarks) - set(config.benchmarks):
            raise ValueError(f"Suite {name} names an undeclared benchmark")
        trigger = suite.trigger
        if trigger.every_outer_steps is not None:
            positive(trigger.every_outer_steps, "every_outer_steps")
        if not (trigger.initial or trigger.every_outer_steps or trigger.training_end):
            raise ValueError(f"Suite {name} never executes")
        for key in ("forward_batch_size", "position_chunk_size", "vocab_chunk_size"):
            positive(getattr(suite.diagnostics.kl, key), f"kl.{key}")
    return config


def due_suites(config, *, outer, final=False):
    if type(outer) is not int or outer < 0:
        raise ValueError("Validation versions count completed nonnegative updates")
    result = {}
    for name, suite in config.suites.items():
        reasons = []
        trigger = suite.trigger
        if outer == 0 and trigger.initial:
            reasons.append("initial")
        if (
            outer
            and trigger.every_outer_steps
            and outer % trigger.every_outer_steps == 0
        ):
            reasons.append("periodic")
        if final and trigger.training_end:
            reasons.append("training_end")
        if reasons:
            result[name] = reasons
    return result
