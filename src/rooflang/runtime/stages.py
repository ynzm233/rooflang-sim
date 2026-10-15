# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Model-owned semantic stages, including repeated and untimed work."""

from dataclasses import dataclass, field

from rooflang.runtime.simulator import SimulationResult


@dataclass
class GraphResult:
    label: str
    result: SimulationResult
    multiplier: int = 1

    def __post_init__(self):
        if type(self.multiplier) is not int or self.multiplier < 1:
            raise ValueError("Graph multiplier must be a positive integer")


@dataclass
class StageResult:
    graphs: list[GraphResult] = field(default_factory=list)
    untimed_rows: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
