# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""A model supplies ordered semantic stages and effective run metadata."""

from collections.abc import Callable
from dataclasses import dataclass

from rooflang.runtime.simulator import SimulationResult
from rooflang.runtime.stages import StageResult


@dataclass
class Workflow:
    stages: dict[str, Callable[[], SimulationResult | StageResult]]
    metadata: dict
    summarize: Callable[[dict], dict] | None = None

    def run(self, selected="all"):
        names = list(self.stages) if selected == "all" else [selected]
        if not names or any(name not in self.stages for name in names):
            raise ValueError(f"Unknown stage {selected!r}; available: {', '.join(self.stages)}")
        results = {name: self.stages[name]() for name in names}
        if self.summarize is not None:
            self.metadata.update(self.summarize(results))
        return results
