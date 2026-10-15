"""Small, executable model-onboarding example; no model weights are loaded.

Run after installing this repository:
    python examples/toy_model.py --steps 3 --output-dir outputs/toy
    python examples/toy_model.py --check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.experiments.run_and_export import run_and_export
from rooflang.programs.experiments.simulation import simulate
from rooflang.programs.experiments.workflow import Workflow
from rooflang.runtime.stages import GraphResult, StageResult


class Affine(Kernel):
    """y[b,t,d] = x[b,t,d] * scale[d] + bias[d], stored/computed in BF16."""

    def __init__(self, batch_size: int, tokens: int, hidden: int):
        for name, value in (("batch_size", batch_size), ("tokens", tokens),
                            ("hidden", hidden)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shape = (batch_size, tokens, hidden)
        super().__init__(
            inputs={"x": Tensor("bf16", shape)},
            weights={"scale": Tensor("bf16", (hidden,), weight_id="affine.scale"),
                     "bias": Tensor("bf16", (hidden,), weight_id="affine.bias")},
            outputs={"y": Tensor("bf16", shape)},
        )

    @property
    def flops(self):
        return 2 * self.inputs["x"].n_elements


def make_hardware():
    """Deliberately simple reference device: 1 BF16 TFLOP/s, 100 GB/s, 1 GB."""
    hardware = HardwareGraph()
    gpu = Compute("toy-gpu", tflops={"bf16": 1.0}, kind="gpu")
    memory = Memory("toy-hbm", capacity_gb=1.0, kind="hbm")
    hardware.add_node(gpu)
    hardware.add_node(memory)
    hardware.add_edge(FabricEdge(
        "toy-memory-bus", gpu, memory, 100.0, 100.0,
        is_full_duplex=False, alpha_us=0.0,
    ))
    return hardware


def create_workflow(hardware, config):
    defaults = {"batch_size": 1, "tokens": 32, "hidden": 16, "steps": 3}
    if unknown := set(config) - defaults.keys():
        raise ValueError(f"Unknown toy fields: {sorted(unknown)}")
    resolved = {**defaults, **config}
    for name, value in resolved.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    devices = sorted((node for node in hardware.nodes
                      if isinstance(node, Compute) and node.kind == "gpu"),
                     key=lambda node: node.name)
    if not devices:
        raise ValueError("Toy workflow needs one GPU component")

    def run_stage():
        kernel = Affine(resolved["batch_size"], resolved["tokens"], resolved["hidden"])
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware, graph)
        placement.set_kernel_device(kernel, devices[0])
        graph.validate()
        placement.validate(graph)
        result = simulate(graph, placement, hardware)
        return StageResult([GraphResult("affine_step", result, resolved["steps"])])

    return Workflow({"transform": run_stage}, {
        "workflow": "toy_affine", "batch_size": resolved["batch_size"],
        "workload": resolved, "request_count": 1,
        "_resolved_config": dict(resolved),
        "timing_status": "modeled_toy_affine_only",
    })


class ExampleChecks(unittest.TestCase):
    def test_small_hand_calculated_case(self):
        # 8 elements: 16 FLOPs, 16 input + 16 output + 16 weight bytes.
        kernel = Affine(1, 2, 4)
        self.assertEqual(kernel.flops, 16)
        self.assertEqual(kernel.transferred_bytes, 48)
        self.assertEqual(kernel.weight_bytes, 16)
        result = create_workflow(make_hardware(), {
            "batch_size": 1, "tokens": 2, "hidden": 4, "steps": 1,
        }).run()["transform"].graphs[0].result
        # Roofline max(16/1e12, 48/100e9) seconds, without launch overhead.
        self.assertAlmostEqual(result.total_time_us, 0.00048, places=10)

    def test_invalid_configuration(self):
        for config in ({"steps": 0}, {"tokens": True}, {"hidden": -1}, {"tp": 2}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_workflow(make_hardware(), config)

    def test_repeats_scale_work_without_multiplying_resident_weights(self):
        rows = []
        with tempfile.TemporaryDirectory(prefix="rooflang-toy-") as directory:
            for steps in (1, 3):
                config = {"batch_size": 1, "tokens": 2, "hidden": 4, "steps": steps}
                hardware = make_hardware()
                _, output = run_and_export(
                    create_workflow(hardware, config), model_name="toy_affine",
                    hardware_name="ToyGPU", output_dir=Path(directory) / str(steps),
                    config=config, hardware=hardware,
                )
                with (output / "stage_metrics.csv").open(newline="") as file:
                    rows.append(next(csv.DictReader(file)))
            self.assertEqual(float(rows[1]["total_flops"]), 48)
            self.assertEqual(float(rows[1]["total_mem_bytes"]), 144)
            self.assertEqual(float(rows[1]["total_weight_bytes"]), 16)
            self.assertEqual(rows[0]["peak_memory_bytes"], rows[1]["peak_memory_bytes"])
            self.assertAlmostEqual(float(rows[1]["execution_time_ms"]),
                                   3 * float(rows[0]["execution_time_ms"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON object overriding toy dimensions")
    for name in ("batch-size", "tokens", "hidden", "steps"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--check", action="store_true", help="Run the three executable checks")
    args = parser.parse_args()
    if args.check:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExampleChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        parser.error("--config must contain a JSON object")
    for name in ("batch_size", "tokens", "hidden", "steps"):
        if (value := getattr(args, name)) is not None:
            config[name] = value
    hardware = make_hardware()
    _, directory = run_and_export(
        create_workflow(hardware, config), model_name="toy_affine", hardware_name="ToyGPU",
        output_dir=args.output_dir, config=config, hardware=hardware,
    )
    print(directory.resolve())


if __name__ == "__main__":
    main()
