"""Ascend 910B3 peaks with explicit host-transfer and HCCS assumptions."""
from copy import deepcopy

from rooflang.language.graph import FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory


# Huawei's LocMoE+ paper names the B3 SKU and gives core count, clock,
# combined FP16 peak, memory capacity and memory bandwidth in Appendix C.2.
OFFICIAL_SPEC = {
    "product": "Ascend 910B3",
    "variant": "910B3, 20 AI cores at 1.8 GHz",
    "source_url": "https://arxiv.org/html/2406.00023v1#A3.SS2",
    "source_section": "LocMoE+, Appendix C.2, Huawei authors",
    "verified_on": "2026-10-20",
    "ai_core_count": 20,
    "ai_core_clock_ghz": 1.8,
    "reported_fp16_total_tflops": 313.0,
    "memory_bandwidth_gbs": 1600.0,
    "memory_capacity_reported_gb": 64.0,
}

# These are architecture-derived unit peaks, not independently published B3
# measurements. Cube and Vector must not both receive the combined 313T rate.
DERIVED_SPEC = {
    "cube_tflops": {"fp16": 294.912, "bf16": 294.912, "fp32": 73.728},
    "vector_tflops": {"fp16": 18.432, "bf16": 18.432, "fp32": 9.216},
    "cube_formula": "20 cores * 1.8 GHz * (16 * 16 * 16 * 2) FLOP/cycle",
    "vector_formula": "20 cores * 1.8 GHz * 512 FLOP/cycle",
    "fp16_total_tflops": 313.344,
    "basis": "architecture-derived, consistent with the published rounded 313T total",
    "bf16_basis": "assume the same unit throughput as FP16; not calibrated",
    "unit_split_reference_url": "https://raw.githubusercontent.com/Ascend/DrivingSDK/master/"
                                "mx_driving/tools/performance_predict/devices_info.csv",
    "unit_split_reference_scope": "generic 910B Cube:Vector ratio; not a B3 peak table",
    "fp32_cube_reference_url": "https://arxiv.org/html/2507.23387v5#S6.SS1",
    "fp32_cube_reference_scope": "SGEMM-cube authors report B3 native FP32 GEMM peak 73.73 TFLOP/s",
}

# Physical capacity is interpreted from 65536 MB npu-smi device reports.
# Framework-usable capacity can be lower; no driver/runtime reserve is modeled.
# The paper reports a 56 GB/s link without resolving its direction convention.
# Treating it as a bidirectional total is an explicit assumption, not a Huawei
# per-peer specification. The SDK's 392 GB/s aggregate is not this direct link.
# Host links model ideal independent per-device bandwidth, not a measured PCIe
# configuration. The hardware preset does not imply that host compute is timed.
SIMULATION_ASSUMPTIONS = {
    "cube_efficiency": 0.7,
    "vector_efficiency": 0.7,
    "cube_memory_efficiency": 0.6,
    "vector_memory_efficiency": 0.6,
    # Keep physical HBM and DMA bandwidth separate from operator utilization.
    # Model bridges apply the unit-specific values to each kernel's local time.
    "memory_efficiency": 1.0,
    "fp32_cube_basis": "FP16 Cube / 4 = 73.728 TFLOP/s, consistent with reported rounded 73.73",
    "fp32_vector_basis": "assume FP16 Vector / 2 = 9.216 TFLOP/s; generic DrivingSDK ratio, not B3 calibrated",
    "supported_device_counts": [1, 2],
    "physical_memory_capacity_gib": 64.0,
    "memory_capacity_gb": 64 * 1024**3 / 1e9,
    "driver_runtime_reserved_bytes": 0,
    "memory_capacity_basis": "physical HBM; usable framework capacity not modeled",
    "memory_capacity_source_urls": [
        "https://github.com/vllm-project/vllm-ascend/issues/5056",
        "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/"
        "910beta1/maintenref/troubleshooting/troubleshooting_0503.html",
    ],
    "hbm_latency_us": 0.0,
    "intra_node_topology": "one direct full-duplex HCCS link between two devices",
    "reported_link_bandwidth_gbs": 56.0,
    "intra_node_bandwidth_gbs": 28.0,
    "intra_node_bandwidth_basis": "assume 56 GB/s bidirectional total, 28 GB/s per direction",
    "intra_node_source_url": "https://arxiv.org/html/2605.22428v1#S6.SS1",
    "intra_node_source_scope": "paper names 910B4; applying its link rate to B3 is an assumption",
    "intra_node_efficiency": 1.0,
    "intra_node_latency_us": 0.0,
    "host_memory_capacity_gib": 1536.0,
    "host_memory_capacity_gb": 1536 * 1024**3 / 1e9,
    "host_memory_basis": "toolkit reports 1.5 TiB physical RAM; no host reserve modeled",
    "host_memory_source_path": "/Users/z.w.y/Desktop/MiniMax-H3-910B部署工具包/docs/"
                               "MiniMax-H3-910B部署与Profiling校准手册.md",
    "host_memory_source_section": "server resource table: RAM total 1.5 TiB",
    "host_bandwidth_gbs_per_device_per_direction": 32.0,
    "host_bandwidth_basis": "ideal assumed host link, not measured or SKU-verified PCIe bandwidth",
    "host_link_is_full_duplex": True,
    "host_latency_us": 0.0,
    "host_shared_bandwidth_contention_modeled": False,
    "host_compute_modeled": False,
    "dma_basis": "one independent transfer engine per NPU for copy/compute overlap",
    "dma_hbm_bandwidth_basis": "independent 1600 GB/s HBM write/read edge; no shared HBM contention",
    "dma_transfer_scope": "critical-rank transfers; host root-complex contention not modeled",
    "calibration": "uncalibrated user-specified efficiencies: compute 0.7, Cube HBM 0.6, Vector HBM 0.6; no measured latency calibration",
}


class Ascend910B3(HardwareGraph):
    """One or two 64 GiB B3 NPUs and a shared host-memory endpoint."""

    def __init__(self, n_devices=1):
        if type(n_devices) is not int or n_devices not in (1, 2):
            raise ValueError("Ascend910B3 models 1 or 2 devices within one assumed server")
        super().__init__()
        self.n_devices = n_devices
        self.metadata = {
            "official_specification": deepcopy(OFFICIAL_SPEC),
            "derived_specification": deepcopy(DERIVED_SPEC),
            "simulation_assumptions": deepcopy(SIMULATION_ASSUMPTIONS),
        }
        spec = self.metadata["official_specification"]
        derived = self.metadata["derived_specification"]
        assumptions = self.metadata["simulation_assumptions"]
        tflops = {}
        for dtype, peak in derived["cube_tflops"].items():
            tflops[dtype] = tflops[f"{dtype}:cube"] = peak * assumptions["cube_efficiency"]
        for dtype, peak in derived["vector_tflops"].items():
            tflops[f"{dtype}:vector"] = peak * assumptions["vector_efficiency"]
        self.host_memory = Memory(
            name="host-dram", kind="dram",
            capacity_gb=assumptions["host_memory_capacity_gb"],
        )
        self.add_node(self.host_memory)
        devices = []
        dma_devices = []
        for rank in range(n_devices):
            device = Compute(name=f"ascend910b3-{rank}", kind="gpu", tflops=tflops.copy())
            dma = Compute(name=f"ascend910b3-dma-{rank}", kind="dma", tflops={})
            memory = Memory(name=f"hbm-{rank}", kind="hbm",
                            capacity_gb=assumptions["memory_capacity_gb"])
            self.add_node(device)
            self.add_node(dma)
            self.add_node(memory)
            devices.append(device)
            dma_devices.append(dma)
            bandwidth = spec["memory_bandwidth_gbs"] * assumptions["memory_efficiency"]
            self.add_edge(FabricEdge(
                name="hbm", src=device, dst=memory,
                src_to_dst_bandwidth_gbs=bandwidth,
                dst_to_src_bandwidth_gbs=bandwidth,
                is_full_duplex=False, alpha_us=assumptions["hbm_latency_us"],
            ))
            self.add_edge(FabricEdge(
                name=f"dma-hbm-{rank}", src=dma, dst=memory,
                src_to_dst_bandwidth_gbs=bandwidth,
                dst_to_src_bandwidth_gbs=bandwidth,
                is_full_duplex=False, alpha_us=assumptions["hbm_latency_us"],
            ))
            host_bandwidth = assumptions["host_bandwidth_gbs_per_device_per_direction"]
            self.add_edge(FabricEdge(
                name=f"host-pcie-{rank}", src=self.host_memory, dst=dma,
                src_to_dst_bandwidth_gbs=host_bandwidth,
                dst_to_src_bandwidth_gbs=host_bandwidth,
                is_full_duplex=assumptions["host_link_is_full_duplex"],
                alpha_us=assumptions["host_latency_us"],
            ))
        self.devices = tuple(devices)
        self.dma_devices = tuple(dma_devices)
        if n_devices == 2:
            bandwidth = (assumptions["intra_node_bandwidth_gbs"]
                         * assumptions["intra_node_efficiency"])
            self.add_edge(FabricEdge(
                name="hccs", src=devices[0], dst=devices[1],
                src_to_dst_bandwidth_gbs=bandwidth,
                dst_to_src_bandwidth_gbs=bandwidth,
                is_full_duplex=True, alpha_us=assumptions["intra_node_latency_us"],
            ))
        # HardwareGraph's aggregate bandwidth is the minimum pair bandwidth:
        # exactly this direct link for two devices, with no aggregate multiplier.


# Compatibility name: old callers now receive the explicit B3 profile too.
Ascend910B = Ascend910B3
