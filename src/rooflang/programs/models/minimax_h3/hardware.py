"""Ascend 950PR specifications with explicit, uncalibrated simulation assumptions."""
from copy import deepcopy

from rooflang.language.graph import HardwareGraph, FabricEdge
from rooflang.language.hardware.component import Compute, Memory


# Huawei Ascend 950 NPU Architecture White Paper, table 3-1, PDF pages 13-15.
# Select the 32-Cube / 64-Vector PR variant; 486 TFLOPS is its FP16
# Cube+Vector total, not its Cube throughput. Units below are decimal.
OFFICIAL_SPEC = {
    'product': 'Ascend 950PR',
    'variant': '32 Cube / 64 Vector, 128 GB',
    'product_url': 'https://www.hiascend.com/hardware/processor',
    'source_url': 'https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/'
                  '%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf',
    'source_table': '3-1 (PDF pages 13-15)',
    'verified_on': '2026-10-15',
    'cube_tflops': {'fp16': 432.0, 'fp8': 865.0},
    'vector_tflops': {'fp16': 54.0},
    'memory_capacity_gb': 128.0,
    'memory_bandwidth_gbs': 1600.0,
}

# Retained from the source model because they affect time or deployment limits.
# These values are NOT Huawei specifications or measured calibration results.
# In particular, 2016 GB/s is the chip's aggregate bidirectional UB bandwidth;
# it does not establish a per-peer link or collective bandwidth for this server.
SIMULATION_ASSUMPTIONS = {
    'cube_efficiency': 0.7,
    'vector_efficiency': 0.6,
    'memory_efficiency': 0.8,
    'max_devices_per_node': 8,
    'intra_node_bandwidth_gbs': 392.0,
    'intra_node_efficiency': 0.7,
    'intra_node_latency_us': 1.0,
    'collective_efficiencies': {2: 0.142857, 4: 0.428571, 8: 1.0},
}


class A5PRServer(HardwareGraph):
    def __init__(self, n_devices=4):
        if type(n_devices) is not int or not 1 <= n_devices <= SIMULATION_ASSUMPTIONS['max_devices_per_node']:
            raise ValueError('A5_pr_server models 1 to 8 devices within one assumed server node')
        super().__init__()
        self.n_devices = n_devices
        self.metadata = {
            'official_specification': deepcopy(OFFICIAL_SPEC),
            'simulation_assumptions': deepcopy(SIMULATION_ASSUMPTIONS),
        }
        spec = self.metadata['official_specification']
        assumptions = self.metadata['simulation_assumptions']
        tflops = {}
        for dtype, peak in spec['cube_tflops'].items():
            tflops[dtype] = tflops[f'{dtype}:cube'] = peak * assumptions['cube_efficiency']
        for dtype, peak in spec['vector_tflops'].items():
            tflops[f'{dtype}:vector'] = peak * assumptions['vector_efficiency']
        gpus = []
        for i in range(n_devices):
            gpu = Compute(name=f'a5-{i}', kind='gpu', tflops=tflops.copy())
            memory = Memory(name=f'hbm-{i}', kind='hbm', capacity_gb=spec['memory_capacity_gb'])
            self.add_node(gpu); self.add_node(memory); gpus.append(gpu)
            bw = spec['memory_bandwidth_gbs'] * assumptions['memory_efficiency']
            self.add_edge(FabricEdge(name='hbm', src=gpu, dst=memory,
                src_to_dst_bandwidth_gbs=bw, dst_to_src_bandwidth_gbs=bw,
                is_full_duplex=False))
        network_bw = assumptions['intra_node_bandwidth_gbs'] * assumptions['intra_node_efficiency']
        for i, src in enumerate(gpus):
            for dst in gpus[i+1:]:
                self.add_edge(FabricEdge(name='intra-node', src=src, dst=dst,
                    src_to_dst_bandwidth_gbs=network_bw, dst_to_src_bandwidth_gbs=network_bw,
                    is_full_duplex=True, alpha_us=assumptions['intra_node_latency_us']))

    def find_aggregate_bandwidth(self, devices):
        assumptions = self.metadata['simulation_assumptions']
        efficiencies = assumptions['collective_efficiencies']
        n = len(devices)
        key = min(efficiencies, key=lambda k: abs(k-n))
        return (assumptions['intra_node_bandwidth_gbs'] * assumptions['intra_node_efficiency']
                * efficiencies[key])
