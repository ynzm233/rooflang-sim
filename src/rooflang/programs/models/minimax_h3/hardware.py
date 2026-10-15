"""Source A5_pr_server configuration, with its original units and efficiencies."""
from rooflang.language.graph import HardwareGraph, FabricEdge
from rooflang.language.hardware.component import Compute, Memory

class A5PRServer(HardwareGraph):
    def __init__(self, n_devices=4):
        if type(n_devices) is not int or not 1 <= n_devices <= 8:
            raise ValueError('A5_pr_server supports 1 to 8 devices within one source node')
        super().__init__()
        self.n_devices = n_devices
        gpus = []
        for i in range(n_devices):
            gpu = Compute(name=f'a5-{i}', kind='gpu', tflops={
                'fp16': 486 * .7, 'fp8': 972 * .7,
                'fp16:cube': 486 * .7, 'fp16:vector': 54 * .6,
                'fp8:cube': 972 * .7, 'fp8:vector': 108 * .6})
            memory = Memory(name=f'hbm-{i}', kind='hbm', capacity_gb=128 * 1024**3 / 1e9)
            self.add_node(gpu); self.add_node(memory); gpus.append(gpu)
            # Source roofline treats configured GB/s as GiB/s; retain explicitly.
            bw = 1600 * .8 * 1024**3 / 1e9
            self.add_edge(FabricEdge(name='hbm', src=gpu, dst=memory,
                src_to_dst_bandwidth_gbs=bw, dst_to_src_bandwidth_gbs=bw,
                is_full_duplex=False, alpha_us=0))
        for i, src in enumerate(gpus):
            for dst in gpus[i+1:]:
                self.add_edge(FabricEdge(name='intra-node', src=src, dst=dst,
                    src_to_dst_bandwidth_gbs=392*.7, dst_to_src_bandwidth_gbs=392*.7,
                    is_full_duplex=True, alpha_us=1))
    def find_aggregate_bandwidth(self, devices):
        efficiencies = {2: .142857, 4: .428571, 8: 1.0}
        n = len(devices)
        key = min(efficiencies, key=lambda k: abs(k-n))
        return 392 * .7 * efficiencies[key]
