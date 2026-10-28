"""Layerwise CPU weight loading on a separate DMA engine.

Models the pinned SGLang read-only weight lifecycle: load, wait, compute,
release; prefetch one following layer. CPU master weights persist. HBM weights
stay alive until the last operator of their layer, including early-use weights.
Only one critical SP rank is represented. No cross-step prefetch, shared host
root-complex contention, or allocator/kernel workspace is inferred.
"""
from collections import defaultdict
from fractions import Fraction

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor


OFFLOADED_COMPONENTS = frozenset({
    'transformer', 'transformer_ref', 'transformer_text_refiner',
    'qwen_text', 'qwen_vision', 'video_vae_decoder',
})


class WeightLoad(Kernel):
    """CPU master weights copied to transient HBM buffers by DMA."""

    def __init__(self, component, layer, layer_weights):
        weights = {
            name: Tensor(t.dtype, t.shape, weight_id=f'host:{t.weight_id}',
                         storage_bytes=t.size_bytes)
            for name, t in layer_weights.items()
        }
        volume = sum(t.size_bytes for t in layer_weights.values())
        # One backing buffer owns the layer's HBM allocation and DMA writes.
        # Compute kernels retain zero-storage weight views for their HBM reads.
        # Load -> first compute and last compute -> release enforce validity.
        super().__init__(weights=weights,
                         outputs={'buffer': Tensor('bf16', (Fraction(volume) / 2,))})
        self.csv_metadata = {
            'node_name': f'{component}.layer{layer}.weight_load',
            'operator': 'WeightLoad', 'component': component,
            'layer_idx': layer, 'compute_scope': 'critical_rank',
            'source_weight_saved': 0, 'activation_peak_bytes_proxy': 0,
            'host_to_device_bytes': volume,
            'offload_direction': 'host_to_device',
            'weight_residency': 'host_master',
        }

    @property
    def flops(self):
        return 0.0

    @property
    def flops_by_dtype(self):
        return {}


class LayerRelease(Kernel):
    """Lifetime marker; does not read, copy back, or clear read-only weights."""

    def __init__(self, component, layer, buffer):
        super().__init__(inputs={'buffer': buffer})
        self.csv_metadata = {
            'node_name': f'{component}.layer{layer}.weight_release',
            'operator': 'LayerRelease', 'component': component,
            'layer_idx': layer, 'compute_scope': 'critical_rank',
            'source_weight_saved': 0, 'activation_peak_bytes_proxy': 0,
        }

    def input_read_fraction(self, port):
        return 0


def apply_layerwise_offload(graph, placement, hardware, component, *,
                           prefetch_size=1, layer_ids=None):
    """Rewrite layer weights as load outputs with bounded lifetime."""
    if component not in OFFLOADED_COMPONENTS and layer_ids is None:
        return
    if prefetch_size not in (0, 1):
        raise ValueError('H3 layerwise offload supports prefetch_size 0 or 1')
    if not getattr(hardware, 'dma_devices', ()) or not getattr(hardware, 'host_memory', None):
        raise ValueError('H3 layerwise offload requires host memory and a dedicated DMA engine')
    ordered = graph.topological_sort()
    if layer_ids is None:
        layer_ids = {
            k.csv_metadata['layer_idx'] for k in ordered
            if k.csv_metadata['operator'] in (
                'FlashAttentionOperator', 'NpuFusedInferAttentionScoreOperator')
        }
    layers = defaultdict(list)
    for kernel in ordered:
        layer = kernel.csv_metadata['layer_idx']
        if layer in layer_ids:
            layers[layer].append(kernel)
    loads, ends = [], []
    for layer, kernels in layers.items():
        layer_weights = {}
        for kernel in kernels:
            for name, tensor in list(kernel.weights.items()):
                port = f'weight_{len(layer_weights)}'
                layer_weights[port] = tensor
                view = Tensor(tensor.dtype, tensor.shape, weight_id=tensor.weight_id,
                              storage_bytes=0)
                kernel.weights[name] = view
                placement.set_tensor_memory(view, hardware.find_local_memory(hardware.devices[0]))
                kernel.csv_metadata['offloaded_weight_bytes'] = (
                    kernel.csv_metadata.get('offloaded_weight_bytes', 0) + tensor.size_bytes)
                kernel.csv_metadata['weight_residency'] = 'layer_buffer_view'
        if not layer_weights:
            continue
        load = WeightLoad(component, layer, layer_weights)
        graph.add_kernel(load)
        for tensor in load.weights.values():
            placement.set_tensor_memory(tensor, hardware.host_memory)
        placement.set_kernel_device(load, hardware.dma_devices[0])
        graph.add_control_edge(load, kernels[0])
        release = LayerRelease(component, layer, load.outputs['buffer'])
        successors = list(graph._dag.successors(kernels[-1]))
        graph.add_kernel(release)
        placement.set_kernel_device(release, hardware.devices[0])
        graph.add_data_edge(load, release, {'buffer': 'buffer'})
        graph.add_control_edge(kernels[-1], release)
        for successor in successors:
            graph.add_control_edge(release, successor)
        # Current + at most one following layer occupy HBM. Predecessor
        # completion releases its buffers before the next allocation starts.
        if loads:
            graph.add_control_edge(loads[-1], load)
        wait_index = len(loads) - 1 - prefetch_size
        if wait_index >= 0:
            graph.add_control_edge(ends[wait_index], load)
        loads.append(load)
        ends.append(release)


def graph_stats(result, hardware):
    """Per-invocation/per-critical-rank transfer and weight residency metrics."""
    unique = {entry.kernel: entry for entry in result.trace}
    copies = [entry for kernel, entry in unique.items() if isinstance(kernel, WeightLoad)]
    fixed = {}
    for kernel in unique:
        if not isinstance(kernel, WeightLoad):
            for tensor in kernel.weights.values():
                fixed[tensor.weight_id or id(tensor)] = tensor.storage_bytes
    serial_compute_us = sum(entry.end_us - entry.start_us
                            for kernel, entry in unique.items()
                            if not isinstance(kernel, WeightLoad))
    return {
        'fixed_weight_bytes': sum(fixed.values()),
        'streamed_weight_bytes': sum(entry.kernel.csv_metadata['host_to_device_bytes']
                                     for entry in copies),
        'weight_peak_bytes': max((peak for memory, peak in result.peak_memory.items()
                                  if memory.kind == 'hbm'), default=0),
        'host_to_device_bytes': sum(entry.kernel.csv_metadata['host_to_device_bytes']
                                    for entry in copies),
        'copy_time_ms': sum(entry.end_us - entry.start_us for entry in copies) / 1000,
        'exposed_transfer_time_ms': max(0.0, result.total_time_us - serial_compute_us) / 1000 if copies else 0.0,
        'offloaded_layer_count': len(copies),
    }
