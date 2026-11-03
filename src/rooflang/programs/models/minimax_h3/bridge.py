"""Lower source cost recipes to serial native graphs, preserving source scope."""
from dataclasses import asdict, is_dataclass
from itertools import accumulate
import math
from types import SimpleNamespace
from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.language.kernels.comm import CommKernel
from rooflang.programs.models.magi2_preview.bridge import AnalyticalKernel, _tensor
from rooflang.programs.models.magi2_preview.operators import (
    FlashAttentionOperator, NpuFusedInferAttentionScoreOperator,
)
from rooflang.programs.presets.ascend910b import Ascend910B3
from .compat import BlockFactory, Parallel, OpType, QuantConfig
from .offload import apply_layerwise_offload
from .decode_ops import DecodeOperator
from .audio_transpose import AudioTransposeOperator
from .qwen_fia import QwenVisionFIA, QwenTextFIA
from . import blocks, mlp, embedding  # register the finite H3 recipe set

class H3Kernel(AnalyticalKernel):
    def __init__(self, operator, metadata, weight_id):
        super().__init__(operator, metadata, weight_id)
        self._compute_tflops_override = None
        details = operator.get_memory_details()
        self.csv_metadata['activation_peak_bytes_proxy'] = details.get(
            'activation_peak', details.get('activation', 0))
        if isinstance(operator, DecodeOperator) and not operator.weight_saved_bytes:
            # Generated weight layouts are read without owning permanent storage.
            for name, tensor in list(self.weights.items()):
                self.weights[name] = Tensor(tensor.dtype, tensor.shape,
                                            weight_id=tensor.weight_id, storage_bytes=0)

    def scheduled_compute_time_us(self, device):
        if self._compute_tflops_override is not None:
            return sum(flops / (self._compute_tflops_override[dtype] * 1e6)
                       for dtype, flops in self.flops_by_dtype.items())
        if isinstance(self.operator, AudioTransposeOperator) and self.operator.tile_cost is not None:
            return self.operator.tile_cost.compute_time_us(device.tflops)
        return None

    @property
    def flops_by_dtype(self):
        c = self.operator.get_compute_component()
        if c is None: return {'fp16': 0.0}
        if c.unit_type == 'mixed':
            return {c.get_unit_dtype('cube') + ':cube': c.cube_flops,
                    c.get_unit_dtype('vector') + ':vector': c.vector_flops}
        return {c.dtype + ':' + c.unit_type: c.flops}


def _set_memory_efficiency(kernel, hardware):
    """Apply B3 unit efficiencies to time, preserving physical traffic bytes."""
    if not isinstance(hardware, Ascend910B3):
        return
    op = kernel.operator
    component = op.get_compute_component()
    if isinstance(op, NpuFusedInferAttentionScoreOperator):
        by_unit = op.get_memory_bytes_by_unit()
        basis = 'single_pass_qkv_cube_output_mask_lse_vector'
    else:
        # Generic mixed recipes expose no per-unit traffic. Keep their fused
        # boundary on cube; never infer a byte split from the FLOP ratio.
        unit = 'vector' if component is None or component.unit_type in ('vector', 'storage') else 'cube'
        by_unit = {unit: op.get_memory_bytes()}
        basis = 'operator_unit_or_cube_for_generic_mixed_boundary'
    assumptions = hardware.metadata['simulation_assumptions']
    efficiencies = {unit: assumptions[f'{unit}_memory_efficiency'] for unit in by_unit}
    total = sum(by_unit.values())
    weighted = sum(size / efficiencies[unit] for unit, size in by_unit.items())
    kernel.local_memory_efficiency = total / weighted if weighted else 1.0
    kernel.csv_metadata.update(memory_bytes_by_unit=by_unit,
                               memory_efficiency_by_unit=efficiencies,
                               local_memory_efficiency=kernel.local_memory_efficiency,
                               memory_efficiency_basis=basis)

class H3Collective(CommKernel):
    """Source all-to-all bytes, timed on the hardware's aggregate fabric."""
    def __init__(self, payload, world):
        self.total_bytes = payload
        self.world = world
        super().__init__()
    @property
    def transferred_bytes(self): return self.total_bytes * (self.world-1) / self.world
    @property
    def input_bytes(self): return self.total_bytes
    @property
    def output_bytes(self): return self.total_bytes
    @property
    def flops(self): return 0.0


def component_nodes(component, architecture, mm, seq_len, sp=1):
    if component == 'video_vae_decoder':
        from .video_decode import video_decode_nodes, video_decode_stitch_nodes
        return video_decode_stitch_nodes(mm) if mm.get('decode_stitch') else video_decode_nodes(seq_len, mm)
    if component == 'audio_vae_decoder':
        from .audio_decode import audio_decode_nodes
        return audio_decode_nodes(mm)
    cfg = next(c for c in architecture['components'] if c['name'] == component)
    local = math.ceil(seq_len / sp)
    context = SimpleNamespace(batch_size=1, seq_len=local, kv_len=local,
        cache_len=0, offload_config=None, deploy_strategies=None,
        multimodal_config=mm, parallel_strategies={Parallel.SP:sp, Parallel.CP:1, Parallel.TP:1},
        network_list={Parallel.SP:[sp], Parallel.CP:[1], Parallel.TP:[1], 'AF_COMM':[1]})
    nodes=[]; layer=0
    if cfg.get('vocab_size'):
        nodes.extend(BlockFactory.registry['emb-token']('emb-token', {'type':'emb-token'}, cfg).build(context, layer))
        layer += 1
    for group in cfg['layer_blocks']:
        for _ in range(group.get('repeat_times',1)):
            for layer_config in group['layers']:
                for _ in range(layer_config.get('count',1)):
                    for bc in layer_config['blocks']:
                        block=BlockFactory.registry[bc['type']](bc['type'],bc,cfg)
                        nodes.extend(block.build(context,layer))
                    layer += 1
    if mm.get('execution_profile') == 'sglang_v0_5_21':
        dtype = ('fp32' if component in ('video_vae_encoder', 'audio_vae_encoder', 'audio_vae_decoder')
                 else 'fp16' if component == 'video_vae_decoder' else 'bf16')
        quant = QuantConfig(dtype, dtype, dtype, dtype)
        for node in nodes:
            node.operator.quant_config = quant
            for field in ('compute_dtype', 'weight_storage_dtype', 'kv_storage_dtype', 'communicate_dtype'):
                setattr(node.operator, field, dtype)
    return nodes


def validate_transformer_fia_compute_efficiency(value):
    if value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or not 0 < value <= 1):
        raise ValueError('transformer_fia_compute_efficiency must be finite and in (0, 1]')


def build_component(component, architecture, mm, seq_len, hardware, sp=1):
    fia_efficiency = mm.get('transformer_fia_compute_efficiency')
    validate_transformer_fia_compute_efficiency(fia_efficiency)
    devices=sorted((n for n in hardware.nodes if isinstance(n,Compute) and n.kind=='gpu'),key=lambda n:n.name)
    ascend_fia = (isinstance(hardware, Ascend910B3)
                  and mm.get('attention_backend', 'auto') == 'auto')
    used_rows = seq_len
    if isinstance(hardware, Ascend910B3) and component == 'transformer_text_refiner':
        # Pinned SGLang refines the full prompt on each rank without Ulysses.
        sp = 1
    elif ascend_fia and component in ('transformer', 'transformer_ref'):
        # packed_sequence.py uses cu_seqlens=[0, used, ceil(used/64)*64].
        seq_len = (seq_len + 63) // 64 * 64
    if ascend_fia and component == 'qwen_vision':
        segments = mm.get('qwen_attention_segments')
        if (not isinstance(segments, (list, tuple)) or not segments
                or any(type(length) is not int or length <= 0 for length in segments)
                or sum(segments) != seq_len):
            raise ValueError('qwen_attention_segments must contain positive integers summing to seq_len')
        # Ulysses splits heads; its local projection padding is not a TND document.
        mm = {**mm, '_qwen_vision_packed_rows': seq_len}
    nodes=component_nodes(component,architecture,mm,seq_len,sp)
    graph=ComputeGraph(); placement=Placement(hardware,graph)
    previous=previous_compute=None
    for index,node in enumerate(nodes):
        op=node.operator
        packed = False
        if ascend_fia and isinstance(op, FlashAttentionOperator):
            p = op.params
            packed = component in ('transformer', 'transformer_ref', 'transformer_text_refiner')
            if component == 'qwen_vision':
                operator_class = QwenVisionFIA
                boundaries = list(accumulate(mm['qwen_attention_segments']))
                layout = 'TND'
                options = mm.get('qwen_fia', {}).get('vision', {})
            elif component == 'qwen_text':
                operator_class = QwenTextFIA
                boundaries = None
                layout = 'BNSD'
                options = mm.get('qwen_fia', {}).get('text', {})
            else:
                operator_class = NpuFusedInferAttentionScoreOperator
                boundaries = ([used_rows] + ([p.q_len] if p.q_len > used_rows else [])) if packed else None
                layout = 'TND' if packed else 'BNSD'
                options = {}
            op = operator_class(batch_size=p.batch_size, num_heads=p.num_heads,
                num_key_value_heads=p.kv_heads, head_dim=p.head_dim,
                q_len=p.q_len, kv_len=p.kv_len,
                input_layout=layout, actual_seq_lengths=boundaries,
                actual_seq_lengths_kv=boundaries.copy() if boundaries is not None else None,
                scale=p.head_dim ** -0.5, causal=p.causal,
                quant_config=op.quant_config, **options)
        if isinstance(op, AudioTransposeOperator) and isinstance(hardware, Ascend910B3):
            op.enable_tiling(hardware.metadata['official_specification']['ai_core_count'])
        if op.params is not None: op.params.validate()
        details=op.get_memory_details()
        meta={'node_name':node.node_name,'operator':type(op).__name__,'module':node.module.value,
              'layer_idx':node.layer_idx,'block_id':node.block_id,'tag_mask':node.tag_mask,
              'shape':asdict(op.params) if is_dataclass(op.params) else {},
              'compute_dtype':op.compute_dtype,'weight_storage_dtype':op.weight_storage_dtype,
              'communicate_dtype':op.communicate_dtype,
              'kv_storage_dtype':op.kv_storage_dtype,'component':component,'recipe_node_index':index,
              'compute_scope':'critical_rank','source_weight_saved':details.get('weight_saved',0),
              'activation_peak_bytes_proxy':details.get('activation_peak',details.get('activation',0))}
        if isinstance(op, DecodeOperator):
            meta.update(operator=op.params.op_type, decode_model='trace_aligned',
                        source_operator=op.params.op_type,
                        vector_compute_dtype=op.vector_compute_dtype,
                        cube_compute_dtype=op.compute_dtype,
                        assumption='measured_topology_analytical_costs_no_latency_fit')
        if isinstance(op, AudioTransposeOperator):
            if op.tile_cost is not None:
                meta.update(decode_model='generic_phase_tiled_transpose',
                            assumption='trace_topology_generic_algorithm_not_cann_kernel_reconstruction',
                            transpose_tiling_status='active_910b3',
                            transpose_schedule=op.tile_cost.to_dict(),
                            boundary_activation_bytes=op.activation_bytes,
                            boundary_bias_bytes=op.weight_saved_bytes)
            else:
                meta.update(transpose_tiling_status='not_applied_hardware_scope',
                            transpose_tiling_scope='Ascend910B3_only')
        if isinstance(op, NpuFusedInferAttentionScoreOperator):
            meta.update(source_operator='torch.ops.npu.npu_fused_infer_attention_score',
                        source_documentation=op.source_url,
                        cube_compute_dtype=op.compute_dtype,
                        vector_compute_dtype=op.vector_compute_dtype,
                        atten_mask_shape=op.atten_mask_shape, assumption=op.assumption,
                        attention_pairs=op.valid_pairs,
                        attention_alignment=('pinned_sglang_dit' if packed
                                             else 'analytical_fia_substitution'))
            if isinstance(op, (QwenVisionFIA, QwenTextFIA)):
                meta.update(op.cost_metadata())
                meta['attention_alignment'] = ('qwen_vision_packed_tnd' if isinstance(op, QwenVisionFIA)
                                               else 'qwen_text_causal_bnsd')
        if node.op_type is OpType.COMMUNICATION:
            world=math.prod(op.net_list); payload=op.op_size*op.quant_config.get_communicate_size()
            kernel=H3Collective(payload,world);kernel.csv_metadata=meta
            for rank,device in enumerate(devices[:world]):
                mem=hardware.find_local_memory(device)
                for ports in (kernel.inputs,kernel.outputs):
                    tensor=_tensor(0,'fp16'); ports[str(rank)]=tensor
                    placement.set_tensor_memory(tensor,mem)
        else:
            kernel=H3Kernel(op,meta,f'{component}:{index}')
            _set_memory_efficiency(kernel, hardware)
            if (fia_efficiency is not None and isinstance(hardware, Ascend910B3)
                    and type(op) is NpuFusedInferAttentionScoreOperator
                    and component in ('transformer', 'transformer_ref')):
                # Start from physical peaks so the global compute efficiencies are replaced once.
                derived = hardware.metadata['derived_specification']
                assumptions = hardware.metadata['simulation_assumptions']
                kernel._compute_tflops_override = {}
                for dtype_unit in kernel.flops_by_dtype:
                    dtype, unit = dtype_unit.split(':')
                    kernel._compute_tflops_override[dtype_unit] = derived[f'{unit}_tflops'][dtype] * fia_efficiency
                kernel.csv_metadata.update(
                    transformer_fia_compute_efficiency=fia_efficiency,
                    compute_efficiency_by_unit={unit: fia_efficiency for unit in ('cube', 'vector')},
                    hardware_compute_efficiency_by_unit={unit: assumptions[f'{unit}_efficiency']
                                                         for unit in ('cube', 'vector')},
                    effective_compute_tflops_by_dtype=dict(kernel._compute_tflops_override),
                    compute_efficiency_basis='transformer_fia_absolute_override')
            # Reference memory peak excludes kv_cache, reported separately.
            kernel.csv_metadata['activation_peak_bytes_proxy']=details.get('activation_peak',details.get('activation',0))
            if previous_compute is not None:
                kernel.inputs['previous_traffic']=previous_compute.outputs['traffic']
        graph.add_kernel(kernel)
        if previous is not None: graph.add_control_edge(previous,kernel)
        if node.op_type is not OpType.COMMUNICATION:
            if previous_compute is not None:
                graph.add_data_edge(previous_compute,kernel,{'traffic':'previous_traffic'})
            placement.set_kernel_device(kernel,devices[0])
            previous_compute=kernel
        previous=kernel
    if mm.get('layerwise_offload'):
        apply_layerwise_offload(graph, placement, hardware, component,
                                prefetch_size=mm['layerwise_offload']['prefetch_size'])
    graph.validate();placement.validate(graph)
    return graph,placement
