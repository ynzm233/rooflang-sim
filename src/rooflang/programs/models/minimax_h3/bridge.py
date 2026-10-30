"""Lower source cost recipes to serial native graphs, preserving source scope."""
from dataclasses import asdict, is_dataclass
import math
from types import SimpleNamespace
from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.placement import Placement
from rooflang.language.kernels.comm import CommKernel
from rooflang.programs.models.magi2_preview.bridge import AnalyticalKernel, _tensor
from rooflang.programs.models.magi2_preview.operators import (
    FlashAttentionOperator, NpuFusedInferAttentionScoreOperator,
)
from rooflang.programs.presets.ascend910b import Ascend910B3
from .compat import BlockFactory, Parallel, OpType, QuantConfig
from .offload import apply_layerwise_offload
from . import blocks, mlp, embedding  # register the finite H3 recipe set

class H3Kernel(AnalyticalKernel):
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


def build_component(component, architecture, mm, seq_len, hardware, sp=1):
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
    nodes=component_nodes(component,architecture,mm,seq_len,sp)
    graph=ComputeGraph(); placement=Placement(hardware,graph)
    previous=previous_compute=None
    for index,node in enumerate(nodes):
        op=node.operator
        if ascend_fia and isinstance(op, FlashAttentionOperator):
            p = op.params
            packed = component in ('transformer', 'transformer_ref', 'transformer_text_refiner')
            boundaries = [used_rows] + ([p.q_len] if p.q_len > used_rows else [])
            op = NpuFusedInferAttentionScoreOperator(
                batch_size=p.batch_size, num_heads=p.num_heads,
                num_key_value_heads=p.kv_heads, head_dim=p.head_dim,
                q_len=p.q_len, kv_len=p.kv_len,
                input_layout='TND' if packed else 'BNSD',
                actual_seq_lengths=boundaries if packed else None,
                actual_seq_lengths_kv=boundaries.copy() if packed else None,
                scale=p.head_dim ** -0.5, causal=p.causal,
                quant_config=op.quant_config)
        if op.params is not None: op.params.validate()
        details=op.get_memory_details()
        meta={'node_name':node.node_name,'operator':type(op).__name__,'module':node.module.value,
              'layer_idx':node.layer_idx,'block_id':node.block_id,'tag_mask':node.tag_mask,
              'shape':asdict(op.params) if is_dataclass(op.params) else {},
              'compute_dtype':op.compute_dtype,'weight_storage_dtype':op.weight_storage_dtype,
              'communicate_dtype':op.communicate_dtype,
              'kv_storage_dtype':op.kv_storage_dtype,'component':component,'recipe_node_index':index,
              'compute_scope':'critical_rank','source_weight_saved':details.get('weight_saved',0),
              'activation_peak_bytes_proxy':details.get('activation',0)}
        if isinstance(op, NpuFusedInferAttentionScoreOperator):
            meta.update(source_operator='torch.ops.npu.npu_fused_infer_attention_score',
                        source_documentation=op.source_url,
                        cube_compute_dtype=op.compute_dtype,
                        vector_compute_dtype=op.vector_compute_dtype,
                        atten_mask_shape=op.atten_mask_shape, assumption=op.assumption,
                        attention_pairs=op.valid_pairs,
                        attention_alignment=('pinned_sglang_dit' if packed
                                             else 'analytical_fia_substitution'))
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
            # Reference memory peak excludes kv_cache, reported separately.
            kernel.csv_metadata['activation_peak_bytes_proxy']=details.get('activation',0)
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
