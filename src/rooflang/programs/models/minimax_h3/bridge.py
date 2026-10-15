"""Lower source cost recipes to serial native graphs, preserving source scope."""
from dataclasses import asdict, is_dataclass
import math
from types import SimpleNamespace
from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.placement import Placement
from rooflang.language.kernels.comm import CommKernel
from rooflang.programs.models.magi2_preview.bridge import AnalyticalKernel, _tensor
from .compat import BlockFactory, Parallel, OpType
from . import blocks, mlp, embedding  # register the finite H3 recipe set

class H3Kernel(AnalyticalKernel):
    @property
    def flops_by_dtype(self):
        c = self.operator.get_compute_component()
        if c is None: return {'fp16': 0.0}
        if c.unit_type == 'mixed':
            return {c.dtype + ':cube': c.cube_flops, c.dtype + ':vector': c.vector_flops}
        return {c.dtype + ':' + c.unit_type: c.flops}

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
    return nodes


def build_component(component, architecture, mm, seq_len, hardware, sp=1):
    devices=sorted((n for n in hardware.nodes if isinstance(n,Compute) and n.kind=='gpu'),key=lambda n:n.name)
    nodes=component_nodes(component,architecture,mm,seq_len,sp)
    graph=ComputeGraph(); placement=Placement(hardware,graph)
    previous=previous_compute=None
    for index,node in enumerate(nodes):
        op=node.operator
        if op.params is not None: op.params.validate()
        details=op.get_memory_details()
        meta={'node_name':node.node_name,'operator':type(op).__name__,'module':node.module.value,
              'layer_idx':node.layer_idx,'block_id':node.block_id,'tag_mask':node.tag_mask,
              'shape':asdict(op.params) if is_dataclass(op.params) else {},
              'compute_dtype':'fp16','weight_storage_dtype':'fp16','communicate_dtype':'fp16',
              'kv_storage_dtype':'fp16','component':component,'recipe_node_index':index,
              'compute_scope':'critical_rank','source_weight_saved':details.get('weight_saved',0),
              'activation_peak_bytes_proxy':details.get('activation',0)}
        if node.op_type is OpType.COMMUNICATION:
            world=math.prod(op.net_list); payload=op.op_size*2
            kernel=H3Collective(payload,world);kernel.csv_metadata=meta
            for rank,device in enumerate(devices[:world]):
                mem=hardware.find_local_memory(device)
                for ports in (kernel.inputs,kernel.outputs):
                    tensor=_tensor(0,'fp16'); ports[str(rank)]=tensor
                    placement.set_tensor_memory(tensor,mem)
        else:
            kernel=H3Kernel(op,meta,f'{component}:{index}')
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
    graph.validate();placement.validate(graph)
    return graph,placement
