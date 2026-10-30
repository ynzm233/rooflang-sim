"""Lower MAGI analytical component recipes to native RoofLang graphs.

Recipes describe the critical active rank, as in the reference model. Compute
is placed on one representative GPU; collectives use all actual CP/EP devices
and RoofLang's fabric model. Composite calls remain ordered individual kernels.
Activation traffic is a byte-volume proxy, not a reconstructed tensor lifetime;
whole-stage capacity is supplied separately by the model's memory accounting.
"""
from __future__ import annotations

from enum import Enum
from fractions import Fraction
import math
from types import SimpleNamespace

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllGather, AllReduce, AllToAll, Broadcast
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes
from rooflang.runtime.simulator import Simulator

from .operators import CompositeOperator, OpType


class Parallel(Enum):
    CP = "cp"
    EP = "ep"
    SP = "sp"


class ModuleType(Enum):
    OTHER = "other"
    MLP = "mlp"
    MOE = "moe"
    ATTENTION = "attention"
    COMMUNICATION = "communication"


class BlockConfig:
    """Small recipe configuration; values live on the derived block class."""
    def __init__(self, **values):
        for key, value in values.items():
            if not hasattr(type(self), key):
                raise ValueError(f"Unknown MAGI block option: {key}")
            setattr(self, key, value)


class BaseBlock:
    def __init__(self, block_id, block_config, comp_config):
        self.block_id = block_id
        self.component_type = comp_config.get("component_type", "")
        self.block_optag = 0

    def _pre_build(self, strategy):
        # MAGI's source deployment uses the ordinary component-prefix tags.
        # Keep unclassified conditioning/resize/visual-encoder blocks at zero.
        tags = {"llm": 3, "vit": 128, "dit": 256,
                "vae_decoder": 1024, "text_encoder": 2048}
        self.block_optag = next((mask for prefix, mask in tags.items()
                                if self.component_type.startswith(prefix)), 0)


class Node(SimpleNamespace):
    pass


def _tensor(size_bytes, dtype, weight_id=None, *, traffic=False):
    return Tensor(dtype, (Fraction(size_bytes) / Fraction(dtype_bytes(dtype)),),
                  weight_id=weight_id, storage_bytes=0 if traffic else None)


class AnalyticalKernel(Kernel):
    """A parameter-derived cost operation, scheduled by the RoofLang DES."""
    def __init__(self, operator, metadata, weight_id):
        self.operator = operator
        self.csv_metadata = metadata
        details = operator.get_memory_details()
        self._activation = details.get("activation", 0) + details.get("kv_cache_read", 0)
        self._weight_read = details.get("weight_read", 0)
        self._weight_saved = details.get("weight_saved", 0)
        self.csv_metadata["activation_peak_bytes_proxy"] = self._activation
        self._mathematical_flops = operator.get_flops()
        component = operator.get_compute_component()
        self._compute_flops = (component.total_flops() if component is not None else 0)
        self._compute_dtype = component.dtype if component is not None else operator.compute_dtype
        self._compute_flops_by_dtype = {}
        if component is not None and component.unit_type == "mixed":
            for unit in ("cube", "vector"):
                dtype = component.get_unit_dtype(unit)
                self._compute_flops_by_dtype[dtype] = (
                    self._compute_flops_by_dtype.get(dtype, 0) + component.get_unit_flops(unit))
        else:
            self._compute_flops_by_dtype[self._compute_dtype] = self._compute_flops
        # The reference formula specifies aggregate activation traffic. A native
        # traffic buffer charges that byte volume exactly on symmetric HBM links.
        # The descriptor has zero physical storage: aggregate traffic is not a
        # live activation. The model-owned working-set scenario checks capacity.
        resident = self._weight_saved or self._weight_read
        weights = ({"weight": _tensor(resident, operator.weight_storage_dtype, weight_id)}
                   if resident else {})
        self.csv_metadata["source_weight_saved"] = self._weight_saved
        super().__init__(weights=weights,
                         outputs={"traffic": _tensor(self._activation, operator.compute_dtype,
                                                     traffic=True)})
        if resident:
            self.weight_read_fraction = self._weight_read / resident

    @property
    def flops(self):
        return self._mathematical_flops

    @property
    def flops_by_dtype(self):
        return self._compute_flops_by_dtype

    @property
    def input_bytes(self):
        return 0.0

    @property
    def loaded_weight_bytes(self):
        return self._weight_read

    def input_read_fraction(self, port):
        return 0.0 if port == "previous_traffic" else 1.0


class _NetworkOnly:
    """Network formulas already contain collective work; no GPU arithmetic."""
    @property
    def flops(self):
        return 0.0

    @property
    def input_bytes(self):
        return self.boundary_payload_bytes

    @property
    def output_bytes(self):
        return self.boundary_payload_bytes


class MagiAllToAll(_NetworkOnly, AllToAll):
    pass


class MagiAllGather(_NetworkOnly, AllGather):
    pass


class MagiAllReduce(_NetworkOnly, AllReduce):
    pass


class MagiBroadcast(_NetworkOnly, Broadcast):
    pass


def _metadata(node, component, subcall=None):
    result = {
        "node_name": node.node_name,
        "operator": type(node.operator).__name__,
        "op_type": "communication" if node.op_type is OpType.COMMUNICATION else "compute",
        "module": node.module.value,
        "layer_idx": node.layer_idx,
        "block_id": node.block_id,
        "shape": node.shape,
        "semantic": node.semantic,
        "assumption": node.assumption,
        "source_compute_dtype": node.source_compute_dtype,
        "component": component,
        "rank_scope": "critical_active_rank",
        "memory_model": "aggregate_activation_traffic_proxy",
        # Composite subcalls retain the source operator's storage declaration;
        # their local descriptors specify arithmetic precision only.
        "kv_storage_dtype": node.operator.kv_storage_dtype,
    }
    if hasattr(node, "tag_mask"):
        result["tag_mask"] = node.tag_mask
    if subcall is not None:
        result["parent_node_name"] = node.node_name
        result["subcall"] = subcall
        result["node_name"] = f"{node.node_name}/{subcall}"
    return result


def _lower(nodes, component, hardware, devices):
    graph = ComputeGraph()
    placement = Placement(hardware, graph)
    previous = previous_compute = None
    resident_weights = {}
    for index, node in enumerate(nodes):
        op = node.operator
        if node.op_type is OpType.COMMUNICATION:
            world = math.prod(op.net_list)
            if world > len(devices):
                raise ValueError(f"MAGI collective requires {world} GPUs, found {len(devices)}")
            name = type(op).__name__
            cls = {"AllToAllOperator": MagiAllToAll, "AllGatherOperator": MagiAllGather,
                   "AllReduceOperator": MagiAllReduce,
                   "TreeBroadcastOperator": MagiBroadcast}[name]
            payload = op.op_size * dtype_bytes(op.communicate_dtype)
            # A blocking binary tree has ceil(log2(W)) sequential critical hops.
            repeats = math.ceil(math.log2(world)) if name == "TreeBroadcastOperator" else 1
            kernels = []
            for hop in range(repeats):
                kernel = cls(payload, world)
                # Reference CSV traffic describes a collective's send/receive
                # interface once. Sequential tree hops affect network duration,
                # not that boundary counter; report it on the first hop only.
                kernel.boundary_payload_bytes = payload if hop == 0 else 0
                kernel.csv_metadata = _metadata(node, component,
                    f"tree_hop_{hop}" if repeats > 1 else None)
                kernel.csv_metadata["communicate_dtype"] = op.communicate_dtype
                kernel.csv_metadata["activation_peak_bytes_proxy"] = 0
                kernel.csv_metadata["traffic_scope"] = "collective_send_receive_boundary"
                for rank, device in enumerate(devices[:world]):
                    memory = hardware.find_local_memory(device)
                    if rank == 0 or not kernel._single_input:
                        tensor = _tensor(0, op.communicate_dtype)
                        kernel.inputs[str(rank)] = tensor
                        placement.set_tensor_memory(tensor, memory)
                    tensor = _tensor(0, op.communicate_dtype)
                    kernel.outputs[str(rank)] = tensor
                    placement.set_tensor_memory(tensor, memory)
                kernels.append(kernel)
        else:
            parts = ((stage.name, stage.operator()) for stage in op.get_stages()) \
                if isinstance(op, CompositeOperator) else [(None, op)]
            kernels = []
            for subcall_index, (label, part) in enumerate(parts):
                weight_id = f"{component}:{node.layer_idx}:{index}:{label}"
                details = part.get_memory_details()
                key = (node.layer_idx, type(op).__name__, part.weight_storage_dtype,
                       details.get("weight_read", 0))
                if not details.get("weight_saved", 0) and details.get("weight_read", 0):
                    if key not in resident_weights:
                        raise ValueError(f"Shared weight has no owning operation: {node.node_name}")
                    weight_id = resident_weights[key]
                elif details.get("weight_saved", 0):
                    resident_weights.setdefault(key, weight_id)
                metadata = _metadata(node, component, label)
                metadata["subcall_index"] = subcall_index
                kernels.append(AnalyticalKernel(part, metadata, weight_id))
        for kernel in kernels:
            kernel.csv_metadata["recipe_node_index"] = index
            kernel.csv_metadata["graph_node_index"] = len(graph.kernels)
            graph.add_kernel(kernel)
            if previous is not None:
                graph.add_control_edge(previous, kernel)
            if isinstance(kernel, AnalyticalKernel):
                if previous_compute is not None:
                    kernel.inputs["previous_traffic"] = previous_compute.outputs["traffic"]
                    graph.add_data_edge(previous_compute, kernel,
                                        {"traffic": "previous_traffic"})
                placement.set_kernel_device(kernel, devices[0])
                previous_compute = kernel
            previous = kernel
    graph.validate()
    placement.validate(graph)
    return graph, placement


def _context(mm, architecture, cp, ep, batch_size):
    return SimpleNamespace(
        deploy_strategies=None, multimodal_config={**mm, "architecture": architecture},
        parallel_strategies={Parallel.CP: cp, Parallel.EP: ep, Parallel.SP: 1},
        network_list={Parallel.CP: [cp], Parallel.EP: [ep]}, batch_size=batch_size,
    )


def _component_nodes(component, mm, architecture, cp, ep, batch_size, *, position=None):
    from .blocks import (
        Magi2PreviewLayer, Magi2RefinerLayer, Magi2PreviewIO, Magi2RefinerIO,
        Magi2ConditionBroadcast, Magi2LatentUpsample,
        Magi2TurboVAEDecoder, Magi2ImageVAEEncoder, Magi2AudioVAEDecoder,
    )
    cfg = architecture[component]
    context = _context(mm, architecture, cp, ep, batch_size)
    if component in ("preview_transformer", "refiner_transformer"):
        preview = component == "preview_transformer"
        layer_cls = Magi2PreviewLayer if preview else Magi2RefinerLayer
        io_cls = Magi2PreviewIO if preview else Magi2RefinerIO
        nodes = []
        if position in (None, "input"):
            block = io_cls(f"{component}-input-io", {"position": "input"}, cfg)
            nodes.extend(block.build(context, -1))
        if position is None:
            for layer_idx in range(cfg["num_layers"]):
                block = layer_cls(f"{component}-layer", {"layer_index": layer_idx}, cfg)
                nodes.extend(block.build(context, layer_idx))
        if position in (None, "output"):
            block = io_cls(f"{component}-output-io", {"position": "output"}, cfg)
            nodes.extend(block.build(context, cfg["num_layers"]))
        return nodes
    classes = {"condition_broadcast": Magi2ConditionBroadcast,
               "latent_upsample": Magi2LatentUpsample,
               "video_vae_decoder": Magi2TurboVAEDecoder,
               "image_vae_encoder": Magi2ImageVAEEncoder,
               "audio_vae_decoder": Magi2AudioVAEDecoder}
    if component not in classes:
        raise ValueError(f"Unknown MAGI accelerator component: {component}")
    return classes[component](component, {}, cfg).build(context, 0)


def build_component(component_name, multimodal_config, architecture, hardware, *,
                    cp=1, ep=1, batch_size=1):
    """Build a native graph/placement for one component invocation.

    ``architecture`` maps names to the local model component configuration.
    ``multimodal_config`` contains resolved geometry plus optional ``branch``,
    ``turbo_window_index`` and ``broadcast_kind``. The caller owns stage grouping
    and repetitions. Text-model CPU costs are deliberately handled separately.
    """
    devices = sorted((node for node in hardware.nodes
                      if isinstance(node, Compute) and node.kind == "gpu"),
                     key=lambda node: node.name)
    if not devices:
        raise ValueError("MAGI accelerator stages require a GPU")
    if type(cp) is not int or type(ep) is not int or cp < 1 or ep < 1:
        raise ValueError("MAGI CP and EP must be positive integers")
    if cp != ep:
        raise ValueError("MAGI uses the same device mesh for CP and EP; set CP=EP")
    if max(cp, ep) > len(devices):
        raise ValueError(f"MAGI CP={cp}/EP={ep} needs more than {len(devices)} available GPUs")
    for device in devices[:cp]:
        fabric = hardware.find_fabric(device, hardware.find_local_memory(device))
        if not math.isclose(fabric.src_to_dst_bandwidth_gbs, fabric.dst_to_src_bandwidth_gbs):
            raise ValueError("MAGI aggregate traffic formulas require equal HBM read/write bandwidth")
    if batch_size != 1:
        raise ValueError("The MAGI request geometry currently supports batch_size=1")
    mm = dict(multimodal_config)
    selected = {}
    # Packing gives unequal modality work to each rank. Select each synchronized
    # adapter phase independently using the same DES/hardware as the final graph.
    if component_name in ("preview_transformer", "refiner_transformer"):
        for position in ("input", "output"):
            costs = []
            for rank in range(cp):
                nodes = _component_nodes(component_name, {**mm, "adapter_rank": rank},
                    architecture, cp, ep, batch_size, position=position)
                graph, placement = _lower(nodes, component_name, hardware, devices)
                costs.append(Simulator(graph, placement, hardware).run().total_time_us)
            selected[f"adapter_{position}_rank"] = max(range(cp), key=costs.__getitem__)
            selected[f"adapter_{position}_rank_costs_us"] = costs
        mm.update(selected)
    nodes = _component_nodes(component_name, mm, architecture, cp, ep, batch_size)
    graph, placement = _lower(nodes, component_name, hardware, devices)
    graph.magi2_adapter_selection = selected
    return graph, placement
