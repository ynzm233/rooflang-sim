"""MAGI-2 component recipes adapted from Markov_Infer_Sim.

Analytical topology is evaluated as native RoofLang kernels; no reference
repository, precomputed costs, or trace files are accessed at runtime.

Blocks assemble one invocation on a critical active rank. The application owns
stage repetition and the model YAML owns architecture, as for MiniMax-H3.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from .bridge import BaseBlock, ModuleType, BlockConfig, Node, Parallel
from .operators import (
    QuantConfig, get_dtype_size, OpType,
    AddOperator, SubOperator, MulOperator, MatmulOperator, GroupMatmulOperator,
    RmsNormOperator, RopeOperator, SiluOperator, ReduceSumOperator,
    AllToAllOperator, AllGatherOperator, AllReduceOperator,
    FlashAttentionSinkOperator, FlashAttentionOperator, FlashMHMoEOperator,
    MoERoutingOperator, SwiGLU7Operator, MHCOperator, TensorMoveOperator,
    InterpolateOperator, FourierFeaturesOperator, SigmoidGateOperator, ReluOperator,
    TreeBroadcastOperator,
)

class Magi2BlockConfig(BlockConfig):
    position: str = "input"
    layer_index: int = 0


def _quant(dtype):
    return QuantConfig(compute_dtype=dtype, weight_storage_dtype=dtype,
                       kv_storage_dtype=dtype, communicate_dtype=dtype)


def rank_modality_counts(shape, stage, branch="cond", cp_size=None):
    """Exact contiguous Ulysses slices of the source's ordered input pack.

    Preview packs positive then negative [video, audio, text, image-prefix,
    image] and finally TEXT padding. Refiner packs [video, text]; its window
    permutation changes only the order within the video region.
    """
    cp = shape.get("cp_size", 8) if cp_size is None else cp_size
    if type(cp) is not int or cp < 1:
        raise ValueError("cp_size must be a positive integer")
    if stage == "preview":
        segments = []
        image = int(shape.get("image_tokens", 0))
        for text_branch in ("positive", "negative"):
            segments.extend((("video", shape["preview_video_tokens"]),
                             ("audio", shape["audio_tokens"]),
                             ("text", shape[f"{text_branch}_text_tokens"])))
            if image:
                segments.extend((("text", 1), ("video", image)))
        total = int(shape["preview_padded_tokens"])
        padding = total - sum(count for _, count in segments)
        if padding < 0:
            raise ValueError("preview_padded_tokens is smaller than its source pack")
        segments.append(("text", padding))
    elif stage == "refiner":
        if branch not in ("cond", "uncond"):
            raise ValueError("refiner branch must be cond or uncond")
        total = int(shape[f"refiner_{branch}_tokens"])
        video = int(shape["refiner_video_tokens"])
        segments = [("video", video), ("text", total-video)]
    else:
        raise ValueError("stage must be preview or refiner")
    if any(count < 0 for _, count in segments):
        raise ValueError("modality counts cannot be negative")
    quotient, remainder = divmod(total, cp)
    counts = []
    start = 0
    for rank in range(cp):
        end = start + quotient + (rank < remainder)
        result = {"video": 0, "audio": 0, "text": 0}
        offset = 0
        for modality, count in segments:
            result[modality] += max(0, min(end, offset+count)-max(start, offset))
            offset += count
        counts.append(result)
        start = end
    return counts


class _Magi2Block(BaseBlock):
    def __init__(self, block_id, block_config, comp_config):
        self.cfg = dict(comp_config)
        super().__init__(block_id, block_config, comp_config)
        self.block_config = Magi2BlockConfig(**block_config)

    def _start(self, metadata, layer_idx):
        self.nodes = []
        self._pre_build(metadata.deploy_strategies)
        self.md, self.li = metadata, layer_idx
        self.mm = metadata.multimodal_config
        self.sh = self.mm["shape_summary"]
        self.cp = metadata.parallel_strategies.get(Parallel.CP, 1)
        self.ep = metadata.parallel_strategies.get(Parallel.EP, 1)
        self.batch = metadata.batch_size
        if metadata.parallel_strategies.get(Parallel.SP, 1) != 1:
            raise ValueError("MAGI uses CP Ulysses; set SP=1")
        return self.sh, self.mm, self.cp

    def _add(self, name, op, module=ModuleType.OTHER):
        node = Node(node_name=name, operator=op, op_type=op.OP_TYPE or OpType.COMPUTE,
                    module=module, tag_mask=self.block_optag, layer_idx=self.li, block_id=self.block_id)
        node.shape = getattr(op, "shape", str(asdict(op.params)) if op.params else "")
        node.semantic = name
        node.assumption = getattr(op, "assumption", "")
        node.source_compute_dtype = op.compute_dtype
        self.nodes.append(node)

    def _linear(self, name, rows, din, dout, *, banks=1, dtype="bf16", bias=False):
        if banks == 1:
            op = MatmulOperator(batch_seq=rows, in_features=din, out_features=dout,
                                has_bias=bias, quant_config=_quant(dtype))
        else:
            if bias:
                raise ValueError("banked MAGI projections are bias-free")
            quotient, remainder = divmod(rows, banks)
            op = GroupMatmulOperator(expert_token_list=[quotient + (i < remainder) for i in range(banks)],
                                     in_features=din, out_features=dout, quant_config=_quant(dtype))
            op.assumption = "one bank per token; balanced bank traffic proxy; all banks resident"
        self._add(name, op, ModuleType.MLP)

    def _norm(self, name, rows, dim, banks=1, shared=False, *, dtype="bf16", affine=True,
              output_dtype=None):
        self._add(name, RmsNormOperator(batch_seq=rows, dim=dim, affine=affine,
                  weight_banks=banks, shared_weight=shared, input_dtype=dtype, output_dtype=output_dtype or dtype,
                  quant_config=_quant(self.cfg.get("norm_dtype", "fp32"))))

    def _move(self, name, input_elements, output_elements=None, *, dtype="bf16", mode="copy",
              output_dtype=None, index_elements=0):
        self._add(name, TensorMoveOperator(mode=mode, input_elements=int(input_elements),
                  output_elements=int(input_elements if output_elements is None else output_elements),
                  input_dtype=dtype, output_dtype=output_dtype or dtype,
                  index_elements=int(index_elements), quant_config=_quant(dtype)))

    def _activation(self, name, rows, dim, dtype, kind="silu"):
        if kind == "silu":
            op = SiluOperator(batch_seq=rows, dim=dim, quant_config=_quant(dtype))
        elif kind == "relu":
            op = ReluOperator(element_count=rows * dim, quant_config=_quant(dtype))
        else:
            raise ValueError(f"unsupported activation: {kind}")
        self._add(name, op)

    def _residual(self, name, elements, dtype, *, input_dtype_x=None, input_dtype_y=None):
        self._add(name, AddOperator(element_count_x=elements, element_count_y=elements,
                                   input_dtype_x=input_dtype_x, input_dtype_y=input_dtype_y,
                                   quant_config=_quant(dtype)))

    def _mean(self, name, output_elements, group_size, dtype):
        self._add(name, ReduceSumOperator(batch_size=1, seq_len=output_elements, hidden_size=1,
                                          hc_mult=group_size, normalize=True,
                                          quant_config=_quant(dtype)))

    def _affine(self, name, elements, channels, dtype, inverse=False):
        # Source latent transforms use channel constants, not trainable weights.
        first, second = (MulOperator, AddOperator) if inverse else (SubOperator, MulOperator)
        for suffix, cls in (("Scale" if inverse else "Center", first),
                            ("Shift" if inverse else "Scale", second)):
            self._add(name + "_" + suffix, cls(element_count_x=elements, element_count_y=channels,
                                               quant_config=_quant(dtype)))

    def _comm(self, name, elements, dim=Parallel.CP, dtype="bf16", gather=False):
        if self.md.parallel_strategies.get(dim, 1) <= 1:
            return
        cls = AllGatherOperator if gather else AllToAllOperator
        self._add(name, cls(op_size=int(elements), net_list=self.md.network_list[dim],
                            quant_config=_quant(dtype)), ModuleType.COMMUNICATION)

    def _ffn(self, name, rows, dim, inner, banks=1, *, refiner=False):
        dtype = self.cfg["weight_dtype"]
        self._linear(name + "_GateUp", rows, dim, 2 * inner, banks=banks, dtype=dtype)
        activation_dtype = "fp32" if refiner else dtype
        if refiner:
            self._move(name + "_GateUpToFP32", rows * 2 * inner, dtype=dtype,
                       output_dtype="fp32", mode="cast")
        self._add(name + "_SwiGLU7", SwiGLU7Operator(batch_seq=rows, dim=inner,
                  input_dtype=activation_dtype, output_dtype=activation_dtype,
                  quant_config=_quant("fp32")), ModuleType.MLP)
        if refiner:
            self._move(name + "_ActivationToBF16", rows * inner, dtype="fp32",
                       output_dtype=dtype, mode="cast")
        self._linear(name + "_Down", rows, inner, dim, banks=banks, dtype=dtype)
        if refiner:
            self._move(name + "_DownToFP32", rows * dim, dtype=dtype,
                       output_dtype="fp32", mode="cast")

    def _attention(self, rows, total, pairs, banks, sink=False):
        cfg = self.cfg
        dim, hd = cfg["hidden_size"], cfg["head_dim"]
        qheads, kvheads = cfg["num_attention_heads"], cfg["num_kv_heads"]
        if qheads % self.cp or kvheads % self.cp:
            raise ValueError("CP must divide query and KV head counts")
        qdim, kvdim = qheads * hd, kvheads * hd
        dtype = cfg["weight_dtype"]
        # Refiner enters layer zero in BF16; its MLP returns FP32, which makes
        # all later residual streams FP32. Every projection still computes BF16.
        residual_dtype = dtype if sink or self.block_config.layer_index == 0 else "fp32"
        self._norm("Attn_PreNorm", rows, dim, banks, dtype=residual_dtype)
        if residual_dtype != dtype:
            self._move("Attn_PreNormToBF16", rows * dim, dtype=residual_dtype,
                       output_dtype=dtype, mode="cast")
        self._linear("Attn_QKV", rows, dim, qdim + 2 * kvdim, banks=banks, dtype=dtype)
        self._linear("Attn_GateProjection", rows, dim, qheads, banks=banks, dtype=dtype)
        if sink:
            for label, width in (("Q", qdim), ("K", kvdim), ("Gate", qheads)):
                self._move("Attn_" + label + "Clone", rows * width, dtype=dtype)
        else:
            self._move("Attn_QKVToFP32", rows * (qdim + 2 * kvdim), dtype=dtype,
                       output_dtype="fp32", mode="cast")
            self._move("Attn_GateToFP32", rows * qheads, dtype=dtype,
                       output_dtype="fp32", mode="cast")
        norm_input = dtype if sink else "fp32"
        self._norm("Attn_QNorm", rows * qheads, hd, banks, dtype=norm_input, output_dtype="fp32")
        self._norm("Attn_KNorm", rows * kvheads, hd, banks, dtype=norm_input, output_dtype="fp32")
        for label, width, source_dtype in (("Q", qdim, "fp32"), ("K", kvdim, "fp32"),
                                            ("V", kvdim, dtype if sink else "fp32")):
            self._move("Attn_" + label + "InversePermute", rows * width, dtype=source_dtype,
                       mode="permute", index_elements=rows)
        # ElementWiseFourierEmbed uses 3 coordinates * (fourier_dim // 8)
        # bands, each with sin/cos. Only 96 of the 128 head channels rotate.
        rotary_dim = int(cfg["fourier_dim"] // 8 * 3 * 2)
        if not 0 < rotary_dim <= hd:
            raise ValueError("Fourier rotary width must fit the attention head")
        self._add("Attn_QK_RoPE", RopeOperator(batch_seq=rows * (qheads + kvheads), dim=rotary_dim,
                                             quant_config=_quant("fp32")), ModuleType.ATTENTION)
        if sink:
            # Preview's torch implementation concatenates rotated and untouched
            # channels after producing the rotary prefix.
            self._move("Attn_QK_RoPEConcat", rows * (qdim + kvdim), dtype="fp32", mode="concat")
        elif rotary_dim < hd:
            tail = TensorMoveOperator(mode="copy", input_elements=rows * (qheads + kvheads) * (hd-rotary_dim),
                       output_elements=rows * (qheads + kvheads) * (hd-rotary_dim),
                       input_dtype="fp32", output_dtype="fp32", quant_config=_quant("fp32"))
            tail.assumption = "unrotated output tail traffic lower bound; external rotary kernel fusion uncalibrated"
            self._add("Attn_QK_RoPETailCopy", tail, ModuleType.ATTENTION)
        cast_width = qdim + kvdim if sink else qdim + 2 * kvdim
        self._move("Attn_QKToBF16" if sink else "Attn_QKVToBF16", rows * cast_width,
                   dtype="fp32", output_dtype=dtype, mode="cast")
        self._comm("Attn_CP_QKV_AllToAll", rows * (qdim + 2 * kvdim), dtype=dtype)
        params = dict(batch_size=self.batch, num_heads=qheads // self.cp, kv_heads=kvheads // self.cp,
                      q_len=total, kv_len=total, head_dim=hd, attention_pairs=pairs,
                      softmax_compute_multiplier=float(self.mm.get("attention_softmax_compute_multiplier", 1)),
                      quant_config=_quant(cfg["attention_dtype"]))
        if sink:
            op = FlashAttentionSinkOperator(**params, sink_weight_heads=qheads)
        else:
            op = FlashAttentionOperator(**params)
        self._add("Attn_FlashVarlenWithSink" if sink else "Attn_FlexWindow", op, ModuleType.ATTENTION)
        self._comm("Attn_CP_Output_AllToAll", self.batch * total * qdim // self.cp, dtype=dtype)
        self._move("Attn_OutputPermute", rows * qdim, dtype=dtype, mode="permute", index_elements=rows)
        self._add("Attn_SigmoidGate", SigmoidGateOperator(batch_seq=rows, num_heads=qheads,
                  head_dim=hd, input_dtype=dtype, output_dtype=dtype if sink else "fp32",
                  quant_config=_quant(dtype if sink else "fp32")), ModuleType.ATTENTION)
        if not sink:
            self._move("Attn_GatedOutputToBF16", rows * qdim, dtype="fp32",
                       output_dtype=dtype, mode="cast")
        self._linear("Attn_OutputProjection", rows, qdim, dim, banks=banks, dtype=dtype)


class Magi2PreviewLayer(_Magi2Block):
    def _mhc_pre(self, prefix, rows, banks, shared_norm=False):
        dim, n = self.cfg["hidden_size"], self.cfg["mhc_multiplier"]
        self._norm(prefix + "_Norm", rows, n * dim, banks, shared_norm, dtype=self.cfg["weight_dtype"], output_dtype="fp32")
        self._linear(prefix + "_Phi", rows, n * dim, 2 * n + n * n, dtype=self.cfg["norm_dtype"])
        for suffix, phase in (("Coefficients", "coefficients"), ("Sinkhorn", "sinkhorn"), ("Hpre", "pre")):
            self._mhc_phase(prefix + "_" + suffix, phase, rows)

    def _mhc_phase(self, name, phase, rows):
        self._add(name, MHCOperator(phase=phase, batch_seq=rows, hidden_size=self.cfg["hidden_size"],
                  streams=self.cfg["mhc_multiplier"], iterations=self.cfg["mhc_sinkhorn_iterations"],
                  input_dtype=self.cfg["weight_dtype"] if phase in ("pre", "post") else self.cfg["norm_dtype"],
                  output_dtype=self.cfg["weight_dtype"] if phase in ("pre", "post") else self.cfg["norm_dtype"],
                  coefficient_dtype=self.cfg["norm_dtype"], quant_config=_quant(self.cfg["norm_dtype"])))

    def build(self, metadata, layer_idx=0):
        self._start(metadata, layer_idx)
        cfg = self.cfg
        total = int(self.sh["preview_padded_tokens"])
        rows = math.ceil(total / self.cp) * self.batch
        idx, dim = self.block_config.layer_index, cfg["hidden_size"]
        banks = cfg["modality_banks"] if idx in cfg["modality_bank_layers"] else 1
        self._mhc_pre("MHC_Attn", rows, banks)
        self._attention(rows, total, self.sh["preview_attention_pairs"], banks, sink=True)
        self._mhc_phase("MHC_Attn_ResidualMixHpost", "post", rows)
        self._mhc_pre("MHC_MLP", rows, banks, shared_norm=True)
        dense = idx in cfg["dense_ffn_layers"]
        self._norm("MLP_PreNorm", rows, dim, banks if dense else cfg["modality_banks"], dtype=cfg["weight_dtype"])
        if dense:
            self._ffn("Dense", rows, dim, cfg["intermediate_size"], banks)
        else:
            self._moe(rows)
        self._mhc_phase("MHC_MLP_ResidualMixHpost", "post", rows)
        return self.nodes

    def _moe(self, rows):
        cfg = self.cfg
        dim, dhead = cfg["hidden_size"], cfg["expert_head_dim"]
        heads = math.ceil(cfg["num_moe_heads"] / self.ep)
        experts, topk, inner = cfg["experts_per_head"], cfg["experts_per_token"], cfg["expert_hidden_size"]
        routed_rows = rows * self.ep
        padded_heads = heads * self.ep
        self._linear("MoE_Split", rows, dim, cfg["num_moe_heads"] * dhead, dtype=cfg["weight_dtype"])
        if padded_heads != cfg["num_moe_heads"]:
            # torch.cat reads real heads and the expanded zero buffer.
            self._move("MoE_PadHeads", rows * padded_heads * dhead,
                       dtype=cfg["weight_dtype"], mode="concat")
        self._comm("MoE_EP_Dispatch", rows * padded_heads * dhead, Parallel.EP, dtype=cfg["weight_dtype"])
        self._linear("MoE_RouterFP32", routed_rows * heads, dhead, experts, banks=heads, dtype=cfg["router_dtype"])
        self._add("MoE_Routing", MoERoutingOperator(tokens=routed_rows, num_heads=heads, num_experts=experts,
                  top_k=topk, score_func="sigmoid", route_norm=True, has_bias=True,
                  quant_config=_quant(cfg["router_dtype"])), ModuleType.MOE)
        self._add("MoE_FlashMHMoE", FlashMHMoEOperator(tokens=routed_rows, num_heads=heads, num_experts=experts,
                  top_k=topk, head_dim=dhead, intermediate_size=inner,
                  active_experts_per_head=int(self.mm.get("moe_active_experts_per_head", experts)),
                  compute_amplification=float(self.mm.get("moe_compute_amplification", 1)),
                  deterministic=self.mm.get("deterministic", False),
                  quant_config=_quant(cfg["weight_dtype"])), ModuleType.MOE)
        self._comm("MoE_EP_Undispatch", rows * padded_heads * dhead, Parallel.EP, dtype=cfg["weight_dtype"])
        if padded_heads != cfg["num_moe_heads"]:
            # Source slices heads and reshapes the last two dimensions as a view.
            self._add("MoE_UnpadHeads", TensorMoveOperator(mode="crop",
                      input_elements=rows * padded_heads * dhead,
                      output_elements=rows * cfg["num_moe_heads"] * dhead,
                      input_dtype=cfg["weight_dtype"], output_dtype=cfg["weight_dtype"], materialized=False,
                      quant_config=_quant(cfg["weight_dtype"])))
        self._linear("MoE_Merge", rows, cfg["num_moe_heads"] * dhead, dim, dtype=cfg["weight_dtype"])
        if cfg["num_shared_experts"] != 2:
            raise ValueError("MAGI requires global and modality shared experts")
        self._ffn("SharedGlobal", rows, dim, inner)
        self._ffn("SharedModality", rows, dim, inner, cfg["modality_banks"])
        self._add("MoE_CombineBranches", ReduceSumOperator(batch_size=1, seq_len=rows, hc_mult=3,
                  hidden_size=dim, quant_config=_quant(cfg["weight_dtype"])), ModuleType.MOE)


class Magi2RefinerLayer(_Magi2Block):
    def build(self, metadata, layer_idx=0):
        self._start(metadata, layer_idx)
        branch = self.mm.get("branch", "cond")
        total = int(self.sh[f"refiner_{branch}_tokens"])
        rows = math.ceil(total / self.cp) * self.batch
        banks = self.cfg["modality_banks"] if self.block_config.layer_index in self.cfg["modality_bank_layers"] else 1
        dim = self.cfg["hidden_size"]
        self._attention(rows, total, self.sh[f"refiner_{branch}_attention_pairs"], banks)
        residual_dtype = self.cfg["weight_dtype"] if self.block_config.layer_index == 0 else "fp32"
        self._residual("Attn_Residual", rows * dim, residual_dtype,
                       input_dtype_x=residual_dtype, input_dtype_y=self.cfg["weight_dtype"])
        self._norm("MLP_PreNorm", rows, dim, banks, dtype=residual_dtype)
        if residual_dtype != self.cfg["weight_dtype"]:
            self._move("MLP_PreNormToBF16", rows * dim, dtype=residual_dtype,
                       output_dtype=self.cfg["weight_dtype"], mode="cast")
        self._ffn("Dense", rows, dim, self.cfg["intermediate_size"], banks, refiner=True)
        self._residual("MLP_Residual", rows * dim, "fp32",
                       input_dtype_x=residual_dtype, input_dtype_y="fp32")
        return self.nodes


class _Magi2IO(_Magi2Block):
    preview = True

    def build(self, metadata, layer_idx=0):
        self._start(metadata, layer_idx)
        cfg = self.cfg
        branch = self.mm.get("branch", "cond")
        ranks = rank_modality_counts(self.sh, "preview" if self.preview else "refiner", branch, self.cp)
        rank = self.mm.get(f"adapter_{self.block_config.position}_rank", self.mm.get("adapter_rank", 0))
        if type(rank) is not int or not 0 <= rank < self.cp:
            raise ValueError("adapter rank must be an integer within the CP mesh")
        local = ranks[rank]
        total = self.sh["preview_padded_tokens"] if self.preview else self.sh[f"refiner_{branch}_tokens"]
        dim = cfg["hidden_size"] * (cfg["mhc_multiplier"] if self.preview else 1)
        dtype = cfg["adapter_dtype"]
        rows = sum(local.values()) * self.batch
        if self.block_config.position == "output":
            self._move("Adapter_ModalityInversePermute", rows * dim,
                       dtype=cfg["weight_dtype"] if self.preview else "fp32",
                       mode="permute", index_elements=rows)
        for name in ("Video", "Audio", "Text"):
            width = cfg[name.lower() + "_dim"]
            rows = local[name.lower()] * self.batch
            if not rows or (self.block_config.position == "output" and name == "Text"):
                continue
            if self.block_config.position == "input":
                self._linear("Adapter_" + name + "_Input", rows, width, dim, dtype=dtype, bias=True)
            else:
                source_dtype = cfg["weight_dtype"] if self.preview else "fp32"
                self._move("Adapter_" + name + "_Select", rows * dim, dtype=source_dtype,
                           mode="gather", index_elements=rows)
                if source_dtype != dtype:
                    self._move("Adapter_" + name + "_ToFP32", rows * dim, dtype=source_dtype,
                               output_dtype=dtype, mode="cast")
                self._norm("Adapter_" + name + "_FinalNorm", rows, dim, dtype=dtype)
                self._linear("Adapter_" + name + "_Output", rows, dim, width, dtype=dtype)
        rows = sum(local.values()) * self.batch
        if self.block_config.position == "input":
            self._add("Adapter_FourierCoordinates", FourierFeaturesOperator(batch_seq=rows,
                      num_bands=cfg["fourier_dim"] // 8, quant_config=_quant("fp32")))
            if self.preview:
                self._move("Adapter_TimeOverwrite", rows * cfg["time_dim"], dtype="fp32")
            self._move("Adapter_InputToBF16", rows * dim, dtype=dtype,
                       output_dtype=cfg["weight_dtype"], mode="cast")
            self._move("Adapter_ModalityPermute", rows * dim, dtype=cfg["weight_dtype"],
                       mode="permute", index_elements=rows)
            if self.preview:
                for parallel in (Parallel.CP, Parallel.EP):
                    if self.md.parallel_strategies.get(parallel, 1) > 1:
                        self._add(f"PaddingMAX_{parallel.value}_int64", AllReduceOperator(
                            op_size=2, net_list=self.md.network_list[parallel], quant_config=_quant("int32")),
                            ModuleType.COMMUNICATION)
        else:
            self._comm("Output_CP_AllGather", total * self.batch * cfg["audio_dim"], dtype=dtype, gather=True)
        return self.nodes


class Magi2PreviewIO(_Magi2IO):
    pass


class Magi2RefinerIO(_Magi2IO):
    preview = False


class Magi2ConditionBroadcast(_Magi2Block):
    def build(self, metadata, layer_idx=0):
        self._start(metadata, layer_idx)
        kind = self.mm.get("broadcast_kind", "all")
        if kind not in ("all", "text", "image"):
            raise ValueError("broadcast_kind must be all, text or image")
        if self.cp == 1:
            return self.nodes
        if kind in ("all", "text"):
            for branch in ("positive", "negative"):
                count = self.sh[f"{branch}_text_tokens"]
                payloads = (("ndim_int64", 8), ("shape_int64", 16),
                            ("embedding_bf16", count * self.cfg["text_dim"] * get_dtype_size(self.cfg["text_dtype"])))
                for suffix, payload in payloads:
                    self._add(f"Text_{branch}_{suffix}", TreeBroadcastOperator(
                        int(payload), self.md.network_list[Parallel.CP]), ModuleType.COMMUNICATION)
        if kind in ("all", "image") and self.sh.get("image_tokens", 0):
            self._add("Image_latent_fp32", TreeBroadcastOperator(
                int(self.sh["image_tokens"] * self.cfg["video_dim"] * get_dtype_size(self.cfg["video_dtype"])),
                self.md.network_list[Parallel.CP]), ModuleType.COMMUNICATION)
        return self.nodes


class Magi2LatentUpsample(_Magi2Block):
    def build(self, metadata, layer_idx=0):
        self._start(metadata, layer_idx)
        self._add("Latent_TrilinearUpsample", InterpolateOperator(
            input_shape=self.sh["preview_latent_shape"], output_shape=self.sh["refiner_latent_shape"],
            mode=self.cfg["mode"], align_corners=self.cfg["align_corners"],
            quant_config=_quant(self.cfg["weight_dtype"])))
        return self.nodes


# Keep public block names stable while separating transformer and codec topology.
from .codec_blocks import (  # noqa: E402
    Magi2TurboVAEDecoder, Magi2ImageVAEEncoder, Magi2AudioVAEDecoder,
)
