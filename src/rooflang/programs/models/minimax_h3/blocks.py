"""Cost recipe adapted from model_Sim; see SOURCES.json."""
from __future__ import annotations
import math
from typing import Any, Dict, List
from collections import Counter
from dataclasses import replace
from .compat import *
from .attention import MHABlock
from .merger import PatchMergerBlock
"""MiniMax-H3 FL2VA/Ref2VA operator blocks.

The transformer implementation is not present in the source checkout.  Its
attention/FFN layers are therefore composed from the repository's existing
generic blocks in the model YAML. This module covers code-visible packed I/O
projections and the released video/audio VAE main operators.
"""

class _MiniMaxH3Block(BaseBlock):

    def __init__(self, block_id: str, block_config: dict[str, Any], comp_config: dict[str, Any]):
        self.raw_block_config = dict(block_config)
        self.raw_comp_config = dict(comp_config)
        super().__init__(block_id, block_config, comp_config)

    def _add(self, name: str, operator, module: ModuleType, layer_idx: int) -> None:
        self.nodes.append(Node(node_name=name, operator=operator, op_type=OpType.COMPUTE, module=module, tag_mask=self.block_optag, layer_idx=layer_idx, block_id=self.block_id))

    def _conv1d(self, name: str, batch: int, c_in: int, length: int, c_out: int, kernel: int, stride: int, pad: int, layer_idx: int) -> int:
        self._add(name, Conv1dOperator(batch=batch, c_in=c_in, L_in=max(1, length), c_out=c_out, kernel_size=kernel, stride=stride, pad=pad), ModuleType.VAEDECODER, layer_idx)
        return max(1, (length + 2 * pad - kernel) // stride + 1)

@BlockFactory.register('minimax-h3-qwen-vision-patch-embed')
class MiniMaxH3QwenVisionPatchEmbedBlock(_MiniMaxH3Block):
    """Qwen3-VL 2x16x16 Conv3D represented as one linear op per local patch."""

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        self._add('QwenVision_PatchEmbed', MatmulOperator(batch_seq=metadata.batch_size * metadata.seq_len, in_features=3 * 2 * 16 * 16, out_features=int(self.raw_comp_config['hidden_size'])), ModuleType.TEXTENCODER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-qwen-vision-attn')
class MiniMaxH3QwenVisionAttentionBlock(MHABlock):
    """Packed QKV/projection with independent attention documents (cu_seqlens)."""

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        if metadata.parallel_strategies.get(Parallel.CP, 1) != 1:
            raise ValueError('MiniMax-H3 segmented Qwen vision requires CP=1')
        super().build(metadata, layer_idx)
        segments = Counter(metadata.multimodal_config['qwen_attention_segments'])
        nodes = []
        for node in self.nodes:
            if node.node_name in ('Attn_QNorm', 'Attn_KNorm'):
                continue
            if node.node_name == 'Attn_RmsNorm':
                node = replace(node, node_name='QwenVision_LayerNorm', operator=LayerNormOperator(batch_seq=metadata.batch_size * metadata.seq_len, dim=self.hidden_size))
            if isinstance(node.operator, FlashAttentionOperator):
                for length, count in segments.items():
                    params = vars(node.operator.params).copy()
                    params.update(q_len=length, kv_len=length, batch_size=metadata.batch_size * count)
                    nodes.append(replace(node, node_name=f'QwenVision_FA_S{length}', operator=FlashAttentionOperator(**params)))
            else:
                nodes.append(node)
        self.nodes = nodes
        return self.nodes

@BlockFactory.register('minimax-h3-qwen-deepstack-merger')
class MiniMaxH3QwenDeepstackMergerBlock(PatchMergerBlock):

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        super().build(metadata, layer_idx)
        self.nodes[0].operator = LayerNormOperator(batch_seq=metadata.batch_size * metadata.seq_len // 4, dim=self.hidden_size * 4)
        return self.nodes

@BlockFactory.register('minimax-h3-qwen-deepstack-add')
class MiniMaxH3QwenDeepstackAddBlock(_MiniMaxH3Block):

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        shards = metadata.parallel_strategies.get(Parallel.SP, 1)
        rows = math.ceil(metadata.multimodal_config['qwen_raw_visual_tokens'] / 4 / shards)
        for index in (8, 16, 24):
            self._add(f'QwenDeepstack_Add_{index}', AddOperator(element_count_x=metadata.batch_size * rows * 5120, element_count_y=metadata.batch_size * rows * 5120), ModuleType.TEXTENCODER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-rmsnorm')
class MiniMaxH3RmsNormBlock(_MiniMaxH3Block):

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        self._add('TextRefiner_FinalNorm', RmsNormOperator(batch_seq=metadata.batch_size * metadata.seq_len, dim=int(self.raw_comp_config['hidden_size'])), ModuleType.OTHER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-packed-io')
class MiniMaxH3PackedIOBlock(_MiniMaxH3Block):
    """Input/output projections around the shared packed transformer rows."""

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        mm = metadata.multimodal_config
        shards = metadata.parallel_strategies.get(Parallel.SP, 1) * metadata.parallel_strategies.get(Parallel.CP, 1)
        tp = metadata.parallel_strategies.get(Parallel.TP, 1)
        hidden = int(self.raw_comp_config['hidden_size'])

        def local(rows: int) -> int:
            return max(1, math.ceil(metadata.batch_size * rows / shards))
        video_rows = mm['condition_video_rows'] + mm['target_video_rows']
        audio_rows = mm['condition_audio_rows'] + mm['target_audio_rows']
        text_rows = mm['text_rows']
        input_projections = (('Packed_Video_In', video_rows, 96, hidden // tp), ('Packed_Audio_In', audio_rows, 32, hidden // tp), ('Packed_Text_In', text_rows, 5120, hidden // tp))
        for name, rows, in_features, out_features in input_projections:
            if rows == 0:
                continue
            self._add(name, MatmulOperator(batch_seq=local(rows), in_features=in_features, out_features=out_features), ModuleType.OTHER, layer_idx)
        timestep_rows = mm['max_adaln_timestep_rows']
        self._add('TimeEmbed_Linear1', MatmulOperator(batch_seq=timestep_rows, in_features=256, out_features=hidden // tp), ModuleType.OTHER, layer_idx)
        self._add('TimeEmbed_SiLU', SiluOperator(batch_seq=timestep_rows, dim=hidden), ModuleType.OTHER, layer_idx)
        self._add('TimeEmbed_Linear2', MatmulOperator(batch_seq=timestep_rows, in_features=hidden, out_features=2688 // tp), ModuleType.OTHER, layer_idx)
        self._add('Final_AdaLN_SiLU', SiluOperator(batch_seq=timestep_rows, dim=2688), ModuleType.OTHER, layer_idx)
        self._add('Final_AdaLN_Projection', MatmulOperator(batch_seq=timestep_rows, in_features=2688, out_features=2 * hidden // tp), ModuleType.OTHER, layer_idx)
        packed_local_rows = metadata.batch_size * metadata.seq_len
        self._add('Final_RmsNorm', RmsNormOperator(batch_seq=packed_local_rows, dim=hidden), ModuleType.OTHER, layer_idx)
        for name, operator in (('Final_Add_Scale', AddOperator(element_count_x=packed_local_rows * hidden, element_count_y=1)), ('Final_Mul_Scale', MulOperator(element_count_x=packed_local_rows * hidden, element_count_y=packed_local_rows * hidden)), ('Final_Add_Shift', AddOperator(element_count_x=packed_local_rows * hidden, element_count_y=packed_local_rows * hidden))):
            self._add(name, operator, ModuleType.OTHER, layer_idx)
        for name, out_features in (('Final_Video_Out', 96), ('Final_Audio_Out', 32)):
            self._add(name, MatmulOperator(batch_seq=packed_local_rows, in_features=hidden // tp, out_features=out_features), ModuleType.OTHER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-adaln-projection')
class MiniMaxH3AdaLNProjectionBlock(_MiniMaxH3Block):
    """Per-block 2688 -> 3 modalities x 6 modulation vectors."""

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        tp = metadata.parallel_strategies.get(Parallel.TP, 1)
        hidden = int(self.raw_comp_config['hidden_size'])
        timestep_rows = metadata.multimodal_config['max_adaln_timestep_rows']
        self._add('DiT_AdaLN_SiLU', SiluOperator(batch_seq=timestep_rows, dim=2688), ModuleType.OTHER, layer_idx)
        self._add('DiT_AdaLN_Projection', MatmulOperator(batch_seq=timestep_rows, in_features=2688, out_features=3 * 6 * hidden // tp), ModuleType.OTHER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-video-vae-encoder')
class MiniMaxH3VideoVAEEncoderBlock(_MiniMaxH3Block):
    """Released EncoderFCN3D main-op graph for visual conditions."""
    CHANNELS = (128, 256, 256, 512, 512, 1024)
    SPACE_DOWN = (2, 2, 2, 2, 1, 1)
    TIME_DOWN = (1, 2, 2, 1, 1, 1)

    def _conv3d(self, name: str, batch: int, c_in: int, c_out: int, t: int, h: int, w: int, stride_t: int, stride_h: int, stride_w: int, kernel: int, layer_idx: int) -> None:
        pad = kernel // 2
        self._add(name, Conv3dOperator(batch=batch, c_in=c_in, c_out=c_out, d_in=max(1, t), h_in=max(1, h), w_in=max(1, w), kernel_t=kernel, kernel_h=kernel, kernel_w=kernel, stride_t=stride_t, stride_h=stride_h, stride_w=stride_w, pad_t=pad, pad_h=pad, pad_w=pad), ModuleType.OTHER, layer_idx)

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        for ref_idx, ref in enumerate(metadata.multimodal_config['visual_conditions']):
            t, h, w = (ref['frames'], ref['height'], ref['width'])
            c = 3
            self._conv3d(f'Ref{ref_idx}_ConvIn', 1, c, 128, t, h, w, 1, 1, 1, 3, layer_idx)
            c = 128
            for level, out_c in enumerate(self.CHANNELS):
                for res in range(2):
                    in_c = c
                    self._add(f'Ref{ref_idx}_L{level}_R{res}_Norm1', RmsNormOperator(batch_seq=max(1, t * h * w), dim=in_c), ModuleType.OTHER, layer_idx)
                    self._conv3d(f'Ref{ref_idx}_L{level}_R{res}_Conv1', 1, in_c, out_c, t, h, w, 1, 1, 1, 3, layer_idx)
                    self._conv3d(f'Ref{ref_idx}_L{level}_R{res}_Conv2', 1, out_c, out_c, t, h, w, 1, 1, 1, 3, layer_idx)
                    if in_c != out_c:
                        self._conv3d(f'Ref{ref_idx}_L{level}_R{res}_Shortcut', 1, in_c, out_c, t, h, w, 1, 1, 1, 1, layer_idx)
                    c = out_c
                st, ss = (self.TIME_DOWN[level], self.SPACE_DOWN[level])
                if st * ss > 1:
                    self._conv3d(f'Ref{ref_idx}_L{level}_Downsample', 1, c, c, t, h, w, st, ss, ss, 3, layer_idx)
                    t = max(1, math.ceil(t / st))
                    h = max(1, math.ceil(h / ss))
                    w = max(1, math.ceil(w / ss))
            self._conv3d(f'Ref{ref_idx}_ConvOut', 1, c, 48, t, h, w, 1, 1, 1, 3, layer_idx)
            self._conv3d(f'Ref{ref_idx}_QuantConv', 1, 48, 48, t, h, w, 1, 1, 1, 1, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-ref2va-audio-vae-encoder')
class Ref2VAAudioVAEEncoderBlock(_MiniMaxH3Block):
    """DAC audio encoder main-op graph; stereo channels form batch=2."""
    RATES = (2, 4, 4, 5, 5)

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        for ref_idx, latent_len in enumerate(metadata.multimodal_config['audio_reference_latents']):
            length = max(1, latent_len * math.prod(self.RATES))
            channels = 64
            length = self._conv1d(f'AudioRef{ref_idx}_ConvIn', 2, 1, length, channels, 7, 1, 3, layer_idx)
            for level, rate in enumerate(self.RATES):
                out_channels = channels * 2
                for dilation in (1, 3, 9):
                    self._conv1d(f'AudioRef{ref_idx}_L{level}_Res{dilation}_Conv7', 2, channels, length, channels, 7, 1, 3, layer_idx)
                    self._conv1d(f'AudioRef{ref_idx}_L{level}_Res{dilation}_Conv1', 2, channels, length, channels, 1, 1, 0, layer_idx)
                length = self._conv1d(f'AudioRef{ref_idx}_L{level}_Down', 2, channels, length, out_channels, 2 * rate, rate, math.ceil(rate / 2), layer_idx)
                channels = out_channels
            self._conv1d(f'AudioRef{ref_idx}_Latent', 2, channels, length, 2048, 3, 1, 1, layer_idx)
            self._add(f'AudioRef{ref_idx}_AttnProjection', MatmulOperator(batch_seq=2 * max(1, latent_len), in_features=2048, out_features=32), ModuleType.OTHER, layer_idx)
            for name in ('Mean', 'LogScale'):
                self._conv1d(f'AudioRef{ref_idx}_{name}', 2, 32, latent_len, 32, 1, 1, 0, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-video-decoder-io')
class MiniMaxH3VideoDecoderIOBlock(_MiniMaxH3Block):
    """ViT3DDecoder input/output projections around its 36 transformer layers."""

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        batch_seq = metadata.batch_size * metadata.seq_len
        self._add('VideoDecoder_XEmbedder', MatmulOperator(batch_seq=metadata.batch_size * (metadata.seq_len - 5), in_features=24, out_features=2048), ModuleType.VAEDECODER, layer_idx)
        self._add('VideoDecoder_ProjOut', MatmulOperator(batch_seq=batch_seq, in_features=2048, out_features=3 * 4 * 16 * 16), ModuleType.VAEDECODER, layer_idx)
        return self.nodes

@BlockFactory.register('minimax-h3-audio-vae-decoder')
class MiniMaxH3AudioVAEDecoderBlock(_MiniMaxH3Block):
    """BigVGAN decode main-op proxy using code-visible rates and residual kernels."""
    RATES = (5, 5, 2, 2, 2, 2, 2)
    KERNELS = (9, 9, 4, 4, 4, 4, 4)

    def build(self, metadata: ModelMetaData, layer_idx: int=0) -> list[Node]:
        self._pre_build(metadata.deploy_strategies)
        mm = metadata.multimodal_config
        length = mm['target_audio_latents_per_channel']
        batch = metadata.batch_size * 2
        self._conv1d('AudioDecoder_InputProjection', batch, 32, length, 2048, 1, 1, 0, layer_idx)
        channels = 1024
        self._conv1d('AudioDecoder_ConvPre', batch, 2048, length, channels, 7, 1, 3, layer_idx)
        for level, (rate, kernel) in enumerate(zip(self.RATES, self.KERNELS)):
            out_channels = channels // 2
            length *= rate
            self._conv1d(f'AudioDecoder_L{level}_ConvTranspose1dProxy', batch, channels, length, out_channels, kernel, 1, kernel // 2, layer_idx)
            channels = out_channels
            for res_kernel in (3, 7, 11):
                for dilation in (1, 3, 5):
                    self._conv1d(f'AudioDecoder_L{level}_K{res_kernel}_D{dilation}_Conv1', batch, channels, length, channels, res_kernel, 1, (res_kernel - 1) // 2, layer_idx)
                    self._conv1d(f'AudioDecoder_L{level}_K{res_kernel}_D{dilation}_Conv2', batch, channels, length, channels, res_kernel, 1, (res_kernel - 1) // 2, layer_idx)
        self._conv1d('AudioDecoder_ConvPost', batch, channels, length, 1, 7, 1, 3, layer_idx)
        return self.nodes
