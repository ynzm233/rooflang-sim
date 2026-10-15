"""MAGI codec graphs composed from standard shape-driven OpLens operators.

Architecture fields come from the component YAML. The application owns window
repetition and whole-module residency; these blocks describe one executed call.
"""

import math

from .blocks import _Magi2Block, _quant
from .operators import (
    Conv1dOperator, Conv3dOperator, ConvTranspose1dOperator,
    FlashAttentionOperator, WeightNormOperator, InterpolateOperator,
    SnakeBetaOperator,
)

class _Magi2CodecBlock(_Magi2Block):
    def _codec_start(self, metadata, layer_idx):
        self.sh, self.mm, _ = self._start(metadata, layer_idx)
        self.batch = metadata.batch_size
        self.dtype = self.cfg["weight_dtype"]

    def _conv3d(self, name, cin, cout, thw, kernel=(3, 3, 3), *,
                stride=(1, 1, 1), padding_style="turbo", bias=True,
                explicit_padding=None):
        """Use native Conv3d geometry, exposing source materialized padding.

        Wan CausalConv3d materializes all spatial/temporal zero padding. Turbo
        replicates temporal end frames, with spatial padding inside Conv3d.
        A spatial Conv2d is represented with time kernel/stride one.
        """
        thw, kernel, stride = tuple(thw), tuple(kernel), tuple(stride)
        if explicit_padding is not None:
            extra = tuple(explicit_padding)
            native_pad = (0, 0, 0)
        elif padding_style == "wan":
            extra = tuple(k - 1 for k in kernel)
            native_pad = (0, 0, 0)
        elif padding_style == "turbo":
            extra = (kernel[0] - 1, 0, 0)
            native_pad = (0, kernel[1] // 2, kernel[2] // 2)
        else:
            raise ValueError(f"unknown codec padding style: {padding_style}")
        padded = tuple(size + pad for size, pad in zip(thw, extra))
        if any(extra):
            input_elements = self.batch * cin * math.prod(thw)
            output_elements = self.batch * cin * math.prod(padded)
            mode = "pad"
            if padding_style == "turbo" and explicit_padding is None:
                # The source materializes each endpoint repeat, then cat reads
                # both repeated endpoints and the original tensor once.
                edge = self.batch * cin * thw[1] * thw[2]
                for side in ("Left", "Right"):
                    self._move(name + "_PadRepeat" + side, edge, edge * extra[0] // 2,
                               dtype=self.dtype)
                input_elements = output_elements
                mode = "concat"
            self._move(name + "_Pad", input_elements, output_elements,
                       dtype=self.dtype, mode=mode)
        op = Conv3dOperator(
            batch=self.batch, c_in=cin, c_out=cout,
            d_in=padded[0], h_in=padded[1], w_in=padded[2],
            kernel_t=kernel[0], kernel_h=kernel[1], kernel_w=kernel[2],
            stride_t=stride[0], stride_h=stride[1], stride_w=stride[2],
            pad_t=native_pad[0], pad_h=native_pad[1], pad_w=native_pad[2],
            has_bias=bias, quant_config=_quant(self.dtype),
        )
        output = (op.d_out, op.h_out, op.w_out)
        if min(output) <= 0:
            raise ValueError(f"{name} produced invalid codec shape {output}")
        op.shape = f"{cin}x{padded} -> {cout}x{output}; kernel={kernel}"
        op.assumption = "native dense convolution; source materialized padding is a separate tensor move"
        self._add(name, op)
        return output


class Magi2TurboVAEDecoder(_Magi2CodecBlock):
    def _res(self, name, cin, cout, thw, modified=False):
        volume = math.prod(thw) * self.batch
        for index, ci, co in ((1, cin, cout), (2, cout, cout)):
            self._norm(f"{name}_Norm{index}", volume, ci, dtype=self.dtype, affine=False)
            kind = "relu" if index == 1 and modified else "silu"
            label = "ReLU" if kind == "relu" else "SiLU"
            self._activation(f"{name}_{label}{index}", volume, ci, dtype=self.dtype, kind=kind)
            self._conv3d(f"{name}.conv{index}", ci, co, thw)
        if cin != cout:
            self._norm(name + "_Norm3", volume, cin, dtype=self.dtype, affine=False)
            self._conv3d(name + ".shortcut", cin, cout, thw, (1, 1, 1))
        self._residual(name + "_Residual", volume * cout, self.dtype)

    def build(self, metadata, layer_idx=0):
        self._codec_start(metadata, layer_idx)
        latent = self.sh["decode_latent_shape"]
        h, w = latent[-2:]
        index = int(self.mm.get("turbo_window_index", 0))
        windows = self.sh["turbo_windows"]["input_lengths"]
        if not 0 <= index < len(windows):
            raise ValueError("Turbo window index is outside the configured clip")
        length = windows[index]
        first = index == 0
        thw = (length, h, w)
        channels = self.cfg["base_channels"]
        levels = self.cfg["channels"]
        repeats = self.cfg["res_blocks"]
        if len(levels) != len(repeats):
            raise ValueError("Turbo channels and res_blocks lengths must match")
        self._conv3d("conv_in", self.cfg["latent_channels"], channels, thw)
        for i in range(self.cfg["mid_res_blocks"]):
            self._res(f"mid.{i}", channels, channels, thw)
        for level, out_channels in enumerate(levels):
            if channels != out_channels:
                self._res(f"up.{level}.transition", channels, out_channels, thw)
            channels = out_channels
            if level in self.cfg["temporal_upsample_levels"]:
                self._conv3d(f"up.{level}.time", channels, channels * 2, thw, (3, 1, 1))
                out = (thw[0] * 2 - int(first), thw[1], thw[2])
                # rearrange materializes all 2T output frames before the first
                # chunk drops frame zero with a storage-sharing slice.
                self._move(f"up.{level}.TimeInterleave", self.batch * channels * 2 * math.prod(thw),
                           dtype=self.dtype, mode="deinterleave")
                thw = out
            if level in self.cfg["spatial_upsample_levels"]:
                out = (thw[0], thw[1] * 2, thw[2] * 2)
                # WanUpsample explicitly converts BF16 to FP32, interpolates,
                # then casts back before the spatial convolution.
                self._move(f"up.{level}.Nearest_ToFP32", self.batch * channels * math.prod(thw),
                           dtype=self.dtype, output_dtype="fp32", mode="cast")
                self._add(f"up.{level}.Nearest", InterpolateOperator(
                    input_shape=[self.batch * thw[0], channels, thw[1], thw[2]],
                    output_shape=[self.batch * out[0], channels, out[1], out[2]],
                    mode="nearest-exact", quant_config=_quant("fp32")))
                self._move(f"up.{level}.Nearest_FromFP32", self.batch * channels * math.prod(out),
                           dtype="fp32", output_dtype=self.dtype, mode="cast")
                self._conv3d(f"up.{level}.spatial", channels, channels, out, (1, 3, 3))
                thw = out
            for i in range(repeats[level]):
                self._res(f"up.{level}.res.{i}", channels, channels, thw,
                          modified=level in self.cfg["modified_residual_levels"])
        volume = math.prod(thw) * self.batch
        self._norm("Output_Norm", volume, channels, dtype=self.dtype, affine=False)
        self._activation("Output_SiLU", volume, channels, dtype=self.dtype)
        self._conv3d("conv_out", channels, self.cfg["output_channels"], thw)
        self._move("Output_Unpatchify", volume * self.cfg["output_channels"],
                   dtype=self.dtype, mode="unpatchify")
        return self.nodes


class Magi2ImageVAEEncoder(_Magi2CodecBlock):
    def _outer_shortcut(self, name, cin, cout, inp, out, temporal):
        ft, fs = 2 if temporal else 1, 2 if inp[-1] != out[-1] else 1
        padded = (math.ceil(inp[0] / ft) * ft, inp[1], inp[2])
        inputs = self.batch * cin * math.prod(inp)
        padded_elements = self.batch * cin * math.prod(padded)
        outputs = self.batch * cout * math.prod(out)
        numerator = cin * ft * fs * fs
        if numerator % cout:
            raise ValueError("Wan AvgDown3D requires integral channel groups")
        group = numerator // cout
        self._move(name + "_AvgDownPad", inputs, padded_elements, dtype=self.dtype, mode="pad")
        if ft * fs > 1:
            self._move(name + "_AvgDownPermute", padded_elements, dtype=self.dtype, mode="permute")
        self._mean(name + "_AvgDownMean", outputs, group, self.dtype)
        self._residual(name + "_OuterResidual", outputs, self.dtype)

    def _res(self, name, cin, cout, thw):
        volume = math.prod(thw) * self.batch
        if cin != cout:
            self._conv3d(name + ".shortcut", cin, cout, thw, (1, 1, 1), padding_style="wan")
        for i, ci in ((1, cin), (2, cout)):
            self._norm(f"{name}_Norm{i}", volume, ci, dtype=self.dtype)
            self._activation(f"{name}_SiLU{i}", volume, ci, dtype=self.dtype)
            self._move(f"{name}.conv{i}_CacheClone", volume * ci, dtype=self.dtype)
            self._conv3d(f"{name}.conv{i}", ci, cout, thw, padding_style="wan")
        self._residual(name + "_Residual", volume * cout, self.dtype)

    def build(self, metadata, layer_idx=0):
        self._codec_start(metadata, layer_idx)
        latent = self.sh["image_latent_shape"]
        if latent[2] != 1:
            raise ValueError("MAGI image encoder currently models a single reference frame")
        h, w = latent[-2:]
        scale = self.cfg["spatial_stride"] // self.cfg["pixel_patch_size"]
        thw = (1, h * scale, w * scale)
        self._move("Image_PixelUnshuffle", self.batch * self.cfg["input_channels"] * math.prod(thw),
                   dtype=self.dtype, mode="permute")
        cin = self.cfg["base_channels"]
        self._move("encoder.conv1_CacheClone", self.batch * self.cfg["input_channels"] * math.prod(thw),
                   dtype=self.dtype)
        self._conv3d("encoder.conv1", self.cfg["input_channels"], cin, thw, padding_style="wan")
        levels = [self.cfg["base_channels"] * m for m in self.cfg["channel_multipliers"]]
        for level, cout in enumerate(levels):
            shortcut_cin, shortcut_shape = cin, thw
            self._move(f"encoder.down.{level}_OuterClone", self.batch * cin * math.prod(thw), dtype=self.dtype)
            for i in range(self.cfg["num_res_blocks"]):
                self._res(f"encoder.down.{level}.res.{i}", cin, cout, thw)
                cin = cout
            if level < len(levels) - 1:
                thw = self._conv3d(f"encoder.down.{level}.spatial", cin, cin, thw, (1, 3, 3),
                                   stride=(1, 2, 2), explicit_padding=(0, 1, 1))
                if level in self.cfg["temporal_downsample_levels"]:
                    self._move(f"encoder.down.{level}.time_CacheClone",
                               self.batch * cin * math.prod(thw), dtype=self.dtype)
            # Native first-frame cache skips time_conv even at temporal levels.
            self._outer_shortcut(f"encoder.down.{level}", shortcut_cin, cout,
                                 shortcut_shape, thw, level in self.cfg["temporal_downsample_levels"])
        self._res("encoder.middle.0", cin, cin, thw)
        volume = math.prod(thw) * self.batch
        self._norm("encoder.middle.attn.Norm", volume, cin, dtype=self.dtype)
        self._conv3d("encoder.middle.attn.qkv", cin, 3 * cin, thw, (1, 1, 1), padding_style="wan")
        self._move("encoder.middle.attn.QKV_Contiguous", volume * 3 * cin,
                   dtype=self.dtype, mode="permute")
        self._add("encoder.middle.attn.SDPA", FlashAttentionOperator(
            batch_size=self.batch, num_heads=1, kv_heads=1, q_len=math.prod(thw),
            kv_len=math.prod(thw), head_dim=cin, causal=False, quant_config=_quant(self.dtype)))
        self._conv3d("encoder.middle.attn.proj", cin, cin, thw, (1, 1, 1), padding_style="wan")
        self._residual("encoder.middle.attn.Residual", volume * cin, self.dtype)
        self._res("encoder.middle.2", cin, cin, thw)
        self._norm("encoder.head.Norm", volume, cin, dtype=self.dtype)
        self._activation("encoder.head.SiLU", volume, cin, dtype=self.dtype)
        moments = 2 * self.cfg["latent_channels"]
        self._move("encoder.head_CacheClone", volume * cin, dtype=self.dtype)
        self._conv3d("encoder.head", cin, moments, thw, padding_style="wan")
        self._conv3d("vae.quant_conv", moments, moments, thw, (1, 1, 1), padding_style="wan")
        self._affine("Image_LatentNormalize", volume * self.cfg["latent_channels"],
                     self.cfg["latent_channels"], self.dtype)
        return self.nodes


class Magi2AudioVAEDecoder(_Magi2CodecBlock):
    def _conv1d(self, name, cin, cout, length, kernel, *, stride=1,
                dilation=1, transpose=False, bias=True):
        # torch.nn.utils.weight_norm defaults to dim=0. ConvTranspose stores
        # [Cin,Cout,K], whereas ordinary Conv stores [Cout,Cin,K].
        weight_shape = [cin, cout, kernel] if transpose else [cout, cin, kernel]
        self._add(name + "_WeightNorm", WeightNormOperator(
            weight_shape=weight_shape, dim=0, quant_config=_quant(self.dtype)))
        cls = ConvTranspose1dOperator if transpose else Conv1dOperator
        padding = math.ceil(stride / 2) if transpose else dilation * (kernel - 1) // 2
        op = cls(batch=self.batch, c_in=cin, c_out=cout, L_in=length,
                 kernel_size=kernel, stride=stride, pad=padding, dilation=dilation,
                 has_bias=bias, quant_config=_quant(self.dtype))
        op.shape = f"[{self.batch},{cin},{length}] -> [{self.batch},{cout},{op.L_out}]; kernel={kernel}; dilation={dilation}"
        self._add(name, op)
        return op.L_out

    def _snake(self, name, channels, length):
        self._add(name, SnakeBetaOperator(batch=self.batch, channels=channels, length=length,
                                          logscale=True, quant_config=_quant(self.dtype)))

    def build(self, metadata, layer_idx=0):
        self._codec_start(metadata, layer_idx)
        length = int(self.sh["audio_tokens"])
        latent = self.cfg["latent_channels"]
        self._affine("Audio_LatentDenormalize", self.batch * length * latent, latent,
                     self.dtype, inverse=True)
        channels = self.cfg["base_channels"]
        self._conv1d("conv_in", latent, channels, length, self.cfg["input_kernel_size"])
        levels, strides = self.cfg["channels"], self.cfg["strides"]
        if len(levels) != len(strides):
            raise ValueError("audio channels and strides lengths must match")
        for level, (out_channels, stride) in enumerate(zip(levels, strides)):
            self._snake(f"up.{level}.Snake", channels, length)
            length = self._conv1d(f"up.{level}", channels, out_channels, length,
                                  2 * stride, stride=stride, transpose=True)
            channels = out_channels
            for residual, dilation in enumerate(self.cfg["residual_dilations"]):
                for i, kernel in ((1, self.cfg["residual_kernel_size"]), (2, 1)):
                    self._snake(f"up.{level}.res.{residual}.Snake{i}", channels, length)
                    self._conv1d(f"up.{level}.res.{residual}.conv{i}", channels, channels,
                                 length, kernel, dilation=dilation if i == 1 else 1)
                self._residual(f"up.{level}.res.{residual}.Residual",
                               self.batch * length * channels, self.dtype)
        self._snake("Output_Snake", channels, length)
        self._conv1d("conv_out", channels, self.cfg["output_channels"], length,
                     self.cfg["output_kernel_size"], bias=False)
        return self.nodes
