"""Pinned MAGI-2 Preview workload geometry, independent of model execution.

Token counts and reference geometry are explicit analytical inputs; no input
media or tokenizer files are required at runtime.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, TypedDict
import json


CONFIG_DIR = Path(__file__).resolve().parent
# Ported from Markov_Infer_Sim; architecture/workload YAML converted to JSON.
# These source paths document origin only and are never read at runtime.
SOURCE_FILES = {'src/configs/magi2_config.py': '1f73965170e171514142d76e51b130de5c730adcd31af884671754afe50953e7', 'configs/models/magi2_preview.yaml': '5b56ed682f87e7c28b90ae1a4a42aec1ab0c3c8a8576cbdcab23514707004ece', 'configs/workload/workload_magi2_preview.yaml': 'dd1d589fb91dc49ffb5eb0c7b0fb8548cf0e48106f21b2722111dd2cd4896e6a'}
PRESETS = {"272p": (448, 256, None, None), "540p": (896, 512, None, None),
           "1080p": (896, 512, 1920, 1088)}


class TurboWindows(TypedDict):
    padding: int
    calls: int
    input_lengths: list[int]
    processed_latent_frames: int
    output_frames: int


class Magi2ShapeSummary(TypedDict):
    """Resolved request geometry shared by blocks, accounting and result exports."""
    workflow: str
    resolution: str
    preview_video_tokens: int
    audio_tokens: int
    image_tokens: int
    image_prefix_tokens: int
    positive_text_tokens: int
    negative_text_tokens: int
    refiner_video_tokens: int
    refiner_audio_tokens: int
    preview_latent_shape: list[int]
    refiner_latent_shape: list[int] | None
    decode_latent_shape: list[int]
    audio_latent_shape: list[int]
    image_latent_shape: list[int] | None
    image_layout: dict[str, list[int]] | None
    preview_sequence_lengths: list[int]
    preview_raw_tokens: int
    preview_padded_tokens: int
    preview_padding_tokens: int
    preview_per_rank_tokens: int
    preview_attention_pairs: int
    preview_modality_tokens: dict[str, int]
    refiner_cond_tokens: int
    refiner_uncond_tokens: int
    refiner_cond_attention_pairs: int
    refiner_uncond_attention_pairs: int
    refiner_cond_per_rank_tokens: list[int]
    refiner_uncond_per_rank_tokens: list[int]
    preview_forward_calls: int
    preview_cfg_batch: int
    refiner_forward_calls: int
    refiner_steps: int
    preview_steps: int
    decoded_video_shape: list[int]
    output_fps: float
    raw_audio_samples: int
    resampled_audio_samples: int
    turbo_windows: TurboWindows
    cp_size: int
    ep_size: int
    real_moe_heads: int
    padded_moe_heads: int
    parallelism_status: str


def load_model_spec(model: Any = None) -> dict[str, Any]:
    """Return a detached local architecture or an explicitly supplied override.

    Raw dictionaries contain the same components and indexed_weight_bytes as
    architecture.json. Legacy model/config objects remain accepted for callers
    carrying their own architecture; no external repository is imported.
    """
    config = getattr(model, "config", model)
    if config is None:
        return json.loads((CONFIG_DIR / "architecture.json").read_text(encoding="utf-8"))
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    return {
        "components": [deepcopy(getattr(component, "config", component))
                       for component in config.components],
        "indexed_weight_bytes": deepcopy(config.indexed_weight_bytes),
    }


def load_workload() -> dict[str, Any]:
    """Return the shared local workload with its i2v/t2v input profiles."""
    return json.loads((CONFIG_DIR / "workload.json").read_text(encoding="utf-8"))


def model_architecture(model: Any = None) -> dict[str, dict[str, Any]]:
    """Read component parameters from the local model or explicit override."""
    components = load_model_spec(model)["components"]
    architecture = {component["name"]: deepcopy({key: value for key, value in component.items()
                    if key not in {"layer_blocks", "name"}})
                    for component in components}
    for name in ("preview_transformer", "refiner_transformer"):
        config = architecture[name]
        layers = _positive_int(name + ".num_layers", config.get("num_layers"))
        fields = ("modality_bank_layers", "dense_ffn_layers") if name == "preview_transformer" else ("modality_bank_layers",)
        for field_name in fields:
            indices = config.get(field_name)
            if (not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < layers for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError(f"{name}.{field_name} must contain distinct layer indices in [0, {layers})")
    return architecture


def window_attention_pairs(shape: tuple[int, int, int], text_tokens: int,
                           block_shape: tuple[int, int, int] = (8, 4, 4),
                           radius: int = 2) -> int:
    """Exact block-grid pairs including partial edge blocks and dense text tail."""
    totals = []
    for dim, block in zip(shape, block_shape):
        sizes = [min(block, dim - start) for start in range(0, dim, block)]
        totals.append(sum(sizes[i] * sizes[j] for i in range(len(sizes))
                          for j in range(max(0, i - radius), min(len(sizes), i + radius + 1))))
    return math.prod(totals) + 2 * math.prod(shape) * text_tokens + text_tokens ** 2


def turbo_windows(frames: int, first: int = 7, step: int = 7,
                  temporal_stride: int = 4) -> TurboWindows:
    """Source window policy: one neighboring latent on either window boundary."""
    for name, value in (("frames", frames), ("first", first), ("step", step),
                        ("temporal_stride", temporal_stride)):
        _positive_int(name, value)
    padding = first - frames if frames < first else (step - (frames - first) % step) % step
    padded = frames + padding
    lengths = [first] if padded == first else [first + 1]
    lengths.extend(step + 1 if start + step == padded else step + 2
                   for start in range(first, padded, step))
    return {"padding": padding, "calls": len(lengths), "input_lengths": lengths,
            "processed_latent_frames": sum(lengths),
            "output_frames": temporal_stride * frames - (temporal_stride - 1)}


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass
class Magi2Config:
    workflow: str = "i2v"
    resolution: str = "1080p"
    seconds: float = 10.0
    preview_steps: int = 100
    refiner_steps: int = 5
    cp_size: int = 8
    ep_size: int = 8
    assume_stage_unload: bool = True
    image_present: bool = True
    image_width: int = 2710
    image_height: int = 1510
    image_orientation: int = 1
    ref_image_type: str = "original"
    preview_width: int | None = None
    preview_height: int | None = None
    refiner_width: int | None = None
    refiner_height: int | None = None
    positive_text_tokens: int | None = None
    negative_text_tokens: int | None = None
    seed: int = 42
    deterministic: bool = False
    cpu_text_timing_included: bool = False
    moe_active_experts_per_head: int = 256
    moe_compute_amplification: float = 1.0
    attention_softmax_compute_multiplier: float = 1.0
    precision_proxy: dict[str, float] = field(default_factory=lambda: {
        "bf16_from_fp16": 1.0, "fp32_from_fp16": 0.5,
    })
    dit_tokens: int = 0
    _architecture: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Magi2Config":
        # Match the shared-workload selector used by MiniMax-H3. Resolve only
        # the selected input profile so image geometry/token overrides cannot leak.
        if "workflows" in data:
            profiles = data["workflows"]
            workflow = data.get("workflow", "i2v")
            if not isinstance(profiles, dict) or set(profiles) - {"i2v", "t2v"}:
                raise ValueError("MAGI-2 workflows must map i2v/t2v to input profiles")
            if workflow not in ("i2v", "t2v") or not isinstance(profiles.get(workflow), dict):
                raise ValueError("MAGI-2 workflows must define the selected workflow")
            selected = profiles[workflow]
            if {"type", "workflow", "workflows"} & selected.keys():
                raise ValueError("MAGI-2 input profiles cannot redefine the workflow selector")
            data = {**{key: value for key, value in data.items() if key != "workflows"},
                    **selected}
        allowed = {key for key, value in cls.__dataclass_fields__.items() if value.init} | {"type", "prompt_enhancement"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown MAGI-2 configuration fields: {sorted(unknown)}")
        if data.get("prompt_enhancement", False):
            raise ValueError("external prompt enhancement is outside this workload")
        values = {key: value for key, value in data.items()
                  if key in cls.__dataclass_fields__ and cls.__dataclass_fields__[key].init and key != "dit_tokens"}
        if data.get("workflow", "i2v") == "t2v" and "image_present" not in data:
            values["image_present"] = False
        config = cls(**values)
        config._resolve_token_counts()
        config.compute_mm_tokens(None)
        return config

    def _resolve_token_counts(self) -> None:
        # Standalone defaults retain the official request; shipped workload JSON
        # declares these values per profile. No text/tokenizer inference occurs.
        if self.positive_text_tokens is None:
            if self.workflow not in {"i2v", "t2v"}:
                raise ValueError(f"unsupported MAGI-2 workflow: {self.workflow}")
            self.positive_text_tokens = {"i2v": 2029, "t2v": 2014}[self.workflow]
        if self.negative_text_tokens is None:
            self.negative_text_tokens = 429

    @property
    def condition_type(self) -> str:
        return self.workflow

    @property
    def num_diffusion_steps(self) -> int:
        return self.preview_steps

    @property
    def vae_downsampling_factor(self) -> tuple[int, int, int]:
        """Report the output decoder stride; Preview temporal stride is 8."""
        decoder = self.architecture["video_vae_decoder"]
        return (decoder["temporal_stride"], decoder["spatial_stride"], decoder["spatial_stride"])

    @property
    def patch_downsampling_factor(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    @property
    def cfg(self) -> int:
        """Preview concatenates two CFG branches; Refiner counts two forwards."""
        return 2

    @property
    def prompt_len(self) -> int:
        self._resolve_token_counts()
        return int(self.positive_text_tokens)

    @property
    def thw_list(self) -> list[tuple[int, int, int]]:
        return [tuple(self.shape_summary()["decoded_video_shape"][:3])]

    @property
    def architecture(self) -> dict[str, dict[str, Any]]:
        return self._architecture or model_architecture()

    def _validate(self) -> None:
        if self.workflow not in ("t2v", "i2v"):
            raise ValueError("MAGI-2 workflow must be t2v or i2v")
        if self.resolution not in PRESETS:
            raise ValueError("MAGI-2 resolution must be 272p, 540p or 1080p")
        if self.seconds != 10:
            raise ValueError("the pinned official model supports only 10 seconds")
        for name in ("preview_steps", "refiner_steps", "positive_text_tokens", "negative_text_tokens"):
            _positive_int(name, getattr(self, name))
        if max(self.preview_steps, self.refiner_steps) > 1000:
            raise ValueError("diffusion steps cannot exceed the 1000 scheduler timesteps")
        if max(self.positive_text_tokens, self.negative_text_tokens) > self.architecture["qwen_text"]["max_tokens"]:
            raise ValueError("text token count exceeds the official 7000-token limit")
        if (type(self.cp_size) is not int or type(self.ep_size) is not int or
                self.cp_size != self.ep_size or self.cp_size not in (1, 2, 4, 8)):
            raise ValueError("supported structural scenarios require CP=EP in {1,2,4,8}")
        if self.cpu_text_timing_included:
            raise ValueError("CPU text latency is uncalibrated; only work and memory are recorded")
        _positive_int("moe_active_experts_per_head", self.moe_active_experts_per_head)
        preview = self.architecture["preview_transformer"]
        if not preview["experts_per_token"] <= self.moe_active_experts_per_head <= preview["experts_per_head"]:
            raise ValueError("moe_active_experts_per_head must be between Top-6 and 256")
        value = self.moe_compute_amplification
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
            raise ValueError("moe_compute_amplification must be finite and >= 1")
        value = self.attention_softmax_compute_multiplier
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("attention_softmax_compute_multiplier must be finite and > 0")
        if self.ref_image_type not in ("original", "square"):
            raise ValueError("ref_image_type must be original or square")
        if type(self.assume_stage_unload) is not bool or type(self.deterministic) is not bool:
            raise ValueError("assume_stage_unload and deterministic must be booleans")
        expected = {"bf16_from_fp16", "fp32_from_fp16"}
        if set(self.precision_proxy) != expected or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
            for v in self.precision_proxy.values()
        ):
            raise ValueError("precision_proxy requires positive finite bf16_from_fp16/fp32_from_fp16")
        if type(self.image_present) is not bool:
            raise ValueError("image_present must be boolean")
        if self.workflow == "i2v":
            if not self.image_present:
                raise ValueError("I2V requires one declared reference image")
            _positive_int("image_width", self.image_width)
            _positive_int("image_height", self.image_height)
            if type(self.image_orientation) is not int or self.image_orientation not in range(1, 9):
                raise ValueError("image_orientation must be an EXIF value in 1..8")
        elif self.image_present:
            raise ValueError("T2V must not include a reference image")

    def shape_summary(self) -> Magi2ShapeSummary:
        self._resolve_token_counts()
        self._validate()
        architecture = self.architecture
        preview = architecture["preview_transformer"]
        refiner = architecture["refiner_transformer"]
        decoder = architecture["video_vae_decoder"]
        audio_decoder = architecture["audio_vae_decoder"]
        stride = decoder["spatial_stride"]
        latent_channels = decoder["latent_channels"]
        pw, ph, rw, rh = PRESETS[self.resolution]
        pw = pw if self.preview_width is None else self.preview_width
        ph = ph if self.preview_height is None else self.preview_height
        if self.resolution == "1080p":
            rw = rw if self.refiner_width is None else self.refiner_width
            rh = rh if self.refiner_height is None else self.refiner_height
        elif self.refiner_width is not None or self.refiner_height is not None:
            raise ValueError("refiner dimensions require the 1080p two-stage profile")
        for name, value in (("preview_width", pw), ("preview_height", ph),
                            ("refiner_width", rw), ("refiner_height", rh)):
            if value is not None:
                _positive_int(name, value)
                if value % stride:
                    raise ValueError(f"{name} must be a multiple of the VAE stride {stride}")
        tv, audio = preview["preview_latent_frames"], preview["audio_latent_frames"]
        video = tv * (ph // stride) * (pw // stride)
        refiner_shape = (refiner["latent_frames"], rh // stride, rw // stride) if rw else None
        rv = math.prod(refiner_shape) if refiner_shape else 0
        image_tokens, image_shape, image_layout = 0, None, None
        if self.workflow == "i2v":
            iw, ih = self.image_width, self.image_height
            if self.image_orientation in (5, 6, 7, 8):
                iw, ih = ih, iw
            if self.ref_image_type == "square":
                cw = ch = 320
            elif iw > ih:
                cw, ch = max(pw, ph), int(ih * max(pw, ph) / iw)
            else:
                ch, cw = max(pw, ph), int(iw * max(pw, ph) / ih)
            aw, ah = cw // 32 * 32, ch // 32 * 32
            if min(aw, ah) < 32:
                raise ValueError("reference image side is below the processor alignment 32")
            scale = min(cw / iw, ch / ih)
            sw, sh = max(1, round(iw * scale)), max(1, round(ih * scale))
            image_layout = {"oriented_wh": [iw, ih], "canvas_wh": [cw, ch],
                            "letterbox_content_wh": [sw, sh], "paste_xy": [(cw-sw)//2, (ch-sh)//2],
                            "processor_wh": [aw, ah]}
            image_shape = [1, latent_channels, 1, ah // stride, aw // stride]
            image_tokens = (ah // stride) * (aw // stride)
        prefix = int(self.workflow == "i2v")
        pos, neg = self.positive_text_tokens, self.negative_text_tokens
        lengths = [video + audio + image_tokens + prefix + n for n in (pos, neg)]
        raw = sum(lengths)
        alignment = preview["sequence_alignment"]
        padded = math.ceil(raw / alignment) * alignment if self.cp_size > 1 else raw
        if padded % self.cp_size:
            raise ValueError("preview sequence alignment must yield equal CP shards")
        pad = padded - raw
        final_shape = refiner_shape or (tv, ph // stride, pw // stride)
        windows = turbo_windows(final_shape[0], decoder["window_first"], decoder["window_step"], decoder["temporal_stride"])
        summary: Magi2ShapeSummary = {
            "workflow": self.workflow, "resolution": self.resolution,
            "preview_video_tokens": video, "audio_tokens": audio, "image_tokens": image_tokens,
            "image_prefix_tokens": prefix, "positive_text_tokens": pos, "negative_text_tokens": neg,
            "refiner_video_tokens": rv, "refiner_audio_tokens": 0,
            "preview_latent_shape": [1, latent_channels, tv, ph // stride, pw // stride],
            "refiner_latent_shape": [1, latent_channels, *refiner_shape] if refiner_shape else None,
            "decode_latent_shape": [1, latent_channels, *final_shape],
            "audio_latent_shape": [1, audio, audio_decoder["latent_channels"]], "image_latent_shape": image_shape,
            "image_layout": image_layout, "preview_sequence_lengths": lengths,
            "preview_raw_tokens": raw, "preview_padded_tokens": padded,
            "preview_padding_tokens": pad, "preview_per_rank_tokens": padded // self.cp_size,
            "preview_attention_pairs": sum(n*n for n in lengths) + pad*pad,
            "preview_modality_tokens": {"video": 2*(video+image_tokens), "audio": 2*audio,
                                        "text": pos+neg+2*prefix+pad},
            "refiner_cond_tokens": rv+pos if rv else 0,
            "refiner_uncond_tokens": rv+neg if rv else 0,
            "refiner_cond_attention_pairs": window_attention_pairs(refiner_shape, pos, tuple(refiner["window_block_shape"]), refiner["window_radius"]) if rv else 0,
            "refiner_uncond_attention_pairs": window_attention_pairs(refiner_shape, neg, tuple(refiner["window_block_shape"]), refiner["window_radius"]) if rv else 0,
            "preview_forward_calls": self.preview_steps, "preview_cfg_batch": 2,
            "refiner_forward_calls": 2*self.refiner_steps if rv else 0,
            "refiner_steps": self.refiner_steps if rv else 0, "preview_steps": self.preview_steps,
            "decoded_video_shape": [windows["output_frames"], final_shape[1]*stride, final_shape[2]*stride, decoder["output_channels"] // decoder["pixel_patch_size"]**2],
            "output_fps": 25.0 if rv else 12.5,
            "raw_audio_samples": audio*math.prod(audio_decoder["strides"]), "resampled_audio_samples": audio*audio_decoder["output_samples_per_latent"],
            "turbo_windows": windows, "cp_size": self.cp_size, "ep_size": self.ep_size,
            "real_moe_heads": preview["num_moe_heads"], "padded_moe_heads": math.ceil(preview["num_moe_heads"]/self.ep_size)*self.ep_size,
            "parallelism_status": "official_8_rank_config" if self.cp_size == 8 else "structural_scenario",
        }
        for branch in ("cond", "uncond"):
            size = summary[f"refiner_{branch}_tokens"]
            quotient, remainder = divmod(size, self.cp_size)
            summary[f"refiner_{branch}_per_rank_tokens"] = [quotient + (r < remainder) for r in range(self.cp_size)]
        return summary

    def compute_mm_tokens(self, model: Any = None) -> None:
        if model is not None or not self._architecture:
            self._architecture = model_architecture(model)
        self.dit_tokens = self.shape_summary()["preview_padded_tokens"]

    def runtime_multimodal_config(self) -> dict[str, Any]:
        summary = self.shape_summary()
        values = asdict(self)
        values.pop("_architecture")
        return {**values, **summary, "shape_summary": summary, "architecture": self.architecture,
                "type": "magi2_preview",
                "precision_proxy_status": "uncalibrated_peak_throughput_proxy",
                "cpu_text_timing_included": False}
