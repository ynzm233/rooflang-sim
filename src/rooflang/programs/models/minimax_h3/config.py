"""MiniMax-H3 input geometry, adapted from source version in SOURCES.json."""
"""Source-derived workload shapes for MiniMax-H3 FL2VA and Ref2VA."""

import math
from dataclasses import dataclass, field
from typing import Any




@dataclass(frozen=True)
class Ref2VAReference:
    kind: str
    frames: int = 0
    height: int = 0
    width: int = 0
    audio_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ref2VAReference":
        kind = str(data["kind"])
        if kind not in ("image", "video", "audio"):
            raise ValueError(f"unsupported Ref2VA reference kind: {kind}")
        return cls(
            kind=kind,
            frames=int(data.get("frames", 1 if kind == "image" else 0)),
            height=int(data.get("height", 0)),
            width=int(data.get("width", 0)),
            audio_seconds=float(data.get("audio_seconds", 0.0)),
        )

    @property
    def has_visual(self) -> bool:
        return self.kind in ("image", "video")

    @property
    def has_audio(self) -> bool:
        return self.kind == "audio" or self.audio_seconds > 0


@dataclass(frozen=True)
class MiniMaxH3ShapeSummary:
    workflow: str
    target_latent_frames: int
    target_latent_height: int
    target_latent_width: int
    target_audio_latents_per_channel: int
    text_rows: int
    condition_video_rows: int
    condition_audio_rows: int
    target_video_rows: int
    target_audio_rows: int
    packed_rows: int
    qwen_raw_visual_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class MiniMaxH3MultimodalConfig:
    workflow: str
    target_frames: int
    target_height: int
    target_width: int
    text_condition_rows: int
    keyframes: tuple[str, ...] = ()
    references: list[Ref2VAReference] = field(default_factory=list)
    num_inference_steps: int = 50
    assume_stage_unload: bool = False
    dit_tokens: int = 0

    cfg: int = 1
    vae_downsampling_factor: tuple[int, int, int] = (4, 16, 16)
    patch_downsampling_factor: tuple[int, int, int] = (1, 2, 2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniMaxH3MultimodalConfig":
        workflow = str(data["workflow"])
        profiles = data.get("workflows")
        if not isinstance(profiles, dict) or set(profiles) != {"fl2va", "ref2va"}:
            raise ValueError("MiniMax-H3 workflows must contain fl2va and ref2va")
        if workflow not in profiles:
            raise ValueError(f"unsupported MiniMax-H3 workflow: {workflow}")
        allowed_fields = {
            "fl2va": {"text_condition_rows", "keyframes"},
            "ref2va": {"text_condition_rows", "references"},
        }
        for name, profile_data in profiles.items():
            if set(profile_data) != allowed_fields[name]:
                raise ValueError(
                    f"{name} profile fields must be {sorted(allowed_fields[name])}"
                )

        target = data["target"]
        profile = profiles[workflow]
        keyframes = tuple(profile.get("keyframes", ()))
        references = [
            Ref2VAReference.from_dict(item) for item in profile.get("references", ())
        ]
        config = cls(
            workflow=workflow,
            target_frames=int(target["frames"]),
            target_height=int(target["height"]),
            target_width=int(target["width"]),
            text_condition_rows=int(profile["text_condition_rows"]),
            keyframes=keyframes,
            references=references,
            num_inference_steps=int(data["num_inference_steps"]),
            assume_stage_unload=bool(data.get("assume_stage_unload", False)),
        )
        config.shape_summary()
        return config

    @property
    def condition_type(self) -> str:
        return self.workflow

    @property
    def num_diffusion_steps(self) -> int:
        return self.num_inference_steps

    @property
    def prompt_len(self) -> int:
        return self.text_condition_rows

    @property
    def thw_list(self) -> list[tuple[int, int, int]]:
        return [(self.target_frames, self.target_height, self.target_width)]

    @staticmethod
    def _video_latent_frames(frames: int) -> int:
        if frames == 1:
            return 1
        if frames < 5 or (frames - 5) % 17:
            raise ValueError(f"visual frames must be 1 or 17*n+5, got {frames}")
        return ((frames - 5) // 17) * 5 + 2

    @staticmethod
    def _validate_spatial(height: int, width: int) -> None:
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(
                f"height/width must be positive multiples of 32, got {height}x{width}"
            )

    @staticmethod
    def _qwen_raw_visual_tokens(ref: Ref2VAReference) -> int:
        spatial = (ref.height // 16) * (ref.width // 16)
        if ref.kind == "image":
            return spatial
        sampled_frames = math.ceil(ref.frames / 12)  # 24 fps input -> 2 fps
        return math.ceil(sampled_frames / 2) * spatial  # temporal patch size 2

    @staticmethod
    def _qwen_grid(ref: Ref2VAReference) -> dict[str, Any]:
        temporal = (
            1
            if ref.kind == "image"
            else math.ceil(math.ceil(ref.frames / 12) / 2)
        )
        height = ref.height // 16
        width = ref.width // 16
        return {
            "kind": ref.kind,
            "temporal": temporal,
            "height": height,
            "width": width,
            "raw_rows": temporal * height * width,
        }

    def _fl2va_conditions(self) -> tuple[int, int]:
        if not 1 <= len(self.keyframes) <= 2:
            raise ValueError("FL2VA requires one or two keyframes")
        if len(set(self.keyframes)) != len(self.keyframes) or not set(
            self.keyframes
        ) <= {"first", "last"}:
            raise ValueError("FL2VA keyframes must be unique first/last anchors")
        rows_per_keyframe = (self.target_height // 32) * (self.target_width // 32)
        raw_per_keyframe = (self.target_height // 16) * (self.target_width // 16)
        return len(self.keyframes) * rows_per_keyframe, len(self.keyframes) * raw_per_keyframe

    def _ref2va_conditions(self) -> tuple[int, int, int]:
        condition_video_rows = 0
        condition_audio_rows = 0
        qwen_raw_visual_tokens = 0
        image_count = video_count = audio_count = 0
        for ref in self.references:
            if ref.has_visual:
                if ref.kind == "image" and ref.frames != 1:
                    raise ValueError("Ref2VA image references must have frames=1")
                if ref.kind == "video" and ref.frames < 13:
                    raise ValueError("Ref2VA video references must have at least 13 frames")
                self._validate_spatial(ref.height, ref.width)
                ref_f = self._video_latent_frames(ref.frames)
                condition_video_rows += ref_f * (ref.height // 32) * (ref.width // 32)
                qwen_raw_visual_tokens += self._qwen_raw_visual_tokens(ref)
                image_count += int(ref.kind == "image")
                video_count += int(ref.kind == "video")
            if ref.has_audio:
                if ref.audio_seconds <= 0:
                    raise ValueError("audio reference duration must be positive")
                condition_audio_rows += 2 * round(ref.audio_seconds * 40)
                audio_count += int(ref.kind == "audio")

        if image_count + video_count == 0:
            raise ValueError("Ref2VA requires at least one image or video reference")
        if len(self.references) > 12 or image_count > 9 or video_count > 3 or audio_count > 3:
            raise ValueError("reference count exceeds MiniMax-H3 Ref2VA limits")
        return condition_video_rows, condition_audio_rows, qwen_raw_visual_tokens

    def shape_summary(self) -> MiniMaxH3ShapeSummary:
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 5 * 24 <= self.target_frames <= 15 * 24:
            raise ValueError("target duration must be within 5-15 seconds at 24 fps")
        if self.text_condition_rows <= 0:
            raise ValueError("text_condition_rows must be positive")
        self._validate_spatial(self.target_height, self.target_width)

        target_f = self._video_latent_frames(self.target_frames)
        target_h = self.target_height // 16
        target_w = self.target_width // 16
        target_video_rows = target_f * (target_h // 2) * (target_w // 2)
        target_audio_latents = round(self.target_frames / 24 * 40)
        target_audio_rows = 2 * target_audio_latents

        if self.workflow == "fl2va":
            condition_video_rows, qwen_raw_visual_tokens = self._fl2va_conditions()
            condition_audio_rows = 0
        elif self.workflow == "ref2va":
            condition_video_rows, condition_audio_rows, qwen_raw_visual_tokens = (
                self._ref2va_conditions()
            )
        else:
            raise ValueError(f"unsupported MiniMax-H3 workflow: {self.workflow}")

        packed_rows = (
            self.text_condition_rows
            + condition_video_rows
            + condition_audio_rows
            + target_audio_rows
            + target_video_rows
        )
        return MiniMaxH3ShapeSummary(
            workflow=self.workflow,
            target_latent_frames=target_f,
            target_latent_height=target_h,
            target_latent_width=target_w,
            target_audio_latents_per_channel=target_audio_latents,
            text_rows=self.text_condition_rows,
            condition_video_rows=condition_video_rows,
            condition_audio_rows=condition_audio_rows,
            target_video_rows=target_video_rows,
            target_audio_rows=target_audio_rows,
            packed_rows=packed_rows,
            qwen_raw_visual_tokens=qwen_raw_visual_tokens,
        )

    def compute_mm_tokens(self, model: Any) -> None:
        self.dit_tokens = self.shape_summary().packed_rows

    def runtime_multimodal_config(self) -> dict[str, Any]:
        shapes = self.shape_summary()
        visual_conditions = (
            [
                {"frames": 1, "height": self.target_height, "width": self.target_width}
                for _ in self.keyframes
            ]
            if self.workflow == "fl2va"
            else [
                {"frames": ref.frames, "height": ref.height, "width": ref.width}
                for ref in self.references
                if ref.has_visual
            ]
        )
        qwen_visual_grids = (
            [
                {
                    "kind": "image",
                    "temporal": 1,
                    "height": self.target_height // 16,
                    "width": self.target_width // 16,
                    "raw_rows": (self.target_height // 16)
                    * (self.target_width // 16),
                }
                for _ in self.keyframes
            ]
            if self.workflow == "fl2va"
            else [self._qwen_grid(ref) for ref in self.references if ref.has_visual]
        )
        audio_reference_latents = (
            [
                round(ref.audio_seconds * 40)
                for ref in self.references
                if ref.has_audio
            ]
            if self.workflow == "ref2va"
            else []
        )
        return {
            **shapes.to_dict(),
            "shape_summary": shapes.to_dict(),
            "workflow": self.workflow,
            "keyframes": list(self.keyframes),
            "references": [vars(ref).copy() for ref in self.references],
            "visual_conditions": visual_conditions,
            "qwen_visual_grids": qwen_visual_grids,
            "audio_reference_latents": audio_reference_latents,
            "target_frames": self.target_frames,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "num_inference_steps": self.num_inference_steps,
            "assume_stage_unload": self.assume_stage_unload,
            "video_scheduler_shift": 12.0,
            "audio_scheduler_shift": 3.0,
            "max_adaln_timestep_rows": 3 if self.workflow == "fl2va" else 4,
            "cfg": 1,
        }
