"""Analytical cost descriptions adapted from Markov_Infer_Sim.

Only operator arithmetic, shape validation, precision and sequential subcall
specifications are retained. RoofLang supplies graph scheduling and timing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum, auto
import functools
from typing import Any, Optional, Literal, get_type_hints

ComputeUnitType = Literal["cube", "vector", "mixed", "storage"]
DTYPE_INFO = {
    "fp4": {"size": 0.5},
    "fp8": {"size": 1},
    "fp16": {"size": 2},
    "fp32": {"size": 4},
    "bf4": {"size": 0.5},
    "bf8": {"size": 1},
    "bf16": {"size": 2},
    "bf32": {"size": 4},
    "int8": {"size": 1},
    "int32": {"size": 4},
    "mix2.5": {"size": 0.3125}
}


DEFAULT_DTYPE = "fp16"


def _get_dtype_info(dtype: str, key: str, default) -> int | float:
    return DTYPE_INFO.get(dtype, {}).get(key, default)


def get_dtype_size(dtype: str) -> int:
    """Get size in bytes for a dtype."""
    return float(_get_dtype_info(dtype, "size", 2))


@dataclass(frozen=True)
class QuantConfig:
    """Precision configuration for quantization."""

    compute_dtype: str = DEFAULT_DTYPE
    weight_storage_dtype: str = DEFAULT_DTYPE
    kv_storage_dtype: str = DEFAULT_DTYPE
    communicate_dtype: str = DEFAULT_DTYPE

    def get_compute_size(self) -> int:
        return get_dtype_size(self.compute_dtype)

    def get_weight_size(self) -> int:
        return get_dtype_size(self.weight_storage_dtype)

    def get_kv_size(self) -> int:
        return get_dtype_size(self.kv_storage_dtype)

    def get_communicate_size(self) -> int:
        return get_dtype_size(self.communicate_dtype)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "QuantConfig":
        """
        从字典创建 QuantConfig 实例，支持缺失字段自动使用默认值
        """
        return cls(
            compute_dtype=config_dict.get("compute_dtype", DEFAULT_DTYPE),
            weight_storage_dtype=config_dict.get("weight_storage_dtype", DEFAULT_DTYPE),
            kv_storage_dtype=config_dict.get("kv_storage_dtype", DEFAULT_DTYPE),
            communicate_dtype=config_dict.get("communicate_dtype", DEFAULT_DTYPE),
        )


@functools.lru_cache(maxsize=128)
def _get_cached_type_hints(cls: type) -> dict[str, Any]:
    """Cached get_type_hints."""
    return get_type_hints(cls)


@dataclass
class ParamSpec:
    """Parameter specification with validation rules."""

    name: str
    type_: type
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list] = None


class OperatorParams:
    """Base class for operator parameters with validation."""

    def validate(self) -> None:
        """Validate all parameters including defaults."""
        for spec in self._get_param_specs():
            value = getattr(self, spec.name)

            if value is not None and spec.type_ is not Any:
                try:
                    type_ok = isinstance(value, spec.type_)
                except TypeError:
                    type_ok = True
                if not type_ok:
                    try:
                        value = spec.type_(value)
                        setattr(self, spec.name, value)
                    except (TypeError, ValueError):
                        raise TypeError(
                            f"Parameter '{spec.name}' must be {spec.type_.__name__}, "
                            f"got {type(value).__name__}"
                        )

            if value is not None and spec.min is not None and value < spec.min:
                raise ValueError(
                    f"Parameter '{spec.name}' must be >= {spec.min}, got {value}"
                )
            if value is not None and spec.max is not None and value > spec.max:
                raise ValueError(
                    f"Parameter '{spec.name}' must be <= {spec.max}, got {value}"
                )

            if value is not None and spec.choices is not None and value not in spec.choices:
                raise ValueError(
                    f"Parameter '{spec.name}' must be one of {spec.choices}, got {value}"
                )

    @classmethod
    def _get_param_specs(cls) -> list[ParamSpec]:
        """Extract parameter specifications from dataclass fields."""
        specs = []
        type_hints = _get_cached_type_hints(cls)

        for f in fields(cls):
            metadata = f.metadata or {}
            specs.append(
                ParamSpec(
                    name=f.name,
                    type_=type_hints.get(f.name, Any),
                    min=metadata.get("min"),
                    max=metadata.get("max"),
                    choices=metadata.get("choices"),
                )
            )
        return specs


@dataclass(frozen=True, slots=True)
class ComputeComponent:
    """Compute component for performance modeling.

    Supports both single-unit and mixed compute scenarios.
    For single-unit operations (cube/vector), use 'flops' and 'unit_type'.
    For mixed operations, use 'cube_flops' and 'vector_flops' with unit_type='mixed'.
    """

    # For single-unit operations
    flops: float = 0.0
    unit_type: ComputeUnitType = "vector"

    # For mixed operations (when unit_type == "mixed")
    cube_flops: float = 0.0
    vector_flops: float = 0.0

    dtype: str = "fp16"
    # Mixed operators may use different arithmetic precision on each unit.
    # dtype remains the fallback for existing single/shared-dtype recipes.
    cube_dtype: str | None = None
    vector_dtype: str | None = None

    def get_unit_dtype(self, unit: str) -> str:
        return (self.cube_dtype if unit == "cube" else self.vector_dtype) or self.dtype

    def total_flops(self) -> float:
        """Get total FLOPs."""
        if self.unit_type == "mixed":
            return self.cube_flops + self.vector_flops
        return self.flops

    def get_unit_flops(self, unit: str) -> float:
        """Get FLOPs for specific unit type."""
        if self.unit_type == "mixed":
            return self.cube_flops if unit == "cube" else self.vector_flops
        return self.flops if self.unit_type == unit else 0.0


class Operator(ABC):
    """Abstract base class for all operators."""

    COMPUTE_UNIT: ComputeUnitType = "vector"
    PARAM_CLASS: Optional[type] = None
    OP_TYPE: Optional["OpType"] = None  # Set by @Registry.register_operator

    def __init__(self, **kwargs: Any) -> None:
        # 先提取 quant_config，不参与后续参数传递
        quant_config = kwargs.pop("quant_config", None)
        # 使用传入的 quant_config 覆盖全局配置
        self.quant_config = quant_config or QuantConfig()
        self.compute_dtype = self.quant_config.compute_dtype
        self.weight_storage_dtype = self.quant_config.weight_storage_dtype
        self.kv_storage_dtype = self.quant_config.kv_storage_dtype
        self.communicate_dtype = self.quant_config.communicate_dtype

        self.params = self.PARAM_CLASS(**kwargs) if self.PARAM_CLASS is not None else None

        self.parse_args(**kwargs)

    @staticmethod
    def memory_details(
        *, activation: int = 0, weight: int = 0, weight_saved: int = 0,
        kv_cache: int = 0, kv_cache_saved: int = 0, kv_cache_load: int = 0
    ) -> dict[str, int]:
        """Create standardized memory breakdown."""
        return {
            "activation": activation,
            "weight_read": weight,
            "weight_saved": weight_saved or weight,
            "kv_cache_read": kv_cache,
            "kv_cache_saved": kv_cache_saved or kv_cache,
            "kv_cache_load": kv_cache_load,
        }

    @abstractmethod
    def parse_args(self, **kwargs: Any) -> None:
        """Parse and validate operator arguments."""
        raise NotImplementedError

    @abstractmethod
    def get_flops(self) -> float:
        """Calculate total FLOPs."""
        raise NotImplementedError

    @abstractmethod
    def get_memory_details(self) -> dict[str, int]:
        """Get memory breakdown by category: 'weight', 'activation', 'kv_cache'."""
        raise NotImplementedError

    def get_quant_config(self) -> dict[str, str]:
        """Return quantization configuration."""
        return {
            "compute_dtype": self.compute_dtype,
            "weight_storage_dtype": self.weight_storage_dtype,
            "kv_storage_dtype": self.kv_storage_dtype,
            "communicate_dtype": self.communicate_dtype,
        }

    def get_memory_bytes(self) -> int:
        memory_details = self.get_memory_details()
        return memory_details.get("activation", 0) + memory_details.get("weight_read", 0) + memory_details.get("kv_cache_read", 0)

    def get_compute_component(self) -> ComputeComponent | None:
        """Return compute component for performance modeling."""
        # storage 纯内存操作：用 kv_storage_dtype，且无需计算 flops
        if self.COMPUTE_UNIT == "storage":
            return ComputeComponent(unit_type="storage", dtype=self.kv_storage_dtype)

        flops = self.get_flops()

        if flops <= 0:
            return None

        if self.COMPUTE_UNIT == "mixed":
            return ComputeComponent(
                unit_type="mixed",
                cube_flops=0.0,
                vector_flops=0.0,
                dtype=self.compute_dtype
            )

        return ComputeComponent(
            flops=flops,
            unit_type=self.COMPUTE_UNIT,
            dtype=self.compute_dtype
        )


class OpType(Enum):
    COMPUTE = auto()
    COMMUNICATION = auto()
    MEMORY = auto()


class Registry:
    """Attach cost metadata without a runtime backend registry."""

    @staticmethod
    def register_operator(name, *, op_type=OpType.COMPUTE):
        def decorate(operator_class):
            operator_class.OP_NAME = name
            operator_class.OP_TYPE = op_type
            return operator_class
        return decorate

@dataclass(frozen=True)
class OperatorStage:
    name: str
    cube_flops: float = 0.0
    vector_flops: float = 0.0
    activation_bytes: int = 0
    weight_read_bytes: int = 0
    weight_saved_bytes: int = 0
    compute_dtype: str = "fp32"
    cube_compute_multiplier: float = 1.0
    vector_compute_multiplier: float = 1.0

    def operator(self) -> "_StageOperator":
        return _StageOperator(stage=self, quant_config=QuantConfig(compute_dtype=self.compute_dtype))


class _StageOperator(Operator):
    def parse_args(self, **kwargs):
        self.stage = kwargs["stage"]

    def get_flops(self):
        return self.stage.cube_flops + self.stage.vector_flops

    def get_memory_details(self):
        s = self.stage
        return {"activation": s.activation_bytes, "weight_read": s.weight_read_bytes,
                "weight_saved": s.weight_saved_bytes, "kv_cache_read": 0,
                "kv_cache_saved": 0, "kv_cache_load": 0}

    def get_compute_component(self):
        s = self.stage
        cube = s.cube_flops * s.cube_compute_multiplier
        vector = s.vector_flops * s.vector_compute_multiplier
        if cube and vector:
            return ComputeComponent(unit_type="mixed", cube_flops=cube,
                                    vector_flops=vector, dtype=s.compute_dtype)
        # Zero-FLOP copies use the vector launch cost, not a fictitious cube launch.
        return ComputeComponent(unit_type="cube" if cube else "vector",
                                flops=cube or vector, dtype=s.compute_dtype)


class CompositeOperator(Operator):
    """One graph node with an ordered list of independently costed subcalls."""
    @abstractmethod
    def get_stages(self) -> list[OperatorStage]:
        raise NotImplementedError

    def get_flops(self):
        return sum(s.cube_flops + s.vector_flops for s in self.get_stages())

    def get_memory_details(self):
        # Like existing standard operators, activation denotes traffic, not a
        # measured live allocation. Capacity users must keep this distinction.
        stages = self.get_stages()
        return {"activation": sum(s.activation_bytes for s in stages),
                "weight_read": sum(s.weight_read_bytes for s in stages),
                "weight_saved": sum(s.weight_saved_bytes for s in stages),
                "kv_cache_read": 0, "kv_cache_saved": 0, "kv_cache_load": 0}
