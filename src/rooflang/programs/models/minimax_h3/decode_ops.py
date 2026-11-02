"""Analytical device-kernel costs with explicit physical tensor precision.

Decode traces establish topology and tensor boundaries, never latency constants.
Inputs describe bytes actually touched (a sliced view need not read its backing
allocation). Activation traffic is a working-set proxy, not allocator liveness.
"""
from dataclasses import dataclass, field
import math

from rooflang.programs.models.magi2_preview._cost.core import (
    ComputeComponent, Operator, QuantConfig,
)


DTYPE_BYTES = {'fp16': 2, 'bf16': 2, 'fp32': 4, 'int32': 4, 'int64': 8,
               'uint8': 1, 'bool': 1}


@dataclass
class DecodeParams:
    op_type: str
    input_shapes: tuple
    input_dtypes: tuple
    output_shapes: tuple
    output_dtypes: tuple
    weight_shapes: tuple
    weight_dtypes: tuple
    attributes: dict = field(default_factory=dict)

    def validate(self):
        if not self.op_type:
            raise ValueError('Decode op_type must be non-empty')
        for shapes, dtypes in ((self.input_shapes, self.input_dtypes),
                               (self.output_shapes, self.output_dtypes),
                               (self.weight_shapes, self.weight_dtypes)):
            if len(shapes) != len(dtypes):
                raise ValueError('Decode shapes and dtypes must align')
            for shape, dtype in zip(shapes, dtypes):
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f'Unsupported decode storage dtype: {dtype}')
                if any(type(dim) is not int or dim < 1 for dim in shape):
                    raise ValueError('Decode tensor dimensions must be positive integers')


class DecodeOperator(Operator):
    """One observed kernel boundary, evaluated by the normal hardware Roofline."""

    def __init__(self, op_type, *, inputs=(), outputs=(), weights=(),
                 cube_flops=0, vector_flops=0, compute_dtype='fp32',
                 vector_dtype='fp32', weight_saved_bytes=None, attributes=None,
                 input_read_bytes=None, output_write_bytes=None):
        super().__init__(quant_config=QuantConfig(compute_dtype, compute_dtype,
                                                 compute_dtype, compute_dtype))
        def unpack(specs):
            return (tuple(tuple(shape) for shape, _ in specs),
                    tuple(dtype for _, dtype in specs))
        self.params = DecodeParams(op_type, *unpack(inputs), *unpack(outputs),
                                   *unpack(weights), dict(attributes or {}))
        self.params.validate()
        if compute_dtype not in ('fp16', 'bf16', 'fp32') or vector_dtype not in ('fp16', 'bf16', 'fp32'):
            raise ValueError('Unsupported decode arithmetic dtype')
        if any(not math.isfinite(f) or f < 0 for f in (cube_flops, vector_flops)):
            raise ValueError('Decode FLOPs must be finite and nonnegative')
        if weight_saved_bytes is not None and (type(weight_saved_bytes) is not int or weight_saved_bytes < 0):
            raise ValueError('Decode saved weights must be nonnegative integer bytes')
        self.cube_flops, self.vector_flops = cube_flops, vector_flops
        self.vector_compute_dtype = vector_dtype
        size = lambda specs: sum(math.prod(shape) * DTYPE_BYTES[dtype] for shape, dtype in specs)
        input_bytes = size(inputs)
        if input_read_bytes is not None:
            if type(input_read_bytes) is not int or not 0 <= input_read_bytes <= input_bytes:
                raise ValueError('Decode view read bytes must lie within the input tensors')
            input_bytes = input_read_bytes
            self.params.attributes['input_read_bytes'] = input_read_bytes
        output_bytes = size(outputs)
        if output_write_bytes is not None:
            if type(output_write_bytes) is not int or not 0 <= output_write_bytes <= output_bytes:
                raise ValueError('Decode view write bytes must lie within the output tensors')
            output_bytes = output_write_bytes
            self.params.attributes['output_write_bytes'] = output_write_bytes
        self.activation_bytes = input_bytes + output_bytes
        self.weight_read_bytes = size(weights)
        self.weight_saved_bytes = self.weight_read_bytes if weight_saved_bytes is None else weight_saved_bytes

    def parse_args(self, **kwargs):
        pass

    def get_flops(self):
        return self.cube_flops + self.vector_flops

    def get_memory_details(self):
        return {'activation': self.activation_bytes, 'weight_read': self.weight_read_bytes,
                'weight_saved': self.weight_saved_bytes}

    def get_compute_component(self):
        if self.cube_flops and self.vector_flops:
            return ComputeComponent(unit_type='mixed', cube_flops=self.cube_flops,
                                    vector_flops=self.vector_flops, dtype=self.compute_dtype,
                                    cube_dtype=self.compute_dtype, vector_dtype=self.vector_compute_dtype)
        return ComputeComponent(unit_type='cube' if self.cube_flops else 'vector',
                                flops=self.cube_flops or self.vector_flops,
                                dtype=self.compute_dtype if self.cube_flops else self.vector_compute_dtype)
