"""Audio transpose boundary with a hardware-enabled generic tile schedule."""
from .decode_ops import DecodeOperator
from .transpose_tiling import TransposeGeometry, transpose_cost
from rooflang.programs.models.magi2_preview._cost.core import ComputeComponent


class AudioTransposeOperator(DecodeOperator):
    def __init__(self, *args, geometry: TransposeGeometry, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.tile_cost = None

    def enable_tiling(self, core_count):
        self.tile_cost = transpose_cost(self.geometry, core_count)

    def get_memory_details(self):
        details = super().get_memory_details()
        if self.tile_cost is not None:
            # Repeated payload reads change traffic, never boundary capacity.
            details.update(activation=self.tile_cost.activation_bytes,
                           weight_read=self.tile_cost.bias_read_bytes,
                           activation_peak=self.activation_bytes)
        return details

    def get_compute_component(self):
        if self.tile_cost is None:
            return super().get_compute_component()
        return ComputeComponent(unit_type='mixed', dtype='fp32',
                                cube_dtype='fp32', vector_dtype='fp32',
                                cube_flops=self.tile_cost.executed_cube_flops,
                                vector_flops=self.tile_cost.executed_vector_flops)
