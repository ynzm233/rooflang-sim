import pytest


def test_decode_copy_uses_physical_storage_dtypes_without_fictitious_flops():
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator

    op = DecodeOperator('Cast', inputs=[((2, 3), 'fp32')],
                        outputs=[((2, 3), 'fp16')])
    assert op.get_flops() == 0
    assert op.get_memory_bytes() == 36
    assert op.get_compute_component().unit_type == 'vector'


def test_decode_mixed_compute_keeps_vector_fp32_and_weights_separate():
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator

    op = DecodeOperator('FlashAttentionScore', inputs=[((2, 3), 'fp16')],
                        outputs=[((2, 3), 'fp16')], weights=[((3,), 'fp16')],
                        cube_flops=100, vector_flops=25, compute_dtype='fp16')
    c = op.get_compute_component()
    assert c.get_unit_dtype('cube') == 'fp16'
    assert c.get_unit_dtype('vector') == 'fp32'
    assert op.get_memory_details() == {'activation': 24, 'weight_read': 6,
                                     'weight_saved': 6}
    assert op.get_flops() == 125


def test_decode_slice_keeps_backing_shape_but_reads_only_selected_elements():
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator

    op = DecodeOperator('Slice', inputs=[((100,), 'fp32')],
                        outputs=[((10,), 'fp32')], input_read_bytes=40)
    assert op.params.input_shapes == ((100,),)
    assert op.get_memory_bytes() == 80


def test_decode_viewcopy_writes_only_selected_output_region():
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator

    op = DecodeOperator('ViewCopy', inputs=[((10,), 'fp32')],
                        outputs=[((100,), 'fp32')], output_write_bytes=40)
    assert op.params.output_shapes == ((100,),)
    assert op.get_memory_bytes() == 80


def test_generated_weights_have_reads_but_no_resident_allocation():
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator
    from rooflang.programs.models.minimax_h3.bridge import H3Kernel

    op = DecodeOperator('Conv2D', weights=[((2, 3), 'fp32')], weight_saved_bytes=0)
    kernel = H3Kernel(op, {}, 'generated')
    assert kernel.loaded_weight_bytes == 24
    assert sum(t.storage_bytes for t in kernel.weights.values()) == 0
    assert kernel.csv_metadata['source_weight_saved'] == 0


@pytest.mark.parametrize('kwargs', [
    {'inputs': [((0, 3), 'fp32')]}, {'inputs': [((True,), 'fp32')]},
    {'outputs': [((3,), 'unknown')]}, {'cube_flops': -1},
    {'vector_flops': float('nan')}, {'weight_saved_bytes': -1},
])
def test_decode_cost_rejects_invalid_shapes_and_costs(kwargs):
    from rooflang.programs.models.minimax_h3.decode_ops import DecodeOperator

    with pytest.raises(ValueError):
        DecodeOperator('test', **kwargs)
