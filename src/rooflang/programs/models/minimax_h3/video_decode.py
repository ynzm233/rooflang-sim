"""Video decoder kernel boundaries reconstructed from measured/video_vae_decode.csv.

The trace supplies shapes, precision and multiplicity, not timing constants.
One invocation is a seven-latent-frame spatial tile. The workflow owns its
tile/chunk multiplier and runs the separate stitching graph once per request.
"""
import math

from .compat import ModuleType, Node, OpType
from .decode_ops import DecodeOperator


def video_decode_nodes(seq_len, mm):
    h, w = min(mm['target_height'], 256) // 16, min(mm['target_width'], 256) // 16
    patches = 7 * h * w
    if seq_len != patches + 5:
        raise ValueError('Video decoder sequence must equal seven-frame tile patches plus five tokens')
    nodes = []
    hidden = (1, seq_len, 2048)
    tokens = (1, patches, 24)
    latent = (1, 24, 7, h, w)

    def emit(name, kind, *, layer=0, **kwargs):
        nodes.append(Node(f'VideoDecoder_{name}', DecodeOperator(kind, **kwargs),
                          OpType.COMPUTE, ModuleType.VAEDECODER, 1024, layer,
                          'vae_decoder_video.measured_decode'))

    def move(name, kind, source, target, dtype='fp16', *, out_dtype=None, layer=0):
        emit(name, kind, layer=layer, inputs=((source, dtype),),
             outputs=((target, out_dtype or dtype),),
             input_read_bytes=math.prod(target) * (2 if dtype == 'fp16' else 4)
             if kind in ('Slice', 'AsStrided') else None)

    def cast_weight(name, shape):
        emit(name, 'Cast', weights=((shape, 'fp32'),), outputs=((shape, 'fp16'),))

    def matmul(name, rows, cin, cout, *, layer=0, kind='MatMulV3', bias=True, saved=None):
        weights = [((cout, cin), 'fp16')]
        if bias:
            weights.append(((cout,), 'fp16'))
        emit(name, kind, layer=layer, inputs=(((rows, cin), 'fp16'),),
             weights=weights, outputs=(((rows, cout), 'fp16'),),
             cube_flops=2 * rows * cin * cout, vector_flops=rows * cout if bias else 0,
             compute_dtype='fp16', weight_saved_bytes=saved)

    def norm(name, shape, *, layer, affine):
        reduced = (*shape[:-1], 1)
        count, rows = math.prod(shape), math.prod(reduced)
        if not affine:
            move(name + '_CastIn', 'Cast', shape, shape, out_dtype='fp32', layer=layer)
        emit(name + '_Pows', 'Pows', layer=layer, inputs=((shape, 'fp32'),),
             outputs=((shape, 'fp32'),), vector_flops=count,
             attributes={'exponent': 2, 'flops_per_element': 1})
        emit(name + '_Mean', 'ReduceMean', layer=layer, inputs=((shape, 'fp32'),),
             outputs=((reduced, 'fp32'),), vector_flops=count,
             attributes={'axis': -1})
        emit(name + '_Epsilon', 'Add', layer=layer, inputs=((reduced, 'fp32'),),
             outputs=((reduced, 'fp32'),), vector_flops=rows,
             attributes={'scalar_parameter': 'epsilon'})
        emit(name + '_Rsqrt', 'Rsqrt', layer=layer, inputs=((reduced, 'fp32'),),
             outputs=((reduced, 'fp32'),), vector_flops=rows,
             attributes={'flops_per_element': 1})
        emit(name + '_Normalize', 'Mul', layer=layer,
             inputs=((shape, 'fp32'), (reduced, 'fp32')),
             outputs=((shape, 'fp32'),), vector_flops=count)
        if affine:
            emit(name + '_Gamma', 'Mul', layer=layer, inputs=((shape, 'fp32'),),
                 weights=(((shape[-1],), 'fp32'),), outputs=((shape, 'fp32'),),
                 vector_flops=count)
        move(name + '_CastOut', 'Cast', shape, shape, 'fp32', out_dtype='fp16', layer=layer)

    # These dtype/layout conversions are distinct device kernels in the trace.
    move('LatentSlice', 'Slice', (24, mm['target_latent_frames'],
         mm['target_height'] // 16, mm['target_width'] // 16), (24, 7, h, w), 'fp32')
    move('LatentCast', 'Cast', latent, latent, 'fp32', out_dtype='fp16')
    cast_weight('PostQuantWeightCast', (24, 24, 1, 1, 1))
    cast_weight('PostQuantBiasCast', (24,))
    move('PostQuantBiasKernelCast', 'Cast', (24,), (24,), out_dtype='fp32')
    emit('PostQuantConv', 'Conv3DV2', inputs=((latent, 'fp16'),),
         weights=(((24, 24, 1, 1, 1), 'fp16'), ((24,), 'fp32')),
         outputs=((latent, 'fp16'),), cube_flops=2 * patches * 24 * 24,
         vector_flops=patches * 24, compute_dtype='fp16', weight_saved_bytes=0,
         attributes={'kernel': [1, 1, 1], 'stride': [1, 1, 1]})
    move('TokensTransposeFP16', 'Transpose', (24, patches), (patches, 24))
    move('TokensCastFP32', 'Cast', tokens, tokens, out_dtype='fp32')
    move('ChannelsTransposeFP32', 'Transpose', (patches, 24), (24, patches), 'fp32')
    move('TokensTransposeFP32', 'Transpose', (24, patches), (patches, 24), 'fp32')
    move('TokensCastFP16', 'Cast', tokens, tokens, 'fp32', out_dtype='fp16')
    move('ChannelsTransposeFP16', 'Transpose', (patches, 24), (24, patches))
    move('XEmbedderTranspose', 'Transpose', (24, patches), (patches, 24))
    cast_weight('XEmbedderWeightCast', (2048, 24))
    cast_weight('XEmbedderBiasCast', (2048,))
    matmul('XEmbedder', patches, 24, 2048, kind='MatMulV2', bias=False, saved=0)
    patch_hidden = (1, patches, 2048)
    emit('XEmbedderBias', 'Add', inputs=((patch_hidden, 'fp16'),),
         weights=(((2048,), 'fp16'),), outputs=((patch_hidden, 'fp16'),),
         vector_flops=patches * 2048, weight_saved_bytes=0)
    move('XEmbedderCast', 'Cast', patch_hidden, patch_hidden, out_dtype='fp32')
    emit('ZeroToken', 'ZerosLike', inputs=(((1, 1, 2048), 'fp16'),),
         outputs=(((1, 1, 2048), 'fp16'),), input_read_bytes=0)
    move('ZeroTokenCast', 'Cast', (1, 1, 2048), (1, 1, 2048), out_dtype='fp32')
    emit('RegisterConcat', 'ConcatD', inputs=((patch_hidden, 'fp32'), ((1, 1, 2048), 'fp32')),
         weights=(((1, 4, 2048), 'fp32'),), outputs=((hidden, 'fp32'),))

    for layer in range(1, 37):
        prefix = f'L{layer}_'
        norm(prefix + 'Norm1', hidden, layer=layer, affine=True)
        matmul(prefix + 'QKV', seq_len, 2048, 6144, layer=layer)
        for qk in ('Q', 'K'):
            move(prefix + qk + 'Slice', 'Slice', (seq_len, 32, 192),
                 (seq_len, 32, 64), layer=layer)
            norm(prefix + qk + 'Norm', (1, seq_len, 32, 64), layer=layer, affine=False)
            rope = prefix + qk + 'RoPE_'
            for label, width in (('RotarySlice', 48), ('LeftSlice', 24),
                                 ('RightSlice', 24), ('PassSlice', 16)):
                move(rope + label, 'Slice', (seq_len, 32, 64),
                     (seq_len, 32, width), layer=layer)
            half = (1, seq_len, 32, 24)
            rotary = (1, seq_len, 32, 48)
            emit(rope + 'Neg', 'Neg', layer=layer, inputs=((half, 'fp16'),),
                 outputs=((half, 'fp16'),), vector_flops=math.prod(half))
            emit(rope + 'RotateConcat', 'ConcatD', layer=layer, inputs=((half, 'fp16'),) * 2,
                 outputs=((rotary, 'fp16'),))
            for label in ('MulCos', 'MulSin'):
                emit(rope + label, 'Mul', layer=layer,
                     inputs=((rotary, 'fp16'), ((1, seq_len, 1, 48), 'fp16')),
                     outputs=((rotary, 'fp16'),), vector_flops=math.prod(rotary))
            emit(rope + 'Add', 'Add', layer=layer, inputs=((rotary, 'fp16'),) * 2,
                 outputs=((rotary, 'fp16'),), vector_flops=math.prod(rotary))
            emit(rope + 'PassConcat', 'ConcatD', layer=layer,
                 inputs=((rotary, 'fp16'), ((1, seq_len, 32, 16), 'fp16')),
                 outputs=(((1, seq_len, 32, 64), 'fp16'),))
            move(prefix + qk + 'AttentionTranspose', 'Transpose', (seq_len, 32, 64),
                 (32, seq_len, 64), layer=layer)
        attention = (1, 32, seq_len, 64)
        move(prefix + 'VAsStrided', 'AsStrided', (3 * seq_len * 2048 - 128,),
             attention, layer=layer)
        pairs = 32 * seq_len**2
        emit(prefix + 'FlashAttention', 'FlashAttentionScore', layer=layer,
             inputs=((attention, 'fp16'),) * 3,
             outputs=(((1, 32, seq_len, 8), 'fp32'),) * 2 + ((attention, 'fp16'),),
             cube_flops=4 * pairs * 64, vector_flops=15 * pairs,
             compute_dtype='fp16', vector_dtype='fp32',
             attributes={'causal': False, 'head_dim': 64, 'num_heads': 32,
                         'q_len': seq_len, 'kv_len': seq_len,
                         'softmax_flops_per_pair': 14, 'scale_flops_per_pair': 1,
                         'traffic_scope': 'QKVO and observed softmax statistics; no score spill'})
        move(prefix + 'AttentionOutputTranspose', 'Transpose', (32, seq_len, 64),
             (seq_len, 32, 64), layer=layer)
        matmul(prefix + 'AttentionOut', seq_len, 2048, 2048, layer=layer)
        for branch in (1, 2):
            if branch == 2:
                norm(prefix + 'Norm2', hidden, layer=layer, affine=True)
                matmul(prefix + 'MlpUp', seq_len, 2048, 16384, layer=layer, kind='MatMulV2')
                emit(prefix + 'SwiGlu', 'SwiGlu', layer=layer,
                     inputs=(((1, seq_len, 16384), 'fp16'),),
                     outputs=(((1, seq_len, 8192), 'fp16'),), vector_flops=5 * seq_len * 8192,
                     attributes={'flops_per_output': 5, 'convention': 'SiLU4 plus gate multiply1'})
                matmul(prefix + 'MlpDown', seq_len, 8192, 2048, layer=layer)
            emit(prefix + f'LayerScale{branch}', 'Mul', layer=layer,
                 inputs=((hidden, 'fp16'),), weights=(((2048,), 'fp32'),),
                 outputs=((hidden, 'fp32'),), vector_flops=math.prod(hidden))
            emit(prefix + f'Residual{branch}', 'Add', layer=layer, inputs=((hidden, 'fp32'),) * 2,
                 outputs=((hidden, 'fp32'),), vector_flops=math.prod(hidden))

    emit('FinalNorm', 'LayerNormV3', inputs=((hidden, 'fp32'),),
         weights=(((2048,), 'fp32'),) * 2,
         outputs=((hidden, 'fp32'), ((1, seq_len, 1), 'fp32'), ((1, seq_len, 1), 'fp32')),
         vector_flops=8 * math.prod(hidden), attributes={'flops_per_element': 8})
    move('FinalNormCast', 'Cast', hidden, hidden, 'fp32', out_dtype='fp16')
    cast_weight('ProjOutWeightCast', (3072, 2048))
    cast_weight('ProjOutBiasCast', (3072,))
    matmul('ProjOut', seq_len, 2048, 3072, saved=0)
    move('ProjOutCast', 'Cast', (1, seq_len, 3072), (1, seq_len, 3072), out_dtype='fp32')
    # Removing the five register tokens is a view; only unpatchify materializes.
    move('Unpatchify', 'Transpose', (7, h, w, 3, 4, 16, 16),
         (3, 7, 4, h, 16, w, 16), 'fp32')
    return nodes


def video_decode_stitch_nodes(mm):
    """Materialized spatial/temporal stitching outside the repeated tile graph.

    For other resolutions, distribute overlap in 16-pixel units as evenly as
    possible. This extrapolation reproduces the measured 80/96-pixel vertical
    and 80/64-pixel horizontal overlap histogram at 768x1344.
    """
    height, width = mm['target_height'], mm['target_width']
    th, tw = min(height, 256), min(width, 256)
    chunks = (mm['target_latent_frames'] - 2) // 5
    if chunks < 1:
        raise ValueError('Video decode stitching requires at least one temporal chunk')
    frames = 17 * chunks + 5
    nodes = []

    def overlaps(length, tile):
        count = max(1, math.ceil((length - 64) / 192))
        if count == 1:
            return [0]
        base, extra = divmod((count * tile - length) // 16, count - 1)
        return [0] + [16 * (base + (i < extra)) for i in range(count - 1)]

    def emit(name, kind, *, inputs=(), outputs=(), **kwargs):
        op = DecodeOperator(kind, inputs=tuple((s, 'fp32') for s in inputs),
                            outputs=tuple((s, 'fp32') for s in outputs), **kwargs)
        nodes.append(Node('VideoStitch_' + name, op, OpType.COMPUTE,
                          ModuleType.VAEDECODER, 1024, 0, 'vae_decoder_video.measured_stitch'))

    def slice_(name, source, target):
        emit(name, 'Slice', inputs=(source,), outputs=(target,),
             input_read_bytes=4 * math.prod(target))

    def copy(name, backing_elements, source):
        touched = 4 * math.prod(source)
        emit(name, 'ViewCopy', inputs=((backing_elements,), source),
             outputs=((backing_elements,),), input_read_bytes=touched,
             output_write_bytes=touched,
             attributes={'traffic_scope': 'source view read and destination view write'})

    def blend(name, source, overlap_shape, remainder_shape, mask):
        # Two neighboring overlap slices, two weighted sides, then their sum.
        for side in ('Previous', 'Current'):
            slice_(name + side + 'Slice', source, overlap_shape[1:])
            emit(name + side + 'Mul', 'Mul', inputs=(overlap_shape, mask),
                 outputs=(overlap_shape,), vector_flops=math.prod(overlap_shape))
        emit(name + 'Add', 'Add', inputs=(overlap_shape,) * 2,
             outputs=(overlap_shape,), vector_flops=math.prod(overlap_shape))
        slice_(name + 'Remainder', source, remainder_shape[1:])
        emit(name + 'Concat', 'ConcatD', inputs=(overlap_shape, remainder_shape),
             outputs=((1, *source),))

    y_overlaps, x_overlaps = overlaps(height, th), overlaps(width, tw)
    for chunk in range(chunks):
        for row, oy in enumerate(y_overlaps):
            for col, ox in enumerate(x_overlaps):
                name = f'T{chunk}_Y{row}_X{col}_'
                source = (3, 28, th, tw)
                if oy:
                    blend(name + 'Vertical', source, (1, 3, 28, oy, tw),
                          (1, 3, 28, th - oy, tw), (1, 1, 1, oy, 1))
                if ox:
                    blend(name + 'Horizontal', source, (1, 3, 28, th, ox),
                          (1, 3, 28, th, tw - ox), (1, 1, 1, 1, ox))
                target = (3, 28, th - oy, tw - ox)
                if oy or ox:
                    slice_(name + 'Crop', source, target)
                copy(name + 'Copy', 3 * 28 * height * width, (1, *target))

    chunk_source = (3, 28, height, width)
    output_elements = 3 * frames * height * width
    overlap = (1, 3, 5, height, width)
    segment = (1, 3, 17, height, width)
    remainder = (1, 3, 12, height, width)
    slice_('FirstFrames', chunk_source, segment[1:])
    copy('FirstFramesCopy', output_elements, segment)
    for chunk in range(1, chunks):
        name = f'Temporal{chunk}_'
        for side in ('Previous', 'Current'):
            slice_(name + side + 'Slice', chunk_source, overlap[1:])
            emit(name + side + 'Mul', 'Mul', inputs=(overlap, (1, 1, 5, 1, 1)),
                 outputs=(overlap,), vector_flops=math.prod(overlap))
        emit(name + 'Add', 'Add', inputs=(overlap,) * 2,
             outputs=(overlap,), vector_flops=math.prod(overlap))
        slice_(name + 'Remainder', chunk_source, remainder[1:])
        emit(name + 'Concat', 'ConcatD', inputs=(overlap, remainder), outputs=(segment,))
        copy(name + 'Copy', output_elements, segment)
    slice_('LastFrames', chunk_source, overlap[1:])
    copy('LastFramesCopy', output_elements, overlap)
    return nodes
