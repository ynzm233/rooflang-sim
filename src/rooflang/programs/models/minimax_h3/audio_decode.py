"""FP32 audio decode reconstructed from the measured Ascend leaf kernels.

The trace supplies topology and layout, never elapsed-time constants. Transcendental
functions count as one vector operation; their hardware implementation and the
slow grouped transpose algorithm remain outside this analytical Roofline model.
"""
import math

from .compat import ModuleType, Node, OpType
from .decode_ops import DecodeOperator


def audio_decode_nodes(mm):
    length = mm['target_audio_latents_per_channel']
    if type(length) is not int or length < 1:
        raise ValueError('Audio decode latent length must be a positive integer')
    builder = _AudioDecode()
    builder.conv('InputProjection', 32, 2048, length, 1, norm=False)
    builder.conv('ConvPre', 2048, 1024, length, 7, padding=3)
    channels = 1024
    for level, (rate, kernel) in enumerate(zip((5, 5, 2, 2, 2, 2, 2),
                                               (9, 9, 4, 4, 4, 4, 4))):
        out_channels = channels // 2
        length = builder.conv(f'L{level}_Upsample', channels, out_channels,
                              length, kernel, stride=rate,
                              padding=(kernel - rate) // 2, transpose=True)
        channels = out_channels
        shape = (2, channels, length)
        for branch, residual_kernel in enumerate((3, 7, 11)):
            for dilation in (1, 3, 5):
                prefix = f'L{level}_K{residual_kernel}_D{dilation}'
                for index, dil in ((1, dilation), (2, 1)):
                    builder.snake(f'{prefix}_Snake{index}', channels, length)
                    builder.conv(f'{prefix}_Conv{index}', channels, channels,
                                 length, residual_kernel, dilation=dil,
                                 padding=dil * (residual_kernel - 1) // 2)
                builder.add(f'{prefix}_Residual', 'Add', [shape, shape], [shape],
                            vector_flops=math.prod(shape))
            if branch:
                builder.add(f'L{level}_BranchAdd{branch}', 'Add', [shape, shape],
                            [shape], vector_flops=math.prod(shape))
        builder.add(f'L{level}_BranchAverage', 'RealDiv', [shape, ()], [shape],
                    vector_flops=math.prod(shape))
    builder.snake('OutputSnake', channels, length)
    builder.conv('ConvPost', channels, 1, length, 7, padding=3, bias=False)
    builder.add('Clamp', 'ClipByValueV2', [(2, 1, length), (), ()],
                [(2, 1, length)], vector_flops=4 * length)
    return builder.nodes


class _AudioDecode:
    def __init__(self):
        self.nodes = []

    def add(self, name, kind, inputs=(), outputs=(), *, weights=(), metadata=(), **kwargs):
        tensor = lambda shapes: [(shape, 'fp32') for shape in shapes]
        operator = DecodeOperator(kind, inputs=tensor(inputs) + list(metadata), outputs=tensor(outputs),
                                  weights=tensor(weights), **kwargs)
        self.nodes.append(Node('AudioDecoder_' + name, operator, OpType.COMPUTE,
                               ModuleType.VAEDECODER, 1024, 0,
                               'vae_decoder_audio.measured_decode'))

    def weight_norm(self, name, shape):
        outer = shape[0]
        count = math.prod(shape)
        self.add(name + '_WeightNormReduce', 'LpNormV2', outputs=[(outer,)],
                 weights=[(outer, count // outer)], vector_flops=2 * count + outer,
                 attributes={'reduction': 'sqrt(sum(v*v))'})
        self.add(name + '_WeightNormScale', 'RealDiv', [(outer, 1, 1)],
                 [(outer, 1, 1)], weights=[(outer, 1, 1)], vector_flops=outer)
        self.add(name + '_WeightNormApply', 'Mul', [(outer, 1, 1)], [shape],
                 weights=[shape], weight_saved_bytes=0, vector_flops=count)

    def conv(self, name, cin, cout, length, kernel, *, stride=1, padding=0,
             dilation=1, transpose=False, groups=1, bias=True, norm=True):
        batch = 2
        out = ((length - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
               if transpose else
               (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
        weight = ((cin, cout // groups, kernel) if transpose else
                  (cout, cin // groups, kernel))
        if norm:
            self.weight_norm(name, weight)
        physical_input = (batch, math.ceil(cin / 8), 1, length, 8)
        physical_output = (batch, math.ceil(cout / 8), 1, out, 8)
        self.add(name + '_InputLayout', 'TransData', [(batch, cin, 1, length)],
                 [physical_input])
        if groups > 1:
            # The trace expands a shared 12-tap filter for each grouped call.
            broadcast = (cout, 1, kernel)
            self.add(name + '_FilterBroadcast', 'BroadcastTo', outputs=[broadcast],
                     weights=[(1, 1, kernel)], metadata=[((3,), 'int64')])
            intermediate = (kernel, math.ceil(cout / 16), 16, 8)
            blocked_weight = (kernel * math.ceil(cout / 8), 1, 16, 8)
            self.add(name + '_WeightLayout', 'TransData', [(cout, 1, 1, kernel)],
                     [intermediate])
            self.add(name + '_FilterClear', 'MemSet', outputs=[blocked_weight],
                     attributes={'traffic_assumption': 'clear padded grouped filter destination; trace has no shape'})
            self.add(name + '_GroupedWeightLayout', 'TransData', [intermediate],
                     [blocked_weight])
        else:
            first, second = (cout, cin) if transpose else (cin, cout)
            blocked_weight = (kernel * math.ceil(first / 8), math.ceil(second / 16), 16, 8)
            physical_weight = (weight[0], weight[1], 1, kernel)
            self.add(name + '_WeightLayout', 'TransData',
                     [physical_weight] if norm else [], [blocked_weight],
                     weights=[] if norm else [physical_weight])
        work_length = length if transpose else out
        flops = 2 * batch * work_length * math.prod(weight)
        if bias:
            flops += batch * cout * out
        self.add(name, 'Conv2DTranspose' if transpose else 'Conv2D',
                 [physical_input, blocked_weight], [physical_output],
                 weights=[(cout,) if transpose else (1, cout, 1, 1)] if bias else [],
                 metadata=[((4,), 'int32')] if transpose else [], cube_flops=flops,
                 attributes={'logical_input_shape': [batch, cin, length],
                             'logical_output_shape': [batch, cout, out],
                             'kernel': kernel, 'stride': stride, 'padding': padding,
                             'dilation': dilation, 'groups': groups,
                             'flops_basis': 'input taps including cropped edges' if transpose else 'output taps'})
        self.add(name + '_OutputLayout', 'TransData', [physical_output],
                 [(batch, cout, 1, out)])
        return out

    def snake(self, name, channels, length):
        original = (2, channels, length)
        padded = (2, channels, length + 10)
        self.add(name + '_UpPad', 'PadV3', [original, ()], [padded],
                 metadata=[((6,), 'int64')])
        self.conv(name + '_UpFilter', channels, channels, length + 10, 12,
                  stride=2, transpose=True, groups=channels, norm=False, bias=False)
        expanded = (2, channels, 2 * length + 30)
        doubled = (2, channels, 2 * length)
        self.add(name + '_UpScale', 'Mul', [expanded, ()], [expanded],
                 vector_flops=math.prod(expanded))
        parameter = (1, channels, 1)
        for label in ('Alpha', 'Beta'):
            self.add(name + '_' + label + 'Exp', 'Exp', outputs=[parameter],
                     weights=[parameter], vector_flops=channels)
        self.add(name + '_BetaEpsilon', 'Add', [parameter, ()], [parameter],
                 vector_flops=channels)
        self.add(name + '_BetaReciprocal', 'Reciprocal', [parameter], [parameter],
                 vector_flops=channels)
        self.add(name + '_MulSlice', 'Slice', [expanded], [doubled],
                 metadata=[((3,), 'int64'), ((3,), 'int64')],
                 input_read_bytes=4 * math.prod(doubled) + 48)
        self.add(name + '_AlphaMul', 'Mul', [parameter, doubled], [doubled],
                 vector_flops=math.prod(doubled))
        self.add(name + '_Sin', 'Sin', [doubled], [doubled],
                 vector_flops=math.prod(doubled))
        self.add(name + '_Square', 'Pows', [doubled, ()], [doubled],
                 vector_flops=math.prod(doubled))
        self.add(name + '_BetaMul', 'Mul', [parameter, doubled], [doubled],
                 vector_flops=math.prod(doubled))
        self.add(name + '_AddSlice', 'Slice', [expanded], [doubled],
                 metadata=[((3,), 'int64'), ((3,), 'int64')],
                 input_read_bytes=4 * math.prod(doubled) + 48)
        self.add(name + '_Add', 'Add', [doubled, doubled], [doubled],
                 vector_flops=math.prod(doubled))
        self.add(name + '_DownPad', 'PadV3', [doubled, ()], [(2, channels, 2 * length + 11)],
                 metadata=[((6,), 'int64')])
        self.conv(name + '_DownFilter', channels, channels, 2 * length + 11, 12,
                  stride=2, groups=channels, norm=False, bias=False)
