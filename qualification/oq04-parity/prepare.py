#!/usr/bin/env python3
"""Freeze the OQ04 parity matrix with pinned Python MLX reference outputs."""

import argparse
import hashlib
import json
import platform
import zlib
from pathlib import Path

import mlx.core as mx

TOLERANCES = {
    'selection': {'abs': 0.0, 'rel': 0.0},
    'matmul': {'abs': 2e-5, 'rel': 2e-5},
    'matmul-decomposed': {'abs': 1e-4, 'rel': 1e-4},
}


def draw(seed, count, lo, hi):
    state = seed & 0xFFFFFFFF
    values = []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values.append(lo + ((state >> 8) % (hi - lo + 1)))
    return values


def exact_values(seed, count):
    return [value / 16.0 for value in draw(seed, count, -63, 63)]


def matrix(seed, shape, dtype):
    count = 1
    for dim in shape:
        count *= dim
    return {'shape': list(shape), 'dtype': dtype, 'values': exact_values(seed, count)}


def draw_full(seed, count):
    state = seed & 0xFFFFFFFF
    values = []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values.append(state)
    return values


def packed_weight_full(seed, logical, quant):
    """Like packed_weight, but every payload word spans the full 32-bit range."""
    width = logical[-1]
    bits = quant['bits']
    group = quant['group_size']
    rows = 1
    for dim in logical[:-1]:
        rows *= dim
    packed_width = width * bits // 32
    payload = {'shape': logical[:-1] + [packed_width], 'words': draw_full(seed, rows * packed_width)}
    scale_shape = logical[:-1] + [width // group]
    scale_count = rows * (width // group)
    scales = {'shape': scale_shape, 'dtype': 'f32',
              'values': [value / 8.0 for value in draw(seed + 1, scale_count, 4, 44)]}
    biases = {'shape': scale_shape, 'dtype': 'f32',
              'values': [value / 8.0 for value in draw(seed + 2, scale_count, -32, 32)]}
    return {'logical_shape': list(logical), 'payload': payload, 'scales': scales, 'biases': biases}


def packed_weight(seed, logical, quant):
    width = logical[-1]
    bits = quant['bits']
    group = quant['group_size']
    rows = 1
    for dim in logical[:-1]:
        rows *= dim
    packed_width = width * bits // 32
    payload = {'shape': logical[:-1] + [packed_width], 'words': draw(seed, rows * packed_width, 0, 2**32 - 1)}
    scale_shape = logical[:-1] + [width // group]
    scale_count = rows * (width // group)
    if quant['mode'] == 'affine':
        scales = {'shape': scale_shape, 'dtype': 'f32',
                  'values': [value / 8.0 for value in draw(seed + 1, scale_count, 4, 44)]}
        biases = {'shape': scale_shape, 'dtype': 'f32',
                  'values': [value / 8.0 for value in draw(seed + 2, scale_count, -32, 32)]}
    else:
        scales = {'shape': scale_shape, 'dtype': 'u8', 'values': draw(seed + 1, scale_count, 120, 134)}
        biases = None
    return {'logical_shape': list(logical), 'payload': payload, 'scales': scales, 'biases': biases}


def cell(ident, kind, family, quant, activation, weight, indices, table):
    return {
        'id': ident,
        'kind': kind,
        'family': family,
        'quant': quant,
        'activation': activation,
        'weight': weight,
        'indices': indices,
        'table': table,
    }


def quant_matrix(quant, bits, group):
    if quant is None:
        return None
    return {'mode': quant['mode'], 'bits': bits, 'group_size': group}


def dense_cells():
    cells = []
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            ident = f'dense-affine-b{bits}-g{group}'
            seed = zlib.crc32(ident.encode())
            quant = quant_matrix({'mode': 'affine'}, bits, group)
            cells.append(cell(ident, 'dequant_matmul', 'matmul', quant,
                              matrix(seed + 1, (2, 128), 'f32'),
                              packed_weight(seed + 2, [2, 128], quant), None, None))
    for bits in (3, 5, 6):
        ident = f'dense-affine-fullwords-b{bits}-g64'
        seed = zlib.crc32(ident.encode())
        quant = quant_matrix({'mode': 'affine'}, bits, 64)
        cells.append(cell(ident, 'dequant_matmul', 'matmul', quant,
                          matrix(seed + 1, (2, 128), 'f32'),
                          packed_weight_full(seed + 2, [2, 128], quant), None, None))
    for mode, bits in (('mxfp4', 4), ('mxfp8', 8)):
        ident = f'dense-{mode}'
        seed = zlib.crc32(ident.encode())
        quant = quant_matrix({'mode': mode}, bits, 32)
        cells.append(cell(ident, 'dequant_matmul', 'matmul', quant,
                          matrix(seed + 1, (2, 128), 'f32'),
                          packed_weight(seed + 2, [2, 128], quant), None, None))
    for dtype in ('f32', 'f16', 'bf16'):
        ident = f'dense-float-{dtype}'
        seed = zlib.crc32(ident.encode())
        cells.append(cell(ident, 'matmul', 'matmul', None,
                          matrix(seed + 1, (2, 8), dtype),
                          matrix(seed + 2, (8, 3), dtype), None, None))
    return cells


def routed_cells():
    cells = []
    for name, quant in (
        ('routed-affine-b4-g32', quant_matrix({'mode': 'affine'}, 4, 32)),
        ('routed-affine-b2-g64', quant_matrix({'mode': 'affine'}, 2, 64)),
        ('routed-mxfp4', quant_matrix({'mode': 'mxfp4'}, 4, 32)),
        ('routed-mxfp8', quant_matrix({'mode': 'mxfp8'}, 8, 32)),
    ):
        seed = zlib.crc32(name.encode())
        cells.append(cell(name, 'gather_dequant_matmul', 'matmul', quant,
                          matrix(seed + 1, (3, 128), 'f32'),
                          packed_weight(seed + 2, [2, 2, 128], quant),
                          {'dtype': 'u32', 'values': [0, 1, 0]}, None))
    seed = zlib.crc32('routed-float-f32'.encode())
    cells.append(cell('routed-float-f32', 'gather_matmul', 'matmul', None,
                      matrix(seed + 1, (3, 8), 'f32'),
                      matrix(seed + 2, (2, 8, 2), 'f32'),
                      {'dtype': 'u32', 'values': [0, 1, 0]}, None))
    return cells


def embedding_cells():
    cells = []
    seed = zlib.crc32('embed-float-f32'.encode())
    cells.append(cell('embed-float-f32', 'embedding_rows', 'selection', None, None, None,
                      {'dtype': 'u32', 'values': [2, 0, 3]}, matrix(seed + 1, (4, 8), 'f32')))
    for name, quant in (
        ('embed-affine-b4-g32', quant_matrix({'mode': 'affine'}, 4, 32)),
        ('embed-mxfp4', quant_matrix({'mode': 'mxfp4'}, 4, 32)),
    ):
        seed = zlib.crc32(name.encode())
        cells.append(cell(name, 'quantized_embedding_rows', 'selection', quant, None,
                          packed_weight(seed + 2, [4, 128], quant),
                          {'dtype': 'u32', 'values': [2, 0, 3]}, None))
    return cells


def mx_tensor(item, dtype):
    return mx.array(item['values'], dtype).reshape(item['shape'])


def mx_float(item):
    dtype = {'f32': mx.float32, 'f16': mx.float16, 'bf16': mx.bfloat16}[item['dtype']]
    return mx_tensor(item, dtype)


def mx_indices(item):
    return mx.array(item['values'], mx.uint32)


def mx_weight_parts(weight):
    payload = mx.array(weight['payload']['words'], mx.uint32).reshape(weight['payload']['shape'])
    scales = mx_tensor(weight['scales'], mx.uint8 if weight['scales']['dtype'] == 'u8' else mx.float32)
    biases = None
    if weight['biases'] is not None:
        biases = mx_tensor(weight['biases'], mx.float32)
    return payload, scales, biases


def quant_kwargs(quant):
    if quant['mode'] == 'affine':
        return {'group_size': quant['group_size'], 'bits': quant['bits'], 'mode': 'affine'}
    return {'group_size': None, 'bits': None, 'mode': quant['mode']}


def freeze(out):
    mx.eval(out)
    values = out.astype(mx.float32).reshape(-1).tolist()
    for value in values:
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError('nonfinite reference value')
    return list(out.shape), values


def dense_quant_reference(item):
    x = mx_float(item['activation'])
    payload, scales, biases = mx_weight_parts(item['weight'])
    out = mx.quantized_matmul(x, payload, scales, biases, transpose=True, **quant_kwargs(item['quant']))
    return freeze(out) + ('mx.quantized_matmul(transpose=True)',)


def dense_float_reference(item):
    x = mx_float(item['activation'])
    weight = mx_float(item['weight'])
    out = mx.matmul(x, weight)
    return freeze(out) + ('mx.matmul',)


def routed_quant_reference(item):
    x = mx_float(item['activation']).reshape([item['activation']['shape'][0], 1, item['activation']['shape'][1]])
    payload, scales, biases = mx_weight_parts(item['weight'])
    indices = mx_indices(item['indices'])
    kwargs = quant_kwargs(item['quant'])
    try:
        out = mx.gather_qmm(x, payload, scales, biases, lhs_indices=None,
                            rhs_indices=indices, transpose=True, **kwargs)
        return freeze(out) + ('mx.gather_qmm(rhs_indices)',)
    except (AttributeError, TypeError):
        rows = []
        for step, expert in enumerate(item['indices']['values']):
            rows.append(mx.quantized_matmul(x[step], payload[expert], scales[expert],
                                            None if biases is None else biases[expert],
                                            transpose=True, **kwargs))
        return freeze(mx.concatenate(rows, axis=0)) + ('per-token mx.quantized_matmul loop',)


def routed_float_reference(item):
    x = mx_float(item['activation']).reshape([item['activation']['shape'][0], 1, item['activation']['shape'][1]])
    weight = mx_float(item['weight'])
    indices = mx_indices(item['indices'])
    try:
        out = mx.gather_mm(x, weight, lhs_indices=None, rhs_indices=indices)
        return freeze(out) + ('mx.gather_mm(rhs_indices)',)
    except (AttributeError, TypeError):
        rows = []
        for step, expert in enumerate(item['indices']['values']):
            rows.append(mx.matmul(x[step], weight[expert]))
        return freeze(mx.concatenate(rows, axis=0)) + ('per-token mx.matmul loop',)


def embedding_reference(item):
    table = mx_float(item['table'])
    out = mx.take(table, mx_indices(item['indices']), axis=0)
    return freeze(out) + ('mx.take(axis=0)',)


def quantized_embedding_reference(item):
    payload, scales, biases = mx_weight_parts(item['weight'])
    indices = mx_indices(item['indices'])
    rows = mx.take(payload, indices, axis=0)
    scale_rows = mx.take(scales, indices, axis=0)
    bias_rows = None if biases is None else mx.take(biases, indices, axis=0)
    kwargs = quant_kwargs(item['quant'])
    out = call_dequantize(rows, scale_rows, bias_rows, kwargs)
    return freeze(out) + ('mx.take(axis=0) then mx.dequantize',)


def call_dequantize(rows, scales, biases, kwargs):
    try:
        return mx.dequantize(rows, scales, biases, dtype=mx.float32, **kwargs)
    except TypeError:
        return mx.dequantize(rows, scales, biases, **kwargs).astype(mx.float32)


def dequantize_probe():
    words = [0x33333333] * 4
    payload = mx.array([words], mx.uint32)
    scales = mx.array([[1.5]], mx.float32)
    biases = mx.array([[0.25]], mx.float32)
    kwargs = {'group_size': 32, 'bits': 4, 'mode': 'affine'}
    out = call_dequantize(payload, scales, biases, kwargs)
    observed = float(out[0, 0])
    if observed != 4.75:
        raise ValueError(f'dequantize affine probe expects 4.75, observed {observed}')
    payload = mx.array([[0x11111111] * 4], mx.uint32)
    scales = mx.array([[127]], mx.uint8)
    kwargs = {'group_size': None, 'bits': None, 'mode': 'mxfp4'}
    out = call_dequantize(payload, scales, None, kwargs)
    observed = float(out[0, 0])
    if observed != 0.5:
        raise ValueError(f'dequantize mxfp4 probe expects 0.5, observed {observed}')
    return [
        {'name': 'dequantize-affine', 'expected': 4.75, 'observed': 4.75},
        {'name': 'dequantize-mxfp4', 'expected': 0.5, 'observed': observed},
    ]


def binary_hashes():
    import glob
    root = Path(mx.__file__).parent
    hashes = {}
    for pattern in ('core.cpython-*.so', 'lib/libmlx.dylib'):
        for path in sorted(glob.glob(str(root / pattern))):
            hashes[Path(path).name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashes


def assemble():
    cells = dense_cells() + routed_cells() + embedding_cells()
    for item in cells:
        shape, values, op = reference_values(item)
        item['expected'] = {'shape': list(shape), 'values': values}
        item['reference_op'] = op
    return {
        'schema': 'oq04-parity-matrix-1',
        'provenance': {
            'mlx_python_version': mx.__version__,
            'python_version': platform.python_version(),
            'binary_hashes': binary_hashes(),
            'probes': dequantize_probe(),
        },
        'tolerances': TOLERANCES,
        'cells': cells,
    }


def reference_values(item):
    kind = item['kind']
    if kind == 'dequant_matmul':
        return dense_quant_reference(item)
    if kind == 'matmul':
        return dense_float_reference(item)
    if kind == 'gather_dequant_matmul':
        return routed_quant_reference(item)
    if kind == 'gather_matmul':
        return routed_float_reference(item)
    if kind == 'embedding_rows':
        return embedding_reference(item)
    return quantized_embedding_reference(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing existing destination: {args.output}')
    args.output.mkdir(parents=True)
    fixture = assemble()
    text = json.dumps(fixture, sort_keys=True, indent=2) + '\n'
    target = args.output / 'fixtures.json'
    target.write_text(text, encoding='utf-8')
    print('fixtures_sha256', hashlib.sha256(text.encode('utf-8')).hexdigest())
    print('cells', len(fixture['cells']))


if __name__ == '__main__':
    main()

