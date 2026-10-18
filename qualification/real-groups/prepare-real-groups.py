#!/usr/bin/env python3
"""Freeze the 1,000 real-group parity fixture for decision 0015 against the pinned MLX build."""

import argparse
import hashlib
import json
import math
import mmap
import platform
import re
import struct
import sys
from pathlib import Path

SCHEMA = 'p0a05-real-group-1'
SEED = 'p0a05-real-group-v1'
GROUP_TARGET = 1000
NEGATIVE_TARGET = 200
CANDIDATE_LIMIT = 1024
GROUP_SIZE = 64
WORDS_PER_GROUP = 8
MLX_VERSION = '0.32.2.dev20260927+1f8e74e3'
CORE_EXTENSION_SHA256 = '935cc4e023f4a3b548981bac866e7edb708256c42b04508f5f633f1c2f7839b3'
INDEX_TENSORS = 1757
ELIGIBLE_MODULES = 432
RANK3_MODULES = 120
INT8_MODULES = 80
DEFAULT_SPEC = {'mode': 'affine', 'bits': 4, 'group_size': GROUP_SIZE}
ITEM_SIZES = {'U32': 4, 'BF16': 2, 'F16': 2, 'F32': 4, 'U8': 1}
MODULE_NAME = re.compile(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*')
INPUT_PINS = {
    'config.json': (23591, 'a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767'),
    'model.safetensors.index.json': (180182, 'e359cd95565275b923fed8df019dd8e869d9af5dc4a621b4114ab7a8e6c534d1'),
    'model-00001-of-00004.safetensors': (4395019264, 'f7fff28fecb8221e53787c6339c938b113bcb94c7bbc60d09a8826eba08ef1ab'),
    'model-00002-of-00004.safetensors': (5368472749, '31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033f'),
    'model-00003-of-00004.safetensors': (5368324139, '3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03'),
    'model-00004-of-00004.safetensors': (4377211365, 'a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422'),
}

_PACKAGE = None


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'duplicate JSON field: {name}')
        result[name] = value
    return result


def load_json(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'JSON exceeds limit: {path.name}')
    return json.loads(raw, object_pairs_hook=unique_object)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def bf16_to_f32(bits):
    # bfloat16 is the top half of an IEEE-754 binary32 value, so the shift is exact.
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def file_key(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def hash_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if file_key(before) != file_key(after):
        raise ValueError(f'source changed during hashing: {path.name}')
    return digest.hexdigest(), before.st_size


def verify_inputs(root):
    observed = {}
    for name, (size, pinned) in INPUT_PINS.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'invalid source file: {name}')
        digest, length = hash_file(path)
        if length != size or digest != pinned:
            raise ValueError(f'source identity mismatch: {name}')
        observed[name] = {'bytes': length, 'sha256': digest}
    return observed


def package_modules():
    global _PACKAGE
    if _PACKAGE is None:
        try:
            import ml_dtypes
            import numpy
            from safetensors import safe_open

            ml_dtypes, numpy
            _PACKAGE = {'safe_open': safe_open}
        except ImportError:
            _PACKAGE = {}
    return _PACKAGE


def read_header(path):
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError('safetensors header length is out of bounds')
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('truncated safetensors header')
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError('safetensors header is not an object')
    return header, 8 + length


def header_entry(name, item, data_end):
    if not isinstance(item, dict):
        raise ValueError(f'invalid header entry: {name}')
    dtype = item.get('dtype')
    shape = item.get('shape')
    offsets = item.get('data_offsets')
    if dtype not in ITEM_SIZES or not isinstance(shape, list) or not shape:
        raise ValueError(f'invalid header entry: {name}')
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f'invalid header shape: {name}')
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f'invalid header offsets: {name}')
    begin, end = offsets
    span = math.prod(shape) * ITEM_SIZES[dtype]
    if type(begin) is not int or type(end) is not int or not 0 <= begin <= end <= data_end:
        raise ValueError(f'invalid header offset range: {name}')
    if end - begin != span:
        raise ValueError(f'header span differs from tensor size: {name}')
    return dtype, shape, begin, end


class ShardReader:
    def __init__(self, path):
        self.path = path
        self.mode = 'struct-fallback'
        self._handle = None
        self._map = None
        self._stream = None
        self._entries = {}
        self._offsets = {}
        self._data_start = 0
        if package_modules():
            self._open_package()
        else:
            self._open_fallback()

    def entries(self):
        return self._entries

    def _open_package(self):
        handle = package_modules()['safe_open'](str(self.path), framework='numpy')
        entries = {}
        for key in handle.keys():
            view = handle.get_slice(key)
            entries[key] = (view.get_dtype(), tuple(view.get_shape()))
        self._handle = handle
        self._entries = entries
        self.mode = 'safetensors'

    def _open_fallback(self):
        header, data_start = read_header(self.path)
        file_size = self.path.stat().st_size
        entries = {}
        offsets = {}
        reach = 0
        for name, item in header.items():
            if name == '__metadata__':
                continue
            dtype, shape, begin, end = header_entry(name, item, file_size - data_start)
            entries[name] = (dtype, tuple(shape))
            offsets[name] = begin
            reach = max(reach, end)
        if data_start + reach != file_size:
            raise ValueError(f'safetensors data span differs from the file size: {self.path.name}')
        self._entries = entries
        self._offsets = offsets
        self._data_start = data_start
        self._stream = self.path.open('rb')
        self._map = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)

    def _check_run(self, name, shape, prefix, start, count):
        if len(prefix) != len(shape) - 1:
            raise ValueError(f'read rank differs for {name}')
        for coordinate, dimension in zip(prefix, shape[:-1]):
            if not 0 <= coordinate < dimension:
                raise ValueError(f'read coordinate out of range for {name}')
        if count <= 0 or start < 0 or start + count > shape[-1]:
            raise ValueError(f'read run out of range for {name}')

    def _read_package(self, name, prefix, start, count):
        selection = tuple(slice(c, c + 1) for c in prefix) + (slice(start, start + count),)
        array = self._handle.get_slice(name)[selection]
        return array.reshape(-1).tobytes()

    def _read_bytes(self, name, shape, prefix, start, count):
        dtype, _ = self._entries[name]
        flat = start
        stride = 1
        for coordinate, dimension in zip(reversed(prefix), reversed(shape[1:])):
            stride *= dimension
            flat += coordinate * stride
        item = ITEM_SIZES[dtype]
        offset = self._data_start + self._offsets[name] + flat * item
        window = self._map[offset:offset + count * item]
        if len(window) != count * item:
            raise ValueError(f'read window out of range for {name}')
        return window

    def read_run(self, name, prefix, start, count):
        shape = self._entries[name][1]
        self._check_run(name, shape, prefix, start, count)
        if self.mode == 'safetensors':
            return self._read_package(name, prefix, start, count)
        return self._read_bytes(name, shape, prefix, start, count)


def open_shards(root):
    names = [name for name in INPUT_PINS if name.endswith('.safetensors')]
    readers = {name: ShardReader(root / name) for name in names}
    if len({reader.mode for reader in readers.values()}) != 1:
        raise ValueError('shard readers disagree on the metadata mode')
    return readers


def module_spec(value):
    if not isinstance(value, dict) or set(value) - {'mode', 'bits', 'group_size'}:
        raise ValueError('invalid quantization settings')
    mode = value.get('mode', 'affine')
    bits, group = value.get('bits'), value.get('group_size')
    if mode != 'affine' or type(bits) is not int or bits not in (4, 8):
        raise ValueError('unexpected checkpoint quantization bits')
    if type(group) is not int or group != GROUP_SIZE:
        raise ValueError('unexpected checkpoint quantization group size')
    return {'mode': mode, 'bits': bits, 'group_size': group}


def resolved_settings(config):
    primary = config.get('quantization')
    fallback = config.get('quantization_config')
    if primary is not None and fallback is not None and primary != fallback:
        raise ValueError('conflicting quantization metadata')
    settings = primary if primary is not None else fallback
    if not isinstance(settings, dict) or 'quant_method' in settings or 'mode' not in settings:
        raise ValueError('invalid MLX quantization metadata')
    if module_spec({key: settings[key] for key in ('mode', 'bits', 'group_size')}) != DEFAULT_SPEC:
        raise ValueError('checkpoint default quantization differs from the pinned contract')
    overrides = {}
    for module, value in settings.items():
        if module in ('mode', 'bits', 'group_size'):
            continue
        if not MODULE_NAME.fullmatch(module):
            raise ValueError(f'invalid module name: {module}')
        overrides[module] = None if value is False else module_spec(value)
    return settings, overrides


def tensor_table(readers, index):
    if not isinstance(index, dict) or len(index) != INDEX_TENSORS:
        raise ValueError('index tensor count differs from the pinned contract')
    table = {}
    for name, shard in index.items():
        entry = readers[shard].entries().get(name)
        if entry is None:
            raise ValueError(f'indexed tensor is missing: {name}')
        table[name] = {'dtype': entry[0], 'shape': list(entry[1]), 'shard': shard}
    return table


def module_entry(table, module):
    payload = table[module + '.weight']
    shape = payload['shape']
    if len(shape) not in (2, 3) or any(d <= 0 for d in shape) or shape[-1] % WORDS_PER_GROUP:
        raise ValueError(f'invalid affine-int4 payload shape: {module}')
    group_shape = shape[:-1] + [shape[-1] // WORDS_PER_GROUP]
    for suffix in ('scales', 'biases'):
        side = table.get(f'{module}.{suffix}')
        if side is None or side['shape'] != group_shape or side['dtype'] != 'BF16':
            raise ValueError(f'invalid affine-int4 sidecar storage: {module}')
    return {'payload_shape': shape, 'group_shape': group_shape,
            'count': math.prod(group_shape), 'shard': payload['shard'],
            'scales_shard': table[module + '.scales']['shard'],
            'biases_shard': table[module + '.biases']['shard']}


def classify_modules(table, overrides):
    eligible = {}
    excluded = []
    for key in sorted(table):
        if not key.endswith('.weight') or table[key]['dtype'] != 'U32':
            continue
        module = key[:-len('.weight')]
        spec = overrides.get(module, DEFAULT_SPEC)
        if spec is None:
            raise ValueError(f'packed tensor declares unquantized storage: {module}')
        if spec['bits'] == 8:
            excluded.append(module)
            continue
        eligible[module] = module_entry(table, module)
    return eligible, sorted(excluded)


def require_inventory(eligible, excluded):
    rank3 = sum(len(info['payload_shape']) == 3 for info in eligible.values())
    if len(eligible) != ELIGIBLE_MODULES or rank3 != RANK3_MODULES or len(excluded) != INT8_MODULES:
        raise ValueError('real checkpoint inventory differs from the pinned contract')


def chain_index(name, ordinal, count):
    material = SEED.encode() + b'\0' + name.encode() + b'\0' + str(ordinal).encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], 'little') % count


def decode_coordinates(flat, group_shape):
    coordinates = []
    for size in reversed(group_shape):
        flat, coordinate = divmod(flat, size)
        coordinates.append(coordinate)
    return list(reversed(coordinates))


class Selector:
    def __init__(self, eligible, probe_bias):
        self.eligible = eligible
        self.probe_bias = probe_bias
        self.order = sorted(eligible)
        self.ordinals = dict.fromkeys(self.order, 0)
        self.used = {name: set() for name in self.order}
        self.picks = []

    def take_one(self, name, negative_only):
        info = self.eligible[name]
        for _ in range(CANDIDATE_LIMIT):
            flat = chain_index(name, self.ordinals[name], info['count'])
            self.ordinals[name] += 1
            if flat in self.used[name]:
                continue
            coordinates = decode_coordinates(flat, info['group_shape'])
            if negative_only and self.probe_bias(name, coordinates) >= 0.0:
                continue
            self.used[name].add(flat)
            return coordinates
        return None

    def take_phase(self, negative_only, target):
        taken = 0
        while taken < target:
            progressed = False
            for name in self.order:
                if taken >= target:
                    break
                coordinates = self.take_one(name, negative_only)
                if coordinates is None:
                    continue
                self.picks.append((name, coordinates))
                taken += 1
                progressed = True
            if not progressed:
                return taken
        return taken


def bias_probe(readers, eligible):
    def probe(name, coordinates):
        shard = eligible[name]['biases_shard']
        raw = readers[shard].read_run(name + '.biases', tuple(coordinates[:-1]), coordinates[-1], 1)
        return bf16_to_f32(int.from_bytes(raw, 'little'))
    return probe


def select_groups(eligible, probe):
    selector = Selector(eligible, probe)
    negatives = selector.take_phase(True, NEGATIVE_TARGET)
    if negatives == 0:
        raise ValueError('selection contains no negative bias')
    open_groups = selector.take_phase(False, GROUP_TARGET - negatives)
    if negatives + open_groups != GROUP_TARGET:
        raise ValueError('insufficient distinct groups for the pinned contract')
    return selector.picks, negatives


def reference_environment():
    import mlx.core as mx

    if mx.__version__ != MLX_VERSION:
        raise ValueError('mlx version differs from the pinned parity build')
    if not mx.metal.is_available():
        raise ValueError('pinned Metal device is unavailable')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(512 * 1024 * 1024)
    mx.set_cache_limit(64 * 1024 * 1024)
    return mx


def binary_hashes(mx):
    import glob

    root = Path(mx.__file__).parent
    hashes = {}
    for pattern in ('core.cpython-*.so', 'lib/libmlx.dylib'):
        for path in sorted(glob.glob(str(root / pattern))):
            hashes[Path(path).name] = sha256_hex(Path(path).read_bytes())
    core = [digest for name, digest in hashes.items() if name.startswith('core.')]
    if len(core) != 1 or core[0] != CORE_EXTENSION_SHA256:
        raise ValueError('mlx core extension differs from the pinned parity build')
    return hashes


def build_activations(mx):
    basis = [[1.0 if row == column else 0.0 for column in range(GROUP_SIZE)] for row in range(GROUP_SIZE)]
    dense = [[1.0, -1.0, 0.5, -0.25, 2.0, -2.0, 0.25, 0.125] * (GROUP_SIZE // WORDS_PER_GROUP)]
    return mx.array(basis, mx.float32), mx.array(dense, mx.float32)


def extract_group(readers, info, name, coordinates):
    prefix = tuple(coordinates[:-1])
    group = coordinates[-1]
    words_raw = readers[info['shard']].read_run(name + '.weight', prefix, group * WORDS_PER_GROUP, WORDS_PER_GROUP)
    scale_raw = readers[info['scales_shard']].read_run(name + '.scales', prefix, group, 1)
    bias_raw = readers[info['biases_shard']].read_run(name + '.biases', prefix, group, 1)
    words = [int.from_bytes(words_raw[4 * i:4 * i + 4], 'little') for i in range(WORDS_PER_GROUP)]
    return words, int.from_bytes(scale_raw, 'little'), int.from_bytes(bias_raw, 'little')


def check_reference_sanity(activation, parts, values):
    words, scale_bits, bias_bits = parts
    scale = bf16_to_f32(scale_bits)
    bias = bf16_to_f32(bias_bits)
    quantized = [((words[index // 8] >> (4 * (index % 8))) & 0xF) for index in range(GROUP_SIZE)]
    for row, got in zip(activation.tolist(), values):
        want = scale * sum(a * q for a, q in zip(row, quantized)) + bias * sum(row)
        if abs(got - want) > 1e-3 * max(1.0, abs(want)):
            raise ValueError(f'reference output diverges from group arithmetic: {got} != {want}')


def reference_values(mx, activation, parts):
    words, scale_bits, bias_bits = parts
    scale = bf16_to_f32(scale_bits)
    bias = bf16_to_f32(bias_bits)
    payload = mx.array(words, mx.uint32).reshape([1, WORDS_PER_GROUP])
    scales = mx.array([[scale]], mx.float32)
    biases = mx.array([[bias]], mx.float32)
    out = mx.quantized_matmul(activation, payload, scales, biases,
                              transpose=True, group_size=GROUP_SIZE, bits=4, mode='affine')
    mx.eval(out)
    mx.synchronize()
    values = out.astype(mx.float32).reshape(-1).tolist()
    if any(not math.isfinite(value) for value in values):
        raise ValueError('nonfinite reference output')
    check_reference_sanity(activation, parts, values)
    return values


def coordinates_of(coordinates):
    if len(coordinates) == 3:
        return {'expert': coordinates[0], 'row': coordinates[1], 'group': coordinates[2]}
    return {'row': coordinates[0], 'group': coordinates[1]}


def make_cell(number, pick, record):
    module, coordinates = pick
    info = record['info']
    words, scale_bits, bias_bits = record['parts']
    scale = bf16_to_f32(scale_bits)
    bias = bf16_to_f32(bias_bits)
    if not math.isfinite(scale) or not math.isfinite(bias):
        raise ValueError('nonfinite selected sidecar')
    return {
        'id': f'group-{number:04d}',
        'module': module,
        'payload_tensor': module + '.weight',
        'shard': info['shard'],
        'sidecar_shards': {'scales': info['scales_shard'], 'biases': info['biases_shard']},
        'rank': len(info['payload_shape']),
        'payload_shape': info['payload_shape'],
        'groups_per_row': info['group_shape'][-1],
        'coordinates': coordinates_of(coordinates),
        'negative_bias': bias < 0.0,
        'dtypes': {'payload': 'U32', 'scales': 'BF16', 'biases': 'BF16'},
        'weight': {'words': words},
        'scales': {'bf16_bits': scale_bits, 'f32': scale},
        'biases': {'bf16_bits': bias_bits, 'f32': bias},
        'expected': {'basis': {'shape': [GROUP_SIZE, 1], 'values': record['basis']},
                     'dense': {'shape': [1, 1], 'values': record['dense']}},
        'reference_op': 'mx.quantized_matmul(transpose=True, group_size=64, bits=4, mode="affine")',
    }


def run_extraction(mx, readers, eligible, picks):
    activations = build_activations(mx)
    cells = []
    for number, pick in enumerate(picks):
        module, coordinates = pick
        info = eligible[module]
        parts = extract_group(readers, info, module, coordinates)
        basis = reference_values(mx, activations[0], parts)
        dense = reference_values(mx, activations[1], parts)
        cells.append(make_cell(number, pick, {'info': info, 'parts': parts,
                                              'basis': basis, 'dense': dense}))
    entries = {name: activation_entry(array) for name, array in zip(('basis', 'dense'), activations)}
    return {'cells': cells, 'activations': entries}


def activation_entry(array):
    return {'shape': list(array.shape), 'dtype': 'f32', 'values': array.reshape(-1).tolist()}


def assemble_fixture(result, selection, provenance, quantization_block):
    cells = result['cells']
    negative = sum(cell['negative_bias'] for cell in cells)
    if negative == 0:
        raise ValueError('selection contains no negative bias')
    counts = {'groups': len(cells), 'negative_bias_groups': negative,
              'matmul_calls': 2 * len(cells), 'compared_values': (GROUP_SIZE + 1) * len(cells)}
    return {
        'schema': SCHEMA,
        'selection': selection,
        'provenance': provenance,
        'quantization': {'default': DEFAULT_SPEC, 'config': quantization_block},
        'tolerances': {'real-group': {'abs': 0.0, 'rel': 0.0}},
        'activations': result['activations'],
        'counts': counts,
        'cells': cells,
    }


def encode_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_exclusive(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os_fsync(stream.fileno())


def os_fsync(fileno):
    import os

    os.fsync(fileno)


def provenance_block(mx, mode):
    return {
        'mlx_python_version': mx.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'reader': mode,
        'binary_hashes': binary_hashes(mx),
    }


def prepare(args):
    root = args.checkpoint
    observed = verify_inputs(root)
    index = load_json(root / 'model.safetensors.index.json', 4 * 1024 * 1024)['weight_map']
    config = load_json(root / 'config.json', 1024 * 1024)
    settings, overrides = resolved_settings(config)
    readers = open_shards(root)
    mode = next(iter(readers.values())).mode
    table = tensor_table(readers, index)
    eligible, excluded = classify_modules(table, overrides)
    require_inventory(eligible, excluded)
    picks, negatives = select_groups(eligible, bias_probe(readers, eligible))
    mx = reference_environment()
    result = run_extraction(mx, readers, eligible, picks)
    if verify_inputs(root) != observed:
        raise ValueError('source identity changed during extraction')
    selection = {'algorithm': 'sha256-candidates-stratified-v1', 'seed': SEED,
                 'group_target': GROUP_TARGET, 'negative_target': NEGATIVE_TARGET,
                 'candidate_limit': CANDIDATE_LIMIT, 'phase_a_negative_groups': negatives}
    fixture = assemble_fixture(result, selection, provenance_block(mx, mode), settings)
    manifest = {
        'schema': 'p0a05-real-group-manifest-1',
        'inputs': observed,
        'identity_recheck': 'passed',
        'reader': mode,
        'selection': selection,
        'counts': fixture['counts'],
        'eligible_modules': sorted(eligible),
        'excluded_int8': excluded,
    }
    publish(args.output, fixture, manifest)


def publish(out_dir, fixture, manifest):
    out_dir.mkdir(parents=True)
    fixture_bytes = encode_json(fixture)
    if len(fixture_bytes) > 4 * 1024 * 1024:
        raise ValueError('fixture exceeds 4 MiB')
    write_exclusive(out_dir / 'real-groups.json', fixture_bytes)
    manifest['fixture'] = {'path': 'real-groups.json', 'bytes': len(fixture_bytes),
                           'sha256': sha256_hex(fixture_bytes)}
    manifest_bytes = encode_json(manifest)
    if len(manifest_bytes) > 4 * 1024 * 1024:
        raise ValueError('manifest exceeds 4 MiB')
    write_exclusive(out_dir / 'manifest.json', manifest_bytes)
    print(json.dumps({'status': 'prepared', 'schema': SCHEMA, 'qualified': False,
                      'fixture_sha256': manifest['fixture']['sha256'],
                      'manifest_sha256': sha256_hex(manifest_bytes),
                      'counts': manifest['counts']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('output already exists')
        prepare(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
