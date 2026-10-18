import argparse
import hashlib
import json
import os
import math
import platform
import re
from pathlib import Path
import sys


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'duplicate JSON field: {name}')
        result[name] = value
    return result


def load_json(path):
    with path.open('rb') as stream:
        data = stream.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('JSON exceeds 4 MiB')
    return json.loads(data, object_pairs_hook=unique_object)


def source_identity(root, files, fixture):
    identities = {}
    for item in files:
        name = item['path']
        if Path(name).name != name or name in ('', '.', '..') or name in identities:
            raise ValueError('invalid or duplicate source path')
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'invalid source file: {name}')
        before = path.stat()
        limit = 1024 * 1024 if fixture else 6 * 1024**3
        if before.st_size != item['bytes'] or before.st_size > limit:
            raise ValueError(f'source length mismatch: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError(f'source changed while opening: {name}')
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        after = path.stat()
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError(f'source changed during hashing: {name}')
        if digest.hexdigest() != item['sha256']:
            raise ValueError(f'source digest mismatch: {name}')
        identities[name] = identity(after)
    return identities


def resolved_settings(config):
    primary = config.get('quantization')
    fallback = config.get('quantization_config')
    if primary is not None and fallback is not None and primary != fallback:
        raise ValueError('conflicting quantization metadata')
    settings = primary if primary is not None else fallback
    if not isinstance(settings, dict) or 'quant_method' in settings:
        raise ValueError('invalid MLX quantization metadata')

    def spec(value):
        if not isinstance(value, dict) or set(value) - {'mode', 'bits', 'group_size'}:
            raise ValueError('invalid quantization settings')
        mode = value.get('mode', 'affine')
        bits, group = value.get('bits'), value.get('group_size')
        if mode != 'affine' or type(bits) is not int or bits not in (4, 8) or type(group) is not int or group != 64:
            raise ValueError('unexpected checkpoint quantization settings')
        return mode, bits, group

    if 'mode' not in settings:
        raise ValueError('top-level quantization mode is required')
    defaults = spec({key: value for key, value in settings.items() if key in ('mode', 'bits', 'group_size')})
    overrides = {}
    for module, value in settings.items():
        if module in ('mode', 'bits', 'group_size'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', module):
            raise ValueError('invalid module name')
        overrides[module] = None if value is False else spec(value)
    return defaults, overrides


def inventory(root, pins):
    from safetensors import safe_open
    index = load_json(root/'model.safetensors.index.json')['weight_map']
    defaults, overrides = resolved_settings(load_json(root/'config.json'))
    files = {item['path'] for item in pins['files']}
    if not isinstance(index, dict) or not index:
        raise ValueError('empty or invalid checkpoint index')
    if any(not isinstance(name, str) or name not in files or not name.endswith('.safetensors') for name in index.values()):
        raise ValueError('unapproved indexed file')
    if set(index.values()) != {name for name in files if name.endswith('.safetensors')}:
        raise ValueError('indexed shard set differs from source pins')
    tensors = {}
    for name in sorted(set(index.values())):
        with safe_open(str(root/name), framework='numpy') as source:
            for key in source.keys():
                if index.get(key) != name or key in tensors:
                    raise ValueError('index does not match stored tensors')
                view = source.get_slice(key)
                tensors[key] = {'shape': view.get_shape(), 'dtype': view.get_dtype(), 'shard': name}
    if set(tensors) != set(index):
        raise ValueError('indexed tensor is missing')
    eligible, excluded = {}, []
    for key, payload in tensors.items():
        if not key.endswith('.weight') or payload['dtype'] != 'U32':
            continue
        module = key[:-7]
        setting = overrides.get(module, defaults)
        if setting is None:
            raise ValueError('packed tensor declares unquantized storage')
        if setting[1] == 8:
            excluded.append(module)
            continue
        shape = payload['shape']
        scales, biases = tensors.get(module+'.scales'), tensors.get(module+'.biases')
        if len(shape) not in (2, 3) or any(d <= 0 for d in shape) or shape[-1] % 8:
            raise ValueError('invalid affine-int4 payload shape')
        expected = shape[:-1] + [shape[-1]//8]
        if any(side is None or side['shape'] != expected or side['dtype'] != 'BF16' for side in (scales, biases)):
            raise ValueError('invalid affine-int4 sidecar storage')
        count = math.prod(expected)
        if count > 2**64-1:
            raise ValueError('group count exceeds u64')
        eligible[module] = {'payload': payload, 'scales': scales, 'biases': biases, 'group_shape': expected, 'count': count}
    return eligible, sorted(excluded)


def select_groups(modules, count):
    if type(count) is not int or not 1 <= count <= 1000 or sum(item['count'] for item in modules.values()) < count:
        raise ValueError('insufficient groups or invalid group count')
    if count < len(modules):
        raise ValueError('group count must cover eligible modules')
    ordinals = {name: 0 for name in modules}
    selected = {name: set() for name in modules}
    result = []
    while len(result) < count:
        for name in sorted(modules):
            if len(selected[name]) == modules[name]['count']:
                continue
            for _ in range(1024):
                digest = hashlib.sha256((name+'\0'+str(ordinals[name])).encode()).digest()
                ordinals[name] += 1
                index = int.from_bytes(digest[:8], 'little') % modules[name]['count']
                if index not in selected[name]:
                    selected[name].add(index)
                    coordinates = []
                    for size in reversed(modules[name]['group_shape']):
                        index, coordinate = divmod(index, size)
                        coordinates.append(coordinate)
                    result.append((name, list(reversed(coordinates))))
                    break
            else:
                raise ValueError('selection candidate limit exceeded')
            if len(result) == count:
                break
    return result


def reference_environment():
    import importlib.metadata
    import mlx.core as mx
    pin_path = Path(__file__).with_name('reference-pin.json')
    pin = load_json(pin_path)
    expected = {'numpy': '2.5.3', 'safetensors': '0.8.0', 'ml_dtypes': '0.5.3', 'mlx': pin['mlx_version']}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected or not mx.metal.is_available():
        raise ValueError('reference environment does not match pinned Metal build')
    installed = Path(mx.__file__).parent
    for name, digest in pin['files'].items():
        if hashlib.sha256((installed/name).read_bytes()).hexdigest() != digest:
            raise ValueError('reference binary digest mismatch')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(512*1024*1024)
    mx.set_cache_limit(64*1024*1024)
    return mx, hashlib.sha256(pin_path.read_bytes()).hexdigest(), actual


def asset(path):
    data = path.read_bytes()
    if len(data) > 64*1024:
        raise ValueError('prepared tensor exceeds 64 KiB')
    return {'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def prepare(args):
    if args.fixture:
        pins = load_json(args.checkpoint/'fixture-source.json')
    else:
        record = load_json(Path(__file__).resolve().parents[2]/'engine/testdata/mac-phase0a.json')
        pins = {'files': [item for item in record['checkpoint']['files']
                          if item['role'] == 'base_weight' or item['path'] in ('config.json', 'model.safetensors.index.json')],
                'group_count': 1000}
    before = source_identity(args.checkpoint, pins['files'], args.fixture)
    modules, excluded = inventory(args.checkpoint, pins)
    if not args.fixture and (len(modules) != 432 or len(excluded) != 80 or sum(len(v['payload']['shape']) == 3 for v in modules.values()) != 120):
        raise ValueError('real checkpoint inventory differs from contract')
    selected = select_groups(modules, pins['group_count'])
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
    mx, reference_pin, versions = reference_environment()
    args.output.mkdir()
    activations = {'basis': np.eye(64, dtype=np.float32),
                   'dense': np.array([[1, -1, .5, -.25, 2, -2, .25, .125]*8], dtype=np.float32)}
    save_file(activations, str(args.output/'activations.safetensors'))
    groups = []
    for number, (module, coordinates) in enumerate(selected):
        tensors = {}
        for role, suffix, stride in [('weight', 'payload', 8), ('scales', 'scales', 1), ('biases', 'biases', 1)]:
            info = modules[module][suffix]
            selection = tuple(slice(c, c+1) for c in coordinates[:-1]) + (slice(coordinates[-1]*stride, (coordinates[-1]+1)*stride),)
            with safe_open(str(args.checkpoint/info['shard']), framework='numpy') as source:
                tensors[role] = source.get_slice(module+'.'+role)[selection].reshape(1, stride).copy()
        if not all(np.isfinite(tensors[key].astype(np.float32)).all() for key in ('scales', 'biases')):
            raise ValueError('nonfinite selected sidecar')
        weight_path = args.output/f'group-{number:04d}.safetensors'
        save_file(tensors, str(weight_path))
        loaded = mx.load(str(weight_path))
        expected = {}
        for probe, activation in activations.items():
            result = mx.quantized_matmul(mx.array(activation), loaded['weight'], loaded['scales'].astype(mx.float32), loaded['biases'].astype(mx.float32), transpose=True, group_size=64, bits=4, mode='affine')
            mx.eval(result)
            mx.synchronize()
            expected[probe] = np.array(result)
            if expected[probe].dtype != np.float32 or not np.isfinite(expected[probe]).all():
                raise ValueError('invalid reference output')
        reference_path = args.output/f'reference-{number:04d}.safetensors'
        save_file(expected, str(reference_path))
        groups.append({'module': module, 'coordinates': coordinates, 'payload_shape': modules[module]['payload']['shape'],
                       'weight': asset(weight_path), 'reference': asset(reference_path),
                       'shards': [modules[module][key]['shard'] for key in ('payload', 'scales', 'biases')],
                       'negative_bias': float(tensors['biases'][0, 0]) < 0})
    negative = sum(group['negative_bias'] for group in groups)
    if negative == 0:
        raise ValueError('selection contains no negative bias')
    if source_identity(args.checkpoint, pins['files'], args.fixture) != before:
        raise ValueError('source identity changed during extraction')
    (args.output/'config.json').write_bytes((args.checkpoint/'config.json').read_bytes())
    manifest = {'version': 1, 'fixture': args.fixture, 'group_count': len(groups), 'negative_biases': negative,
                'eligible_modules': sorted(modules), 'excluded_int8': excluded, 'reference_pin': reference_pin,
                'source_files': [{key: item[key] for key in ('path', 'bytes', 'sha256')} for item in pins['files']],
                'config': asset(args.output/'config.json'), 'activations': asset(args.output/'activations.safetensors'),
                'groups': groups, 'versions': versions, 'hardware': platform.platform()}
    if sum(item.stat().st_size for item in args.output.iterdir()) > 32*1024*1024:
        raise ValueError('prepared artifacts exceed 32 MiB')
    data = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    if len(data) > 4*1024*1024:
        raise ValueError('manifest exceeds 4 MiB')
    temporary = args.output/'manifest.pending'
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, args.output/'manifest.json')
    temporary.unlink()
    print(json.dumps({'status': 'prepared', 'fixture': args.fixture, 'qualified': False,
                      'manifest_sha256': hashlib.sha256(data).hexdigest(), 'group_count': len(groups)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
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
