"""Prepares the pinned int8 MLX reference fixture of decision 0031.

Usage: python -B prepare.py --checkpoint DIR --output NEW_DIR
The output directory must not exist. The script builds it in a temporary
directory next to it and publishes it with one rename.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

LAYERS = 40
KINDS = ('mlp.gate', 'mlp.shared_expert_gate')
HIDDEN = 2048
GROUP = 64
BITS = 8
SEED_BASE = 31000
SHARD = 'int8-fixture-00001-of-00001.safetensors'
CORE_PIN = Path(__file__).with_name('reference-pin.json')
RECORD = Path(__file__).resolve().parents[2] / 'engine/testdata/mac-phase0a.json'
STORED = {'mlp.gate': ([256, 512], [256, 32]), 'mlp.shared_expert_gate': ([1, 512], [1, 32])}
NAMED = [('l0_router', 0, 'mlp.gate'), ('l0_shared', 0, 'mlp.shared_expert_gate'),
         ('l39_router', 39, 'mlp.gate'), ('l39_shared', 39, 'mlp.shared_expert_gate')]
KERNEL = {('mlp.gate', 1): 'affine_qmv_fast', ('mlp.shared_expert_gate', 1): 'affine_qmv',
          ('mlp.gate', 5): 'affine_qmv_wide', ('mlp.shared_expert_gate', 5): 'affine_qmv_wide',
          ('mlp.gate', 16): 'affine_qmm_splitk', ('mlp.shared_expert_gate', 16): 'affine_qmm_splitk'}


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'duplicate JSON field: {name}')
        result[name] = value
    return result


def load_json_bytes(data):
    return json.loads(data, object_pairs_hook=unique_object)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def module_name(layer, kind):
    return f'language_model.model.layers.{layer}.{kind}'


def case_plan():
    """Returns the 90 cases of decision 0031 in order."""
    cases = []
    for layer in range(LAYERS):
        for kind in KINDS:
            cases.append({'module': module_name(layer, kind), 'kind': kind, 'm': 1, 'signed_zero': False})
    for m in (5, 16):
        for _, layer, kind in NAMED:
            cases.append({'module': module_name(layer, kind), 'kind': kind, 'm': m, 'signed_zero': False})
    for layer, kind in ((0, 'mlp.gate'), (0, 'mlp.shared_expert_gate')):
        cases.append({'module': module_name(layer, kind), 'kind': kind, 'm': 1, 'signed_zero': True})
    for index, case in enumerate(cases):
        case['id'] = f'c{index:03d}'
        case['seed'] = SEED_BASE + index
        case['expected_kernel'] = KERNEL[(case['kind'], case['m'])]
    return cases


def pins():
    record = load_json_bytes(RECORD.read_bytes())
    files = [{key: item[key] for key in ('path', 'bytes', 'sha256')}
             for item in record['checkpoint']['files']
             if item['role'] == 'base_weight' or item['path'] in ('config.json', 'model.safetensors.index.json')]
    if len(files) != 6:
        raise ValueError('checkpoint pin list is not six files')
    return files


def check_pins(root, files):
    """Hashes every pinned file. Runs before any tensor read."""
    before = {}
    for item in files:
        path = root / item['path']
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'invalid source file: {item["path"]}')
        first = identity(path)
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'source length mismatch: {item["path"]}')
        if sha256_file(path) != item['sha256']:
            raise ValueError(f'source digest mismatch: {item["path"]}')
        if identity(path) != first:
            raise ValueError(f'source changed during hashing: {item["path"]}')
        before[item['path']] = first
    return before


def read_header(path):
    with path.open('rb') as stream:
        (length,) = struct.unpack('<Q', stream.read(8))
        if length > 16 << 20:
            raise ValueError('shard header too large')
        header = load_json_bytes(stream.read(length))
    header.pop('__metadata__', None)
    return 8 + length, header


def read_tensor(path, data_start, entry):
    start, end = entry['data_offsets']
    with path.open('rb') as stream:
        stream.seek(data_start + start)
        data = stream.read(end - start)
    if len(data) != end - start:
        raise ValueError('short tensor read')
    return data


def reference_environment():
    os.environ['MLX_DISABLE_COMPILE'] = '1'
    import importlib.metadata
    import mlx.core as mx
    pin = load_json_bytes(CORE_PIN.read_bytes())
    expected = {'numpy': '2.5.3', 'safetensors': '0.8.0', 'ml_dtypes': '0.5.3', 'mlx': pin['mlx_version']}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected or not mx.metal.is_available():
        raise ValueError('reference environment does not match the pinned Metal build')
    installed = Path(mx.__file__).parent
    for name, digest in pin['asserted_files'].items():
        if sha256_file(installed / name) != digest:
            raise ValueError(f'reference binary digest mismatch: {name}')
    recorded = {name: sha256_file(installed / name) for name in pin['recorded_files']}
    mx.set_default_device(mx.gpu)
    try:
        raw = mx.metal.device_info()
        info = {key: str(raw[key]) for key in ('architecture', 'device_name')}
    except Exception as error:
        raise ValueError(f'device record read failed: {error!r}') from error
    if not all(info.values()):
        raise ValueError('device record is empty')
    return mx, hashlib.sha256(CORE_PIN.read_bytes()).hexdigest(), actual, recorded, info


def cpu_brand():
    try:
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def dequant_float64(np, payload, scales, biases, x):
    """Independent float64 dequantized matmul. payload [N,K/4] u32, sidecars [N,K/64]."""
    n = payload.shape[0]
    q = payload.astype('<u4').view(np.uint8).reshape(n, -1).astype(np.float64)
    k = q.shape[1]
    s = np.repeat(scales.astype(np.float64), GROUP, axis=1)
    b = np.repeat(biases.astype(np.float64), GROUP, axis=1)
    if s.shape[1] != k:
        raise ValueError('sidecar width does not match payload')
    return x.astype(np.float64) @ (q * s + b).T


def build(args, work):
    import ml_dtypes
    import numpy as np
    from safetensors.numpy import save_file
    files = pins()
    before = check_pins(args.checkpoint, files)
    config = load_json_bytes((args.checkpoint / 'config.json').read_bytes())
    index = load_json_bytes((args.checkpoint / 'model.safetensors.index.json').read_bytes())['weight_map']
    settings = config.get('quantization')
    if settings is None or settings.get('mode') != 'affine':
        raise ValueError('config has no affine quantization block')
    if 'quantization_config' in config and config['quantization_config'] != settings:
        raise ValueError('conflicting quantization metadata')
    plan = case_plan()
    modules = sorted({c['module'] for c in plan})
    if len(modules) != 80:
        raise ValueError('module list is not 80')
    overrides = {}
    for module in modules:
        value = settings.get(module)
        if value != {'group_size': GROUP, 'bits': BITS} and value != {'bits': BITS, 'group_size': GROUP}:
            raise ValueError(f'unexpected quantization override for {module}')
        overrides[module] = {'bits': BITS, 'group_size': GROUP}
    derived = {'quantization': {'mode': 'affine', 'bits': settings['bits'], 'group_size': settings['group_size'], **overrides}}
    mx, reference_pin, versions, recorded, device = reference_environment()

    tensors, sources, headers = {}, [], {}
    for module in modules:
        kind = module.split('.', 4)[4]
        for role in ('weight', 'scales', 'biases'):
            key = f'{module}.{role}'
            shard = index.get(key)
            if shard is None:
                raise ValueError(f'tensor not indexed: {key}')
            if shard not in headers:
                headers[shard] = read_header(args.checkpoint / shard)
            start, header = headers[shard]
            entry = header[key]
            want_shape = STORED[kind][0] if role == 'weight' else STORED[kind][1]
            want_dtype = 'U32' if role == 'weight' else 'BF16'
            if entry['dtype'] != want_dtype or entry['shape'] != want_shape:
                raise ValueError(f'unexpected tensor layout: {key}')
            data = read_tensor(args.checkpoint / shard, start, entry)
            raw = np.frombuffer(data, dtype='<u4' if role == 'weight' else '<u2').reshape(want_shape)
            if role != 'weight':
                if not np.isfinite(raw.view(ml_dtypes.bfloat16).astype(np.float32)).all():
                    raise ValueError(f'nonfinite sidecar: {key}')
                raw = raw.view(ml_dtypes.bfloat16)
            tensors[key] = raw.copy()
            sources.append({'tensor': key, 'shard': shard, 'data_offsets': entry['data_offsets'],
                            'sha256': hashlib.sha256(data).hexdigest()})
    if sum(1 for _ in tensors) != 240:
        raise ValueError('tensor count is not 240')
    shard_path = work / SHARD
    save_file(tensors, str(shard_path))
    (work / 'config.json').write_bytes((json.dumps(derived, indent=2, sort_keys=True) + '\n').encode())
    (work / 'model.safetensors.index.json').write_bytes(
        (json.dumps({'metadata': {}, 'weight_map': {key: SHARD for key in sorted(tensors)}}, indent=2) + '\n').encode())
    loaded = mx.load(str(shard_path))

    cases_out, case_records, worst = {}, [], 0.0
    for case in plan:
        module = case['module']
        rng = np.random.Generator(np.random.PCG64(case['seed']))
        x = rng.standard_normal((case['m'], HIDDEN)).astype(np.float32)
        if case['signed_zero']:
            x[0, 0::4] = 0.0
            x[0, 1::4] = -0.0
        w, s, b = (loaded[f'{module}.{r}'] for r in ('weight', 'scales', 'biases'))
        result = mx.quantized_matmul(mx.array(x), w, s.astype(mx.float32), b.astype(mx.float32),
                                     transpose=True, group_size=GROUP, bits=BITS, mode='affine')
        mx.eval(result)
        mx.synchronize()
        y = np.array(result)
        n = STORED[case['kind']][0][0]
        if y.dtype != np.float32 or y.shape != (case['m'], n) or not np.isfinite(y).all():
            raise ValueError('invalid reference output')
        check = dequant_float64(np, tensors[f'{module}.weight'], tensors[f'{module}.scales'].astype(np.float32),
                                tensors[f'{module}.biases'].astype(np.float32), x)
        scale = max(1.0, float(np.abs(check).max()))
        diff = float(np.abs(check - y.astype(np.float64)).max())
        if diff > 1e-3 * scale:
            raise ValueError(f'float64 structural check failed for {case["id"]}: {diff}')
        worst = max(worst, diff / scale)
        cases_out['act.' + case['id']] = x
        cases_out['ref.' + case['id']] = y
        case_records.append({**{k: case[k] for k in ('id', 'module', 'm', 'seed', 'signed_zero', 'expected_kernel')},
                             'bits': BITS, 'group_size': GROUP, 'mode': 'affine',
                             'activation': 'act.' + case['id'], 'reference': 'ref.' + case['id'],
                             'float64_max_abs_diff': diff})
    save_file(cases_out, str(work / 'cases.safetensors'))

    def asset(name):
        path = work / name
        return {'path': name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}

    def inventory(name):
        start, header = read_header(work / name)
        out = []
        for key in sorted(header):
            data = read_tensor(work / name, start, header[key])
            out.append({'name': key, 'dtype': header[key]['dtype'], 'shape': header[key]['shape'],
                        'sha256': hashlib.sha256(data).hexdigest()})
        return out

    for item in files:
        if identity(args.checkpoint / item['path']) != before[item['path']]:
            raise ValueError(f'source identity changed during extraction: {item["path"]}')
    manifest = {
        'schema': 'int8-reference/1', 'synthetic': False, 'layer_count': LAYERS,
        'reference_pin': reference_pin, 'versions': versions, 'recorded_binaries': recorded,
        'build_flags': load_json_bytes(CORE_PIN.read_bytes())['build_flags'],
        'env': {'MLX_DISABLE_COMPILE': '1'},
        'device': {'cpu_brand': cpu_brand(), 'platform': platform.platform(), **device},
        'source_files': files, 'sources': sources,
        'assets': {'config': asset('config.json'), 'index': asset('model.safetensors.index.json'),
                   'shard': asset(SHARD), 'cases': asset('cases.safetensors')},
        'shard_tensors': inventory(SHARD), 'case_tensors': inventory('cases.safetensors'),
        'cases': case_records, 'float64_worst_relative_diff': worst,
    }
    data = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    (work / 'manifest.json').write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(case_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = None
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('output already exists')
        work = Path(os.environ.get('TMPDIR', args.output.parent)) / f'.int8-prepare-{os.getpid()}'
        work.mkdir()
        digest, count = build(args, work)
        if args.output.exists():
            raise ValueError('output appeared during preparation')
        os.rename(work, args.output)
        work = None
        print(json.dumps({'status': 'prepared', 'manifest_sha256': digest, 'case_count': count}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
