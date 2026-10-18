"""Probe nvfp4 device support against the pinned MLX build."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import mlx.core as mx


def binary_hashes() -> dict:
    root = Path(mx.__file__).parent
    hashes = {}
    for pattern in ('core.cpython-*.so', 'lib/libmlx.dylib'):
        for path in root.glob(pattern):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def probe(call) -> dict:
    try:
        result = call()
        return {'ok': True, 'result': result}
    except Exception as error:
        return {'ok': False, 'error': f'{type(error).__name__}: {error}'}


def device_name() -> dict:
    return probe(lambda: mx.metal.device_info())


def main() -> None:
    record = {
        'mlx_python_version': mx.__version__,
        'python_version': platform.python_version(),
        'binary_hashes': binary_hashes(),
        'device': device_name(),
    }
    mx.eval(mx.zeros((2, 2)) @ mx.zeros((2, 2)))
    record['metal_smoke'] = {'ok': True}

    weights = mx.array([0x11111111] * 4, mx.uint32).reshape(1, 4)
    scales = mx.array([127], mx.uint8).reshape(1, 1)
    activation = mx.ones((1, 16))
    record['nvfp4_matmul'] = probe(
        lambda: mx.quantized_matmul(
            activation,
            mx.array([0x11111111] * 2, mx.uint32).reshape(1, 2),
            mx.array([127], mx.uint8).reshape(1, 1),
            None,
            group_size=16,
            bits=4,
            mode='nvfp4',
        ).tolist()
    )
    record['nvfp4_dequantize'] = probe(
        lambda: mx.dequantize(
            mx.array([0x11111111] * 2, mx.uint32).reshape(1, 2),
            mx.array([127], mx.uint8).reshape(1, 1),
            None,
            group_size=16,
            bits=4,
            mode='nvfp4',
        ).tolist()
    )
    record['mxfp4_control'] = probe(
        lambda: mx.quantized_matmul(
            mx.ones((1, 32)), weights, scales, None, mode='mxfp4'
        ).tolist()
    )
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
