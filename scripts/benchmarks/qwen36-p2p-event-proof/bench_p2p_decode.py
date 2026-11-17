#!/usr/bin/env python3
"""Measure server-reported decode throughput for a deterministic request."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request


def payload(model, repeat):
    if repeat < 0 or not model:
        raise ValueError('invalid model or repeat count')
    prompt = ('The sky is blue. ' * repeat +
              '\nCount from 1 to 200, one number per line, without skipping numbers.')
    data = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 128, 'temperature': 0, 'stream': False, 'x-thinking': 'off'}
    return (json.dumps(data) + '\n').encode()


def sample_of(reply, telemetry, elapsed):
    choice = reply['choices'][0]
    usage = reply['usage']
    content = choice['message'].get('content') or ''
    request_id = reply.get('id')
    rate = telemetry.get('decode_tokens_per_second')
    if (not isinstance(request_id, str) or not request_id or telemetry.get('request_id') != request_id or
            telemetry.get('outcome') != 'ok' or telemetry.get('status') != 200 or
            type(telemetry.get('reused_tokens')) is not int or telemetry['reused_tokens'] != 0 or
            type(usage.get('reused_tokens')) is not int or usage['reused_tokens'] != 0 or
            type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 1 or
            type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] < 64 or
            type(telemetry.get('prompt_tokens')) is not int or telemetry['prompt_tokens'] != usage['prompt_tokens'] or
            type(telemetry.get('completion_tokens')) is not int or telemetry['completion_tokens'] != usage['completion_tokens'] or
            not isinstance(content, str) or not content.strip().startswith('1\n2\n3') or
            isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 or
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0):
        raise ValueError('reply or server decode telemetry failed validation')
    return {'request_id': request_id, 'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'finish_reason': choice.get('finish_reason'),
            'answer_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'answer_prefix': content[:80],
            'time_to_first_token_ms': telemetry.get('time_to_first_token_ms'),
            'decode_tokens_per_second': rate, 'total_seconds': round(elapsed, 6)}


def telemetry_for(log_path, request_id):
    for line in reversed(log_path.read_text(errors='replace').splitlines()):
        if not line.startswith('{'):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get('request_id') == request_id:
            return entry
    return None


def self_test():
    reply = {'id': 'test', 'choices': [{'message': {'content': '1\n2\n3\n4'},
             'finish_reason': 'length'}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 128, 'reused_tokens': 0}}
    log = {'request_id': 'test', 'outcome': 'ok', 'status': 200, 'reused_tokens': 0,
           'prompt_tokens': 20, 'completion_tokens': 128,
           'time_to_first_token_ms': 500.0, 'decode_tokens_per_second': 25.0}
    assert sample_of(reply, log, 5)['decode_tokens_per_second'] == 25
    for bad in ({**log, 'completion_tokens': 127}, {**log, 'request_id': 'other'},
                {**log, 'decode_tokens_per_second': None}, {**log, 'reused_tokens': 3}):
        try:
            sample_of(reply, bad, 5)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid telemetry accepted')
    assert payload('fifi-q36', 0) == payload('fifi-q36', 0)
    print('decode measurement self-test passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--url', default='http://127.0.0.1:18792')
    ap.add_argument('--model', default='fifi-q36')
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--expected-context', type=int, default=131072)
    ap.add_argument('--timeout', type=float, default=300)
    ap.add_argument('--api-key-env', help='name of an environment variable holding the key')
    ap.add_argument('--server-log', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.repeat < 0 or args.samples < 1 or args.timeout <= 0 or not args.server_log or not args.output:
        ap.error('valid sample count, log, and output are required')
    if args.output.exists():
        ap.error('the output already exists')
    body = payload(args.model, args.repeat)
    headers = {'Content-Type': 'application/json'}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            ap.error('the requested API key environment variable is empty')
        headers['Authorization'] = 'Bearer ' + key
    base = args.url.rstrip('/')
    results = []
    for index in range(args.samples):
        with urllib.request.urlopen(urllib.request.Request(base + '/v1/system', headers=headers), timeout=5) as response:
            info = json.load(response)
        engine = next((e for e in info['engines'] if e['model'] == args.model), None)
        if engine is None or engine['state'] != 'ready' or engine['context_limit'] != args.expected_context:
            raise RuntimeError('engine is unavailable, busy, or has the wrong context')
        req = urllib.request.Request(base + '/v1/chat/completions', body, headers)
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            reply = json.load(response)
        elapsed = time.monotonic() - started
        deadline = time.monotonic() + 5
        entry = telemetry_for(args.server_log, reply.get('id'))
        while entry is None and time.monotonic() < deadline:
            time.sleep(0.1)
            entry = telemetry_for(args.server_log, reply.get('id'))
        if entry is None:
            raise RuntimeError('request telemetry was not found')
        sample = sample_of(reply, entry, elapsed)
        results.append(sample)
        print(f'sample {index + 1}/{args.samples}: '
              f'{sample["decode_tokens_per_second"]:.2f} decode tokens/s', flush=True)
    if len({(r['prompt_tokens'], r['completion_tokens'], r['answer_sha256'], r['finish_reason'])
            for r in results}) != 1:
        raise RuntimeError('deterministic reply or token counts changed within an arm')
    result = {'model': args.model, 'repeat': args.repeat,
              'payload_sha256': hashlib.sha256(body).hexdigest(), 'samples': results,
              'median_decode_tokens_per_second': statistics.median(
                  r['decode_tokens_per_second'] for r in results)}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
