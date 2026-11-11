#!/usr/bin/env python3
"""Join cold request samples to server timing by request ID and compare two arms."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def records(text):
    result = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get('endpoint') != '/v1/chat/completions':
            continue
        request_id = row.get('request_id')
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('chat log has no request ID')
        if request_id in result:
            raise ValueError('duplicate chat request ID')
        result[request_id] = row
    return result


def positive_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def summarize(measurement, log_text, min_samples=3):
    samples = measurement.get('samples')
    if not isinstance(samples, list) or len(samples) < min_samples:
        raise ValueError('insufficient completed samples')
    by_id = records(log_text)
    ids = [s.get('request_id') for s in samples]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('sample IDs missing or duplicated')
    rows = []
    for sample, request_id in zip(samples, ids, strict=True):
        row = by_id.get(request_id)
        if row is None:
            raise ValueError('request missing from engine log')
        prompt, completion = sample.get('prompt_tokens'), sample.get('completion_tokens')
        if (type(prompt) is not int or prompt < 1 or type(completion) is not int or completion < 1
                or type(row.get('prompt_tokens')) is not int or row.get('prompt_tokens') != prompt
                or type(row.get('completion_tokens')) is not int or row.get('completion_tokens') != completion):
            raise ValueError('client and server token counts disagree')
        if (sample.get('reuse_class') != 'cold' or type(sample.get('reused_tokens')) is not int
                or sample.get('reused_tokens') != 0 or type(row.get('reused_tokens')) is not int
                or row.get('reused_tokens') != 0):
            raise ValueError('cache reuse is nonzero or unknown')
        if row.get('status') != 200 or row.get('outcome') != 'ok' or not positive_number(row.get('prefill_ms')):
            raise ValueError('server did not provide a successful finite timing')
        if not positive_number(sample.get('seconds')):
            raise ValueError('invalid client wall time')
        rows.append({'request_id': request_id, 'prompt_tokens': prompt, 'completion_tokens': completion,
                     'reused_tokens': 0, 'prefill_ms': row['prefill_ms'],
                     'started_to_first_token_prompt_tokens_per_second': prompt * 1000.0 / row['prefill_ms'],
                     'complete_reply_prompt_tokens_per_second': prompt / sample['seconds'],
                     'decode_tokens_per_second': row.get('decode_tokens_per_second')})
    if len({r['prompt_tokens'] for r in rows}) != 1:
        raise ValueError('prompt count changed within an arm')
    return {'samples': rows, 'median_started_to_first_token_prompt_tokens_per_second':
            statistics.median(r['started_to_first_token_prompt_tokens_per_second'] for r in rows),
            'median_complete_reply_prompt_tokens_per_second':
            statistics.median(r['complete_reply_prompt_tokens_per_second'] for r in rows)}


def compare(control, control_log, candidate, candidate_log, min_samples=3):
    digest = control.get('payload_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or digest != candidate.get('payload_sha256'):
        raise ValueError('payloads differ or have an invalid hash')
    a = summarize(control, control_log, min_samples)
    b = summarize(candidate, candidate_log, min_samples)
    if len(a['samples']) != len(b['samples']):
        raise ValueError('sample counts differ')
    if ([x['prompt_tokens'] for x in a['samples']] != [x['prompt_tokens'] for x in b['samples']]
            or [x['completion_tokens'] for x in a['samples']] != [x['completion_tokens'] for x in b['samples']]):
        raise ValueError('arm token counts differ')
    ca = a['median_started_to_first_token_prompt_tokens_per_second']
    cb = b['median_started_to_first_token_prompt_tokens_per_second']
    return {'payload_sha256': control['payload_sha256'], 'control': a, 'candidate': b,
            'started_to_first_token_speedup': cb / ca,
            'timing_note': 'Server timing runs from job start to first emitted batch and includes first-token generation; client timing runs to the complete reply.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'control-log', 'candidate', 'candidate-log', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--min-samples', type=int, default=3)
    args = parser.parse_args()
    if args.min_samples < 1:
        parser.error('--min-samples must be positive')
    result = compare(json.loads(args.control.read_text()), args.control_log.read_text(),
                     json.loads(args.candidate.read_text()), args.candidate_log.read_text(), args.min_samples)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
