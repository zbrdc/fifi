#!/usr/bin/env python3
"""Capture tojson goldens from the pinned reference stack (decision 0029).

Runs on the Mac with the Edge0 virtual environment (transformers 5.17.0,
jinja2 3.1.6). CPU only. It reads the pinned model directory and writes
p1-tokenizer-tojson-goldens-1 JSON. Every expected string is the output of
transformers itself: `tokenizer.apply_chat_template` for the full renders and
`render_jinja_template` for the filter rows.

Usage: capture-tojson-goldens.py --output FILE [--model-dir DIR]
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

NASTY = "Weather for <city> & \"region\"; it's café ☕ — 日本\U0001F600 \x7f tab\there"

# Keys are deliberately not in alphabetical order, and the OpenAI wrapper is kept.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": NASTY,
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "unit": {"type": "string", "default": "celsius", "enum": ["celsius", "fahrenheit"]},
                    "city": {"type": "string", "description": "City, e.g. Zürich or São Paulo"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "parameters": {"type": "object", "properties": {}},
            "name": "noop",
            "description": "",
        },
    },
]

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Weather in Zürich?"},
    {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "get_weather",
                "arguments": {
                    "city": "Zürich",
                    "opts": {"zeta": 1, "alpha": [1, 2, {"b": True, "a": None}]},
                    "note": "it's <b>&</b> \x7f 日本",
                    "count": 3,
                    "ratio": 0.5,
                    "flag": False,
                },
            },
            {"id": "call_2", "name": "noop", "arguments": {"z": 1, "a": 2}},
        ],
    },
    {"role": "tool", "content": "18 C", "tool_call_id": "call_1"},
    {"role": "tool", "content": "ok", "tool_call_id": "call_2"},
]

RENDERS = [
    ("openai-tools-default", "default", MESSAGES[:2], TOOLS),
    ("openai-tools-history-default", "default", MESSAGES, TOOLS),
    ("openai-tools-history-off", "off", MESSAGES, TOOLS),
    ("history-without-tools", "default", MESSAGES, []),
]
THINKING_KWARGS = {"default": {}, "on": {"enable_thinking": True}, "off": {"enable_thinking": False}}

OBJ = {
    "zeta": 1,
    "alpha": {"y": [1, 2.5, None, True, False], "x": "café 日本 \U0001F600"},
    "text": "<&>' \"q\" \\ / \x7f   \u0085",
    "empty_list": [],
    "empty_map": {},
    "big": 12345678901234567890,
    "neg": -7,
}
CONTROLS = {"c": "".join(chr(i) for i in range(0x20)) + "end"}
UNI_KEYS = {"été": 1, "Z": 2, "a": 3, "日": 4}

FILTERS = [
    ("default", "{{ tools[0] | tojson }}", OBJ),
    ("controls", "{{ tools[0] | tojson }}", CONTROLS),
    ("controls-ensure-ascii", "{{ tools[0] | tojson(ensure_ascii=True) }}", CONTROLS),
    ("ensure-ascii-kw", "{{ tools[0] | tojson(ensure_ascii=True) }}", OBJ),
    ("ensure-ascii-positional", "{{ tools[0] | tojson(true) }}", OBJ),
    ("ensure-ascii-false-kw", "{{ tools[0] | tojson(ensure_ascii=False) }}", OBJ),
    ("sort-keys", "{{ tools[0] | tojson(sort_keys=True) }}", OBJ),
    ("sort-keys-unicode", "{{ tools[0] | tojson(sort_keys=True) }}", UNI_KEYS),
    ("unsorted-unicode-keys", "{{ tools[0] | tojson }}", UNI_KEYS),
    ("indent-2", "{{ tools[0] | tojson(indent=2) }}", OBJ),
    ("indent-0", "{{ tools[0] | tojson(indent=0) }}", OBJ),
    ("indent-tab", "{{ tools[0] | tojson(indent='\\t') }}", OBJ),
    ("indent-string", "{{ tools[0] | tojson(indent='--') }}", OBJ),
    ("indent-negative", "{{ tools[0] | tojson(indent=-2) }}", OBJ),
    ("indent-positional", "{{ tools[0] | tojson(false, 1) }}", OBJ),
    ("indent-none", "{{ tools[0] | tojson(indent=None) }}", OBJ),
    ("separators-compact", "{{ tools[0] | tojson(separators=(',', ':')) }}", OBJ),
    ("separators-custom", "{{ tools[0] | tojson(separators=(' ;', ' = ')) }}", OBJ),
    ("separators-with-indent", "{{ tools[0] | tojson(indent=2, separators=(',', ':')) }}", OBJ),
    ("separators-none", "{{ tools[0] | tojson(separators=None) }}", OBJ),
    ("combo", "{{ tools[0] | tojson(ensure_ascii=True, indent=1, sort_keys=True) }}", OBJ),
    ("all-positional", "{{ tools[0] | tojson(true, 2, (',', ': '), true) }}", OBJ),
    ("nested-empty-indent", "{{ tools[0] | tojson(indent=2) }}", {"a": [], "b": {}, "c": [[], {}]}),
    ("in-filter-chain", "{{ tools[0].alpha | tojson | length }}", OBJ),
    ("error-bad-separators", "{{ tools[0] | tojson(separators=1) }}", OBJ),
    ("error-bad-separators-length", "{{ tools[0] | tojson(separators=(',',)) }}", OBJ),
    ("error-bad-indent", "{{ tools[0] | tojson(indent=1.5) }}", OBJ),
    ("error-unknown-kwarg", "{{ tools[0] | tojson(pretty=True) }}", OBJ),
    ("error-too-many-args", "{{ tools[0] | tojson(false, None, None, false, 5) }}", OBJ),
    ("floats", "{{ tools[0] | tojson }}", {"a": [0.1, 1.0, 1e15, 1e16, 1e-4, 1e-5, 1e-7, 5e-324, 1.5e300, 1e30, -0.0, 2.5e-5, 1.7976931348623157e308, 0.30000000000000004, 100.0, 1e22, 123456789012345680.0, 1e21, 0.0001234, 123456789.123, 0.0]}),
    # 16 and 17 digit decimals: a parser without float_roundtrip reads many of them 1 ulp off.
    ("long-floats", "{{ tools[0] | tojson }}", {"a": [22707.319846318922, -95693268348.29051, -7442.733041219079, 26107.355549524742, -55534.990616406096, -38475.04162811635, 390.5098331794725, 88123.31515433741, 287746003547.011, 643774435254.5903, -900311238071.5613, 414689105006.1384, 9.998037984729436e-06, 9.202560400067246e-07, 2.6808819269098948e-14]}),
    ("indent-float-integral", "{{ tools[0] | tojson(indent=2.0) }}", OBJ),
    ("indent-true", "{{ tools[0] | tojson(indent=True) }}", {"a": [1]}),
    ("trim-chars-kw", "{{ 'xxhixx' | trim(chars='x') }}", OBJ),
    ("trim-unknown-kw", "{{ 'x' | trim(bogus='x') }}", OBJ),
    ("trim-duplicate", "{{ 'x' | trim('x', chars='x') }}", OBJ),
    # Rows with a fourth element record a known divergence of the engine.
    ("big-integer", "{{ tools[0] | tojson }}", {"n": 12345678901234567890123},
     "integers above u64::MAX print as floats"),
    ("string-bad-indent", "{{ 'x' | tojson(indent=1.5) }}", OBJ,
     "Python does not check indent for a top-level string"),
    ("string-bad-separators", "{{ 'x' | tojson(separators=1) }}", OBJ),
    ("list-display", "{{ [1, 'a', none, true] }}", OBJ,
     "a list prints with double quotes, Python prints single quotes"),
]


def file_record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def capture_renders(tokenizer):
    rows = []
    for name, thinking, messages, tools in RENDERS:
        reference = tokenizer.apply_chat_template(
            messages, tools=(tools or None), tokenize=False,
            add_generation_prompt=True, **THINKING_KWARGS[thinking])
        rows.append({
            "id": name,
            "thinking": thinking,
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "tools_json": json.dumps(tools, ensure_ascii=False),
            "reference": reference,
        })
    return rows


def capture_filters(source_unused):
    from transformers.utils.chat_template_utils import render_jinja_template

    rows = []
    for entry in FILTERS:
        name, template, value = entry[:3]
        row = {"id": name, "template": template, "x_json": json.dumps(value, ensure_ascii=False)}
        if len(entry) > 3:
            row["known_divergence"] = entry[3]
        try:
            out = render_jinja_template(
                [[{"role": "user", "content": "hi"}]], tools=[value], chat_template=template,
                add_generation_prompt=True)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, list):
                out = out[0]
            row["reference"] = out
        except Exception as error:  # noqa: BLE001 - Python raised; record the class only
            row["error"] = type(error).__name__
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/Users/dan/edge0/models/edge0-35b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    version = importlib.metadata.version
    goldens = {
        "schema": "p1-tokenizer-tojson-goldens-1",
        "reference": {
            "python": sys.version.split()[0],
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "jinja2": version("jinja2"),
            "model_dir": str(model_dir),
            "chat_template.jinja": file_record(model_dir / "chat_template.jinja"),
        },
        "renders": capture_renders(tokenizer),
        "filters": capture_filters(None),
    }
    Path(args.output).write_text(json.dumps(goldens, ensure_ascii=True, indent=1) + "\n")
    print("goldens written: %s renders=%d filters=%d" % (args.output, len(goldens["renders"]), len(goldens["filters"])))


if __name__ == "__main__":
    main()
