#!/usr/bin/env python3
"""OpenAI SDK 2.6.1 flows against a running enginesrv on the fake engine.

Usage: sdk_check.py ADDRESS KEY
The server serves two fake models. "demo" replies "Hello from the fake engine." to every prompt.
"toolbot" replies with a think block, a sentence, and a call of the function "add" with a=1 and b=2.
Each check prints PASS or FAIL. The exit code is 1 when a check fails.
"""
import json
import sys

import openai

REPLY = "Hello from the fake engine."
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS sdk {name}")
    else:
        print(f"FAIL sdk {name} {detail}")
        failures.append(name)


def main(address, key):
    client = openai.OpenAI(base_url=f"http://{address}/v1", api_key=key, max_retries=0, timeout=30)
    messages = [{"role": "user", "content": "hi"}]

    models = list(client.models.list())
    check("models", [m.id for m in models] == ["demo", "toolbot"], str([m.id for m in models]))

    reply = client.chat.completions.create(model="demo", messages=messages)
    check("chat content", reply.choices[0].message.content == REPLY, repr(reply.choices[0].message.content))
    check("chat finish", reply.choices[0].finish_reason == "stop")
    usage = reply.usage
    check("chat usage", usage is not None and usage.total_tokens == usage.prompt_tokens + usage.completion_tokens)

    stream = client.chat.completions.create(model="demo", messages=messages, stream=True)
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    check("stream text", text == REPLY, repr(text))
    check("stream role", chunks[0].choices[0].delta.role == "assistant")
    check("stream finish", chunks[-1].choices[0].finish_reason == "stop")
    check("stream usage", chunks[-1].usage is not None and chunks[-1].usage.completion_tokens > 0)

    limited = client.chat.completions.create(model="demo", messages=messages, max_tokens=3)
    check("max_tokens", limited.choices[0].finish_reason == "length" and limited.choices[0].message.content == REPLY[:3])

    tools = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]
    round_trip = client.chat.completions.create(
        model="demo",
        tools=tools,
        messages=[
            {"role": "user", "content": "add 1 and 2"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 2}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "3"},
        ],
    )
    check("tool round trip accepted", round_trip.choices[0].message.content == REPLY)

    for name, kwargs in [
        ("max_tokens 0", {"max_tokens": 0}),
        ("response_format json_object", {"response_format": {"type": "json_object"}}),
        ("tool_choice required", {"tools": tools, "tool_choice": "required"}),
    ]:
        try:
            client.chat.completions.create(model="demo", messages=messages, **kwargs)
            check(f"400 for {name}", False, "no error")
        except openai.BadRequestError as error:
            check(f"400 for {name}", error.status_code == 400)
    try:
        client.chat.completions.create(model="other", messages=messages)
        check("404 for an unknown model", False, "no error")
    except openai.NotFoundError as error:
        check("404 for an unknown model", error.status_code == 404)
    add_tool = {"type": "function", "function": {
        "name": "add",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}
    called = client.chat.completions.create(model="toolbot", tools=[add_tool], messages=messages)
    choice = called.choices[0]
    check("tool call finish", choice.finish_reason == "tool_calls")
    check("tool call content", choice.message.content == "Adding now.", repr(choice.message.content))
    check("tool call reasoning", getattr(choice.message, "reasoning_content", None) == "I need the sum.")
    calls = choice.message.tool_calls or []
    check("tool call shape", len(calls) == 1 and calls[0].function.name == "add"
          and json.loads(calls[0].function.arguments) == {"a": 1, "b": 2})

    pieces = list(client.chat.completions.create(model="toolbot", tools=[add_tool], messages=messages, stream=True))
    names, arguments, reasoning, text = {}, {}, "", ""
    for piece in pieces:
        if not piece.choices:
            continue
        delta = piece.choices[0].delta
        text += delta.content or ""
        reasoning += getattr(delta, "reasoning_content", None) or ""
        for entry in delta.tool_calls or []:
            if entry.function and entry.function.name:
                names[entry.index] = entry.function.name
            arguments[entry.index] = arguments.get(entry.index, "") + (entry.function.arguments or "")
    check("stream tool call", names == {0: "add"} and json.loads(arguments[0]) == {"a": 1, "b": 2}, str((names, arguments)))
    check("stream tool call text", text == "Adding now." and reasoning == "I need the sum.", repr((text, reasoning)))
    check("stream tool call finish", pieces[-1].choices[0].finish_reason == "tool_calls")

    wrong = openai.OpenAI(base_url=f"http://{address}/v1", api_key="wrong", max_retries=0, timeout=30)
    try:
        wrong.models.list()
        check("401 for a wrong key", False, "no error")
    except openai.AuthenticationError as error:
        check("401 for a wrong key", error.status_code == 401)


main(sys.argv[1], sys.argv[2])
sys.exit(1 if failures else 0)
