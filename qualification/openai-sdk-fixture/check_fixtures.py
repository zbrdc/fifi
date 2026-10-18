#!/usr/bin/env python3
"""Offline check of recorded payloads against the pinned OpenAI SDK models.

Usage: check_fixtures.py [--write-expected]
Also validates contract_*.json against api/openapi.yaml.
Exit 0 only when the output equals expected_results.json.
No network access: the client path uses an httpx MockTransport.
"""
import copy, json, pathlib, sys
import httpx, jsonschema, openai, yaml
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Model

HERE = pathlib.Path(__file__).parent
SPEC = HERE.parent.parent / "api" / "openapi.yaml"
FIX = HERE / "fixtures"
load = lambda n: json.loads((FIX / n).read_text())


def strict_ok(cls, payload):
    try:
        cls.model_validate(payload)
        return None
    except Exception as e:  # pydantic ValidationError
        return sorted({".".join(map(str, x["loc"])) + ":" + x["type"] for x in e.errors()})


def required_declared(cls):
    return sorted(k for k, f in cls.model_fields.items() if f.is_required())


def ablate(cls, payload, paths):
    """Return the paths whose removal makes strict validation fail."""
    needed = []
    for p in paths:
        d = copy.deepcopy(payload)
        t = d
        for k in p[:-1]:
            t = t[k]
        if p[-1] not in t:
            continue
        del t[p[-1]]
        if strict_ok(cls, d) is not None:
            needed.append(".".join(map(str, p)))
    return sorted(needed)


def paths_of(payload, prefix=()):
    out = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            out.append(prefix + (k,))
            out += paths_of(v, prefix + (k,))
    elif isinstance(payload, list) and payload:
        out += paths_of(payload[0], prefix + (0,))
    return out


def client(handler):
    return openai.OpenAI(api_key="k", base_url="http://offline.invalid/v1", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def sse(chunks, done=True):
    body = "".join("data: %s\n\n" % json.dumps(c) for c in chunks)
    return body + ("data: [DONE]\n\n" if done else "")


def try_client(fn):
    try:
        return fn()
    except Exception as e:
        return "ERROR:" + type(e).__name__


def client_paths(prefix):
    res = {}
    comp = load(prefix + "chat_completion.json")
    c = client(lambda r: httpx.Response(200, json=comp))
    def nonstream():
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        return {"content": r.choices[0].message.content, "finish": r.choices[0].finish_reason,
                "object": r.object, "created": r.created, "model": r.model,
                "usage.total_tokens": r.usage.total_tokens}
    res["chat_nonstream"] = try_client(nonstream)
    chunks = load(prefix + "chunks.json")
    c = client(lambda r: httpx.Response(200, content=sse(chunks), headers={"content-type": "text/event-stream"}))
    def stream():
        s = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
        text, fin, usage, n = "", None, None, 0
        for ch in s:
            n += 1
            if ch.choices:
                text += ch.choices[0].delta.content or ""
                fin = ch.choices[0].finish_reason or fin
            usage = ch.usage or usage
        return {"events": n, "text": text, "finish": fin, "usage": usage.completion_tokens if usage else None}
    res["chat_stream"] = try_client(stream)
    models = load(prefix + "models.json")
    c = client(lambda r: httpx.Response(200, json=models))
    res["models_list"] = try_client(lambda: [m.id for m in c.models.list()])
    return res


def contract_schema_errors():
    """Validate each contract_*.json fixture against its api/openapi.yaml schema."""
    spec = yaml.safe_load(SPEC.read_text())
    comps = spec["components"]
    listing = (spec["paths"]["/v1/models"]["get"]["responses"]["200"]["content"]
               ["application/json"]["schema"])

    def errors(schema, payload):
        schema = dict(schema, components=comps)
        v = jsonschema.Draft202012Validator(schema)
        return sorted("/".join(map(str, e.absolute_path)) + ":" + e.message for e in v.iter_errors(payload))

    ref = lambda n: {"$ref": "#/components/schemas/" + n}
    return {
        "contract_chat_completion": errors(ref("ChatCompletion"), load("contract_chat_completion.json")),
        "contract_chunks": [errors(ref("ChatCompletionChunk"), c) for c in load("contract_chunks.json")],
        "contract_models": errors(listing, load("contract_models.json")),
    }


def main():
    out = {"openai_version": openai.__version__}
    out["declared_required"] = {n: required_declared(c) for n, c in [
        ("ChatCompletion", ChatCompletion), ("ChatCompletionChunk", ChatCompletionChunk), ("Model", Model)]}
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as CChoice, ChoiceDelta, ChoiceDeltaToolCall
    from openai.types.chat import ChatCompletionMessage
    from openai.types import CompletionUsage
    out["declared_required"].update({
        "ChatCompletion.Choice": required_declared(Choice),
        "ChatCompletionMessage": required_declared(ChatCompletionMessage),
        "CompletionUsage": required_declared(CompletionUsage),
        "Chunk.Choice": required_declared(CChoice),
        "Chunk.ChoiceDelta": required_declared(ChoiceDelta),
        "ChoiceDeltaToolCall": required_declared(ChoiceDeltaToolCall)})
    strict = {}
    for pre in ("contract_", "proposed_"):
        p = load(pre + "chat_completion.json")
        strict[pre + "chat_completion"] = strict_ok(ChatCompletion, p)
        strict[pre + "chunks"] = [strict_ok(ChatCompletionChunk, c) for c in load(pre + "chunks.json")]
        m = load(pre + "models.json")
        strict[pre + "models.data[0]"] = strict_ok(Model, m["data"][0])
    strict["proposed_chat_completion_tool_calls"] = strict_ok(ChatCompletion, load("proposed_chat_completion_tool_calls.json"))
    strict["proposed_chunks_tool_calls"] = [strict_ok(ChatCompletionChunk, c) for c in load("proposed_chunks_tool_calls.json")]
    out["strict_validation_errors"] = strict
    out["contract_schema_errors"] = contract_schema_errors()
    p = load("proposed_chat_completion.json")
    out["required_by_ablation"] = {
        "ChatCompletion": ablate(ChatCompletion, p, paths_of(p)),
        "ChatCompletionChunk(final)": ablate(ChatCompletionChunk, load("proposed_chunks.json")[-1],
                                              paths_of(load("proposed_chunks.json")[-1])),
        "ChatCompletionChunk(delta)": ablate(ChatCompletionChunk, load("proposed_chunks.json")[1],
                                              paths_of(load("proposed_chunks.json")[1])),
        "ChatCompletionChunk(tool_calls)": ablate(ChatCompletionChunk, load("proposed_chunks_tool_calls.json")[1],
                                                   paths_of(load("proposed_chunks_tool_calls.json")[1])),
        "Model": ablate(Model, load("proposed_models.json")["data"][0], paths_of(load("proposed_models.json")["data"][0])),
    }
    out["client_path"] = {"contract": client_paths("contract_"), "proposed": client_paths("proposed_")}
    def flat(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from flat(v)
        elif isinstance(x, list):
            for v in x:
                yield from flat(v)
        elif x:
            yield x
    schema_errors = list(flat(out["contract_schema_errors"]))
    if schema_errors:
        print("contract schema errors: " + json.dumps(schema_errors, indent=2), file=sys.stderr)
        return 1
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    exp = HERE / "expected_results.json"
    if "--write-expected" in sys.argv:
        exp.write_text(text)
        print(text)
        return 0
    print(text)
    if exp.read_text() != text:
        print("MISMATCH with expected_results.json", file=sys.stderr)
        return 1
    print("OK: matches expected_results.json")
    return 0


sys.exit(main())
