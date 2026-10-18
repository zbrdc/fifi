"""Smoke check: GET /v1/models, then one short plain chat request. Prints one line and exits.

Exit status: 0 for HTTP 200 with a finish reason and usage, 2 otherwise, 1 for a usage error.
It is a structural check, not a speed claim.
"""

import argparse
import http.client
import json
import os
import sys

import agent_client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-env", default="ENGINESRV_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--out", help="write the reply summary to this JSON file")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    key = os.environ.get(args.key_env, "")
    if not key:
        print(f"the environment variable {args.key_env} is not set", file=sys.stderr)
        return 1
    try:
        status, models = agent_client.get_json(args.base_url, "/models", key, 60)
        data = (models or {}).get("data") or []
        if status != 200 or not data:
            print(f"smoke: GET /models status {status}, {len(data)} models")
            return 2
        reply = agent_client.chat(
            args.base_url, key,
            {"model": data[0]["id"], "messages": [{"role": "user", "content": "Say hi."}],
             "max_tokens": args.max_tokens, "temperature": 0.2, "x-thinking": "off"},
            args.timeout)
    except agent_client.ClientError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, http.client.HTTPException) as error:
        print(f"smoke: request failed: {error!r}")
        return 2
    message = reply.get("message") or {}
    usage = message.get("usage") or {}
    summary = {"status": reply.get("status"), "model": data[0]["id"],
               "finish_reason": message.get("finish_reason"), "usage": usage,
               "total_seconds": reply.get("total_seconds")}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2) + "\n")
    ok = (reply.get("status") == 200 and message.get("finish_reason")
          and isinstance(usage.get("prompt_tokens"), int))
    print(f"smoke: {'ok' if ok else 'failed'} {json.dumps(summary)}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
