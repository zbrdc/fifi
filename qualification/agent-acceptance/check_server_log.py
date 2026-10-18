"""Post-check of the server request log for an ACC-A run whose server ran on another host.

It reads the copied `server.log` and the `result.json` of acc_a.py. It checks one log line per
request that the client sent, and HTTP 200 with outcome `ok` on every line. A failed check turns a
`pass` verdict into `pipeline_defect` in result.json.

Exit status: 0 ok, 2 pipeline defect, 1 usage error.
"""

import argparse
import json
import pathlib
import sys

import reqlog_check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", required=True)
    parser.add_argument("--result", required=True, help="result.json of acc_a.py (updated)")
    parser.add_argument("--key", default="finish_reason")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = json.loads(pathlib.Path(args.result).read_text(encoding="utf-8"))
        entries = reqlog_check.read_entries(args.log, args.key)
    except (OSError, ValueError) as error:
        print(f"cannot read the inputs: {error!r}", file=sys.stderr)
        return 1
    sent = result.get("requests_sent")
    problems = []
    if sent is None or len(entries) != sent:
        problems.append(f"{sent} requests sent, {len(entries)} log lines")
    if not reqlog_check.all_served_ok(entries):
        problems.append(reqlog_check.not_ok_detail(entries))
    result["server_log"] = {"lines": len(entries), "ok": not problems, "problems": problems}
    if problems and result.get("verdict") == "pass":
        result["verdict"] = "pipeline_defect"
    pathlib.Path(args.result).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("server log: " + ("ok" if not problems else "; ".join(problems)))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
