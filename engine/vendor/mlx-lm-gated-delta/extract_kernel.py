#!/usr/bin/env python3
"""Extracts the Metal source of the mlx-lm gated delta kernel from the vendored Python file.

Usage: python3 -I engine/vendor/mlx-lm-gated-delta/extract_kernel.py [--check] [--has-mask] [--vectorized]

The Metal source in `gated_delta.py` is an f-string inside `_make_gated_delta_kernel`. This script
parses that file with `ast`, keeps the statements of the function up to the assignment of `source`,
drops the Metal availability test, and evaluates them with Python itself, so the f-string rules are
not reimplemented here. The result is the exact `source` argument that mlx-lm 0.31.0 passes to
`mx.fast.metal_kernel` for the chosen variant.

Without `--check` it writes the source to standard output. With `--check` it compares the source of
the scalar, unmasked variant (the default variant) with `gated_delta_step.metal` and with the hash in
`provenance.json`, and prints `ok` or `FAIL ...`.
"""
import ast
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE / "src" / "gated_delta.py"
DERIVED = HERE / "gated_delta_step.metal"


def source_of(has_mask, vectorized):
    tree = ast.parse(UPSTREAM.read_bytes().decode("utf-8"))
    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_make_gated_delta_kernel"
    )
    body = []
    for stmt in func.body:
        if isinstance(stmt, ast.If) and "metal" in ast.dump(stmt.test):
            continue  # the `mx.metal.is_available()` test
        body.append(stmt)
        if isinstance(stmt, ast.Assign) and stmt.targets[0].id == "source":
            break
    else:
        raise SystemExit("FAIL no assignment to `source` in _make_gated_delta_kernel")
    body.append(ast.Return(value=ast.Name(id="source", ctx=ast.Load())))
    probe = ast.FunctionDef(
        name="probe", args=func.args, body=body, decorator_list=[], returns=None,
        type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    scope = {}
    exec(compile(module, str(UPSTREAM), "exec"), scope)
    return scope["probe"](has_mask=has_mask, vectorized=vectorized)


def main(argv):
    args = set(argv)
    if "--check" in args:
        text = source_of(False, False).encode("utf-8")
        provenance = json.loads((HERE / "provenance.json").read_text())
        want = provenance["derived"]["gated_delta_step.metal"]["sha256"]
        if hashlib.sha256(text).hexdigest() != want:
            print("FAIL extracted source does not match the hash in provenance.json")
            return 1
        if DERIVED.read_bytes() != text:
            print("FAIL gated_delta_step.metal differs from the extracted source")
            return 1
        print("ok extract")
        return 0
    sys.stdout.write(source_of("--has-mask" in args, "--vectorized" in args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
