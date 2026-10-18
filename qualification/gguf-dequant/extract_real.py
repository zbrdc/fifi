#!/usr/bin/env python3
"""Read-only extraction of a few blocks per tensor type from a GGUF shard.

Usage: extract_real.py <shard.gguf> <out-dir> [--list]
Writes <out-dir>/real-<type>.blocks and <out-dir>/real-index.json.
The shard is only read. It is never modified.
"""
import json, os, struct, sys

# GGUF type id -> (name, block_elems, block_bytes)
TYPES = {0: ("f32", 1, 4), 1: ("f16", 1, 2), 30: ("bf16", 1, 2), 12: ("q4_k", 256, 144),
         13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210), 16: ("iq2_xxs", 256, 66),
         17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98), 20: ("iq4_nl", 32, 18),
         21: ("iq3_s", 256, 110), 22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
         42: ("q2_0", 64, 18), 8: ("q8_0", 32, 34)}
BLOCKS_WANTED = {"f32": 128, "f16": 128, "bf16": 128, "iq4_nl": 16, "q2_0": 8, "q8_0": 32}  # default 8

def parse(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = rd("IQQ")
    assert version == 3
    def rstr():
        (n,) = rd("Q"); return f.read(n).decode("utf-8", "replace")
    def rval(t):
        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in sizes: return rd(sizes[t])[0]
        if t == 8: return rstr()
        if t == 9:
            (et, n) = rd("IQ")
            return [rval(et) for _ in range(n)]
        raise SystemExit(f"bad kv type {t}")
    align = 32
    for _ in range(n_kv):
        k = rstr(); (t,) = rd("I"); v = rval(t)
        if k == "general.alignment": align = v
    tensors = []
    for _ in range(n_tensors):
        name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd); (ty, off) = rd("IQ")
        tensors.append((name, dims, ty, off))
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return tensors, data_start, align

def main():
    path, out = sys.argv[1], sys.argv[2]
    tensors, data_start, align = parse(path)
    size = os.path.getsize(path)
    by_type = {}
    for name, dims, ty, off in tensors:
        by_type.setdefault(ty, []).append((name, dims, off))
    summary = {}
    for ty, lst in sorted(by_type.items()):
        summary[TYPES.get(ty, (f"type{ty}",))[0]] = len(lst)
    print("tensors by type:", summary)
    if "--list" in sys.argv: return
    os.makedirs(out, exist_ok=True)
    f = open(path, "rb")
    index = {"shard": os.path.basename(path), "shard_size": size, "data_start": data_start,
             "alignment": align, "tensor_counts": summary, "types": {}}
    for ty, (tn, be, bb) in TYPES.items():
        lst = by_type.get(ty)
        if not lst: continue
        want = BLOCKS_WANTED.get(tn, 8)
        # First tensor of this type with enough blocks; blocks start at the middle.
        for name, dims, off in lst:
            n_elems = 1
            for d in dims: n_elems *= d
            nb = n_elems // be
            if nb >= 4 * want:
                start_block = (nb // 2) // want * want
                file_off = data_start + off + start_block * bb
                f.seek(file_off); data = f.read(want * bb)
                assert len(data) == want * bb and file_off + len(data) <= size
                open(f"{out}/real-{tn}.blocks", "wb").write(data)
                index["types"][tn] = {"tensor": name, "dims": list(dims), "tensor_offset": off,
                    "start_block": start_block, "blocks": want, "file_offset": file_off,
                    "byte_len": len(data), "tensors_of_type": len(lst)}
                break
    import hashlib
    hs = hashlib.sha256()
    with open(path, "rb") as g:
        while chunk := g.read(16 << 20): hs.update(chunk)
    index["shard_sha256"] = hs.hexdigest()
    json.dump(index, open(f"{out}/real-index.json", "w"), indent=1)
    print("extracted:", sorted(index["types"]))
    print("absent:", sorted(set(v[0] for v in TYPES.values()) - set(index["types"])))
main()
