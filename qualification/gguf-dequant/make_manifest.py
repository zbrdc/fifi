#!/usr/bin/env python3
"""Usage: make_manifest.py <work-dir> <fixture-out-dir> <seed> <oracle-src-dir> [<real-dir>]
Copies the nofma fixtures (and real-tensor fixtures when <real-dir> is given)
and writes manifest.json with provenance, hashes and fma differences.
Existing synthetic fixture files are rewritten with identical bytes."""
import hashlib, json, os, shutil, sys
work, out, seed, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
real = sys.argv[5] if len(sys.argv) > 5 else None
BB = {"q2_0":(18,64,"Q20"),"q4_k":(144,256,"Q4K"),"q5_k":(176,256,"Q5K"),"q6_k":(210,256,"Q6K"),"q8_0":(34,32,"Q80"),
 "iq2_xxs":(66,256,"Iq2Xxs"),"iq2_xs":(74,256,"Iq2Xs"),"iq2_s":(82,256,"Iq2S"),
 "iq3_xxs":(98,256,"Iq3Xxs"),"iq3_s":(110,256,"Iq3S"),"iq4_nl":(18,32,"Iq4Nl"),
 "iq4_xs":(136,256,"Iq4Xs"),"f32":(4,1,"F32"),"f16":(2,1,"F16"),"bf16":(2,1,"Bf16")}
# 12 upstream files; "included" = pulled in by the oracle build (clang -MM), LICENSE and
# ggml-cpu/quants.h are recorded for provenance.
UPSTREAM = [("ggml/src/ggml-common.h",1),("ggml/src/ggml-quants.c",1),("ggml/src/ggml-quants.h",1),
 ("ggml/src/ggml-impl.h",1),("ggml/include/ggml.h",1),("ggml/include/ggml-backend.h",1),
 ("ggml/include/gguf.h",1),("ggml/include/ggml-cpu.h",1),("ggml/include/ggml-alloc.h",1),
 ("ggml/src/ggml-cpu/ggml-cpu-impl.h",1),("ggml/src/ggml-cpu/quants.h",0),("LICENSE",0)]
hb = lambda b: hashlib.sha256(b).hexdigest()
h = lambda p: hb(open(p,'rb').read())
os.makedirs(out, exist_ok=True)

def entry(name, ext_work, base, bb, be, variant, fma_path, extra=None):
    for ext in ("blocks","f32"):
        dst = f"{out}/{base}.{ext}"
        s = ext_work(ext)
        if os.path.abspath(s) != os.path.abspath(dst): shutil.copyfile(s, dst)
    a = open(f"{out}/{base}.f32",'rb').read()
    b = open(fma_path,'rb').read()
    n = len(a)//4
    diff = sum(1 for i in range(n) if a[4*i:4*i+4] != b[4*i:4*i+4])
    e = {"name":base,"variant":variant,"block_bytes":bb,"block_elems":be,
      "blocks":os.path.getsize(f"{out}/{base}.blocks")//bb,"elems":n,
      "input_sha256":h(f"{out}/{base}.blocks"),"output_sha256":h(f"{out}/{base}.f32"),
      "fma_differing_elems":diff}
    if extra: e.update(extra)
    return e

types = []
for name,(bb,be,v) in BB.items():
    types.append(entry(name, lambda ext,name=name: f"{work}/fixtures-nofma/{name}.{ext}", name, bb, be, v,
                       f"{work}/fixtures-fma/{name}.f32"))
extra = []
for name in ("q4_k","q5_k"):
    bb,be,v = BB[name]
    extra.append(entry(name, lambda ext,name=name: f"{work}/fixtures-nofma/{name}_scales.{ext}",
                       f"{name}_scales", bb, be, v, f"{work}/fixtures-fma/{name}_scales.f32",
                       {"note":"64 blocks; every 6-bit scale and min value occurs in all 8 slots"}))
realents = []
if real:
    idx = json.load(open(f"{real}/real-index.json"))
    shard_sha = idx.get("shard_sha256")
    for name,info in idx["types"].items():
        bb,be,v = BB[name]
        realents.append(entry(name, lambda ext,name=name: f"{real}/real-{name}.{ext}", f"real-{name}", bb, be, v,
            f"{work}/real-{name}.fma.f32",
            {"type":name,"shard":idx["shard"],"shard_size":idx["shard_size"],"shard_sha256":shard_sha,
             "tensor":info["tensor"],"dims":info["dims"],"tensor_offset":info["tensor_offset"],
             "start_block":info["start_block"],"file_offset":info["file_offset"],"byte_len":info["byte_len"],
             "data_start":idx["data_start"]}))
files = {rel: h(f"{src}/{rel}") for rel,_ in UPSTREAM}
agg = hashlib.sha256()
for e in types + extra + realents:
    agg.update(bytes.fromhex(e["input_sha256"])); agg.update(bytes.fromhex(e["output_sha256"]))
info = open(f"{work}/build-info.txt").read().splitlines()
json.dump({"oracle_repo":"https://github.com/ggml-org/llama.cpp",
 "oracle_commit":"fb4b2737a808a3fb7c2117a498f43815dc9be53e","seed":seed,
 "oracle_files_sha256":files,
 "oracle_files_included_in_build":[r for r,i in UPSTREAM if i],
 "oracle_build":{"info":info,"fixture_flags":"-O2 -ffp-contract=off"},
 "aggregate_fixture_sha256":agg.hexdigest(),
 "types":types,"extra":extra,"real":realents}, open(f"{out}/manifest.json","w"), indent=1)
print("aggregate", agg.hexdigest())
