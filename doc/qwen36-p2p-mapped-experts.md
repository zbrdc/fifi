# Experimental host-backed experts for two-device Qwen3.6

The native GGUF two-device path can place some of either device's sliced
expert weights in page-locked host RAM. The GPU reads those weights over PCIe;
both GPUs still execute their shards and use peer access. This option is off by
default and has not been qualified for production use.

Set `Q36_P2P_HOST_EXPERT_BYTES` to the minimum number of expert bytes to place
in host RAM before starting `enginesrv`, for example `2147483648` for 2 GiB.
The loader selects complete expert slices, so the actual amount can exceed the
requested minimum by at most one slice. Only sliced quantized routed expert
stacks on the second device in `cuda_devices` are eligible under that setting.
Set `Q36_P2P_HOST_EXPERT_BYTES_0` independently to map slices on the first
device. Both settings default to zero. A request that cannot be met fails
before that device's weights are uploaded.

The loader checks available host RAM and leaves an additional 1 GiB in its
admission calculation. It also retains the usual per-device reservations and
the free-VRAM check after loading. Available host RAM can change during a
load, and pinned pages cannot be swapped. Direct host-memory reads can reduce
inference throughput; measure on the target hardware before relying on this
option. Remove both host-expert environment variables to restore the existing
GPU-only placement.

On systems where peers can advance at different rates, set
`Q36_P2P_SYNC_BROADCAST=1` as well. This experimental option waits for every
card to consume a broadcast before reusing the ring's two mailbox slots;
it prevents a faster publisher from overwriting a slower peer's message.
The additional synchronizations can reduce throughput.

## Tune host placement

The requested host-byte values are minimums. Reducing them keeps more expert
slices in GPU memory, which can improve throughput when PCIe is the limiting
factor, but leaves less free VRAM. Choose values for the actual device order,
context size, and other GPU users. The loader reports the selected host bytes
and remaining GPU memory at startup.

First measure the existing placement against an idle, ready engine. These
commands use a key-free endpoint at `http://127.0.0.1:18792`; pass `--url`
to both scripts if the endpoint is elsewhere. Use output paths that do not
already exist:

```sh
python3 scripts/bench_p2p_prefill.py --samples 3 --repeat 100 --output /tmp/qwen36-p2p-before.json
python3 scripts/compare_p2p_placement.py --output /tmp/qwen36-p2p-replies-before.json
```

For example, a 12 GB RTX 3060 followed by a 10 GB RTX 3080, with a 131,072-token
context and q8 KV cache, loaded with these settings on one system:

```sh
export Q36_P2P_HOST_EXPERT_BYTES_0=1342177280
export Q36_P2P_HOST_EXPERT_BYTES=3489660928
```

Restart the engine with candidate values, verify its startup memory report,
and measure again using a new output filename:

```sh
python3 scripts/bench_p2p_prefill.py --samples 3 --repeat 100 --output /tmp/qwen36-p2p-after.json
python3 scripts/compare_p2p_placement.py --baseline /tmp/qwen36-p2p-replies-before.json --output /tmp/qwen36-p2p-replies-after.json
```

Compare the samples and the free VRAM after loading. The comparison script
also checks that four deterministic replies, finish reasons, and token counts
match. The rates include the complete short reply, not just the prefill kernels.
The example values are not a general recommendation: a different model,
context, device order, or background GPU process can require more host-backed
memory. Restore the prior settings and restart if the candidate does not fit
or is slower.

### Measured placement with a 256k context limit

On one 12 GB RTX 3060 followed by one 10 GB RTX 3080, an experimental build
using Q4_K_M weights, q8 KV, a configured 262,144-token limit, 256-row P2P
chunks and Q8 staging was measured with these settings:

```sh
export Q36_P2P_HOST_EXPERT_BYTES_0=1395864370
export Q36_P2P_HOST_EXPERT_BYTES=3470000000
```

Compared with `Q36_P2P_HOST_EXPERT_BYTES=3489660928`, this moved one
75,497,472-byte expert slice from host memory onto the 3080. In one
control-then-candidate run of three cold 4,274-token prompts per setting,
median server job-start-to-first-token rate was 304.20 versus 307.80 tokens/s;
median decode rate over three deterministic 128-token replies was 38.66
versus 38.94 tokens/s. The first-token rate includes first-token generation.
The candidate had 1,400 MiB free on the 3060 and 1,435 MiB free on the 3080
after warmup. These small differences may include order or thermal effects.

The run used a [separate integration commit and recorded the full samples,
guard and restoration evidence](../scripts/benchmarks/qwen36-p2p-3080-placement-ab.json).
Its longest prompt was 4,274 tokens; the configured context limit was not a
128k or 256k prompt test. This setting is experimental and should be measured
on the target system, with the prior values retained for rollback. A more
aggressive setting that also moved two slices onto the 3060 was refused by
that run's memory admission check.
