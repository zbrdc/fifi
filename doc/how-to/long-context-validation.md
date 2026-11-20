# Validate long-context recall

Use the long-context driver to check a cold, synthetic prompt at a server-counted
length. It places three random codes near the beginning, middle, and end, then
checks the complete reply. Run it only when the server and device are available
for the test; it does not reserve hardware or verify the running binary's identity.

## Run a 128k prompt

Start a fifi engine with enough context for the prompt and reserved reply. From
the repository root, choose a writable scratch directory and run:

```sh
mkdir -p "$HOME/fifi-scratch/long-context"
export TMPDIR="$HOME/fifi-scratch/long-context"
python3 scripts/bench_long_context.py \
  --url http://127.0.0.1:18792 \
  --model fifi-q36 \
  --prompt-tokens 131072 \
  --max-tokens 8192 \
  --output "$TMPDIR/recall-128k.json"
```

The server's `/v1/system`, `/v1/token/encode`, and `/v1/chat/completions`
endpoints must be available at the supplied URL. `/v1/recent-requests` is
optional and supplies the decode rate when it can be matched to the response.
If the server uses an API key, pass its environment variable name with
`--api-key-env FIFI_API_KEY`; the key is not written to the result.

The driver counts the rendered chat prompt using the server's tokenizer. It
requires at least `--prompt-tokens` and allows up to `--tolerance` additional
tokens, 128 by default. Prompt and reply both consume context. A 131,072-token
prompt with an 8,192-token reply reservation therefore needs at least 139,264
tokens of context, plus any tokens by which the selected prompt exceeds its target.

For a run near a 262,144-token *total* context limit, leave room for a reply:

```sh
python3 scripts/bench_long_context.py \
  --url http://127.0.0.1:18792 \
  --model fifi-q36 \
  --prompt-tokens 257900 \
  --max-tokens 4096 \
  --output "$TMPDIR/recall-near-256k.json"
```

A 262,144-token input plus a reply requires a context limit greater than
262,144. The driver checks both the requested and actual prompt lengths against
the reported limit before sending the generation request.

## Check a two-turn coding session

The coding-session driver fills the first prompt with synthetic Python source,
asks for a small code edit, and then asks for a revision in the same chat. It
compares the returned statements as Python syntax trees and never executes the
model's code. It counts both rendered prompts and verifies the first was cold.
By default, the second turn must report at least one reused token; set a higher
`--min-reused-tokens` threshold to require substantial reuse. This is the
`required` reuse mode. Use `--reuse-mode disabled` to check a server or path
that does not support prefix reuse and must report zero reused tokens on both
turns. A pass in disabled mode does not qualify warm prefix reuse.

```sh
python3 scripts/bench_coding_session.py \
  --url http://127.0.0.1:18792 \
  --model fifi-q36 \
  --prompt-tokens 131072 \
  --max-tokens 128 \
  --min-reused-tokens 100000 \
  --output "$TMPDIR/coding-session-128k.json"
```

For a path without prefix reuse, replace the reuse threshold with
`--reuse-mode disabled` and use a separate output file. In this mode the result
records `reuse_passed` as null and checks the expected absence of reuse through
`reuse_check_passed`.

For a 262,144-token total context, use a first-prompt target such as `257000`
to leave space for both turns. The driver reserves two replies before starting
and checks the exact follow-up count before sending it. Its result contains
each turn's count, reuse, answer, edit verdict, and matched server timings.
The estimated uncached-token rates exclude reported reused tokens. The client
rate includes tokenization, queueing and response overhead; the server rate
uses the engine-start-to-first-batch interval when it is available.
The generated project and edits are synthetic: passing checks a narrow coding
interaction and does not establish general coding quality.

## Compare prefix retention after a competing request

Use the same binary, model, tokenizer, seed and context limit for both runs.
For a controlled comparison, start each run with an empty prefix store. Run
the following command with one cache entry, then reload with two entries and
repeat with `--expect-return retained` and a separate output path.

A running endpoint can also supply a baseline when its serving window allows
inference traffic. Check the model, the serving binary and the request queue,
then use a fresh seed. The driver requires zero reused tokens on the first
request at each depth. Check the server request log for competing traffic
before interpreting the timings. Changing the server or cache configuration
uses the [GPU service handoff procedure](https://github.com/zbrdc/fifi/blob/906e162ff66331c1d8ce7550180f5ce46f42b84d/doc/how-to/second-gpu-instance.md#gpu-test-coordination).

```sh
python3 scripts/bench_prefix_returns.py \
  --url http://127.0.0.1:18792 --model fifi-q36 \
  --seed prefix-return-01 --expect-return evicted \
  --output "$TMPDIR/one-entry.json"
```

At each of eight total context sizes from 2,048 to 262,144 tokens, the driver
reserves reply space and sends A, repeats A, sends a different archive B,
then returns to A. Use `--contexts 2048` for a smoke test. Every answer must
contain its archive's three codes in order. The adjacent repeat must reuse
tokens beyond the prefix common to A and B. A retained return must recover at
least the adjacent repeat's reuse. Compare prompt token hashes across both runs.

The JSON records cached and newly processed tokens, time to first visible
content, and request-matched server timings. Prefill rate estimates count
only newly processed tokens. A capacity refusal records the engine's advertised
limit and skips generation at that depth. Exit status 2 covers a failed check,
a refused depth, or missing required timings. An interrupted or invalid stream
leaves evidence in the corresponding `.partial` file.

An [RTX 2080 Ti measurement](../../scripts/benchmarks/qwen36-prefix-returns-2080.json) records all four turns at each admitted depth, fixed-length decode controls, and the server’s responses to over-limit requests.

### Record committed checkpoint bytes

Retain the engine log from each server process that ran the comparison.
Join the finished benchmark to that process's log by request ID.

```sh
python3 scripts/record_prefix_checkpoint_bytes.py \
  --benchmark "$TMPDIR/one-entry.json" --request-log "$TMPDIR/engine.log" \
  --output "$TMPDIR/one-entry-checkpoint-bytes.json"
```

The report contains the checkpoint entry count and committed payload bytes
sampled by the engine for each verified turn. It also lists depths refused by
the advertised context limit. Allocator reserves, temporary copy buffers and
GPU memory high-water marks need separate measurements. Keep one log per server
process because request IDs can repeat after restart.

## Read the result

The JSON result records the counted prompt and completion tokens, prefix reuse,
finish reason, expected codes, answer, and `recall_passed`. A passing recall
requires all three codes in order and a normal `stop` finish. A truncated or
incorrect answer is saved as a completed failure and the command exits with
status 2. A missing token count, unconfirmed cold prompt, or incomplete stream
leaves a `.partial` failure record and exits with status 1. Existing output and
partial files are not overwritten.

`estimated_prompt_tokens_per_second` divides the server-counted prompt by the
time until the first visible streamed content. It includes tokenization,
queueing, and response overhead. When a matching completed server record is
available, `server_prefill_ms` measures engine start to the first generated
batch, and `server_prompt_tokens_per_second` divides the cold prompt count by
that interval. It includes the first-batch work, so it is also an estimate.
`server_queue_wait_ms` and `server_decode_tokens_per_second` come from the same
record when those fields are available. Stream chunks are never counted as
tokens.

This synthetic check does not establish coding quality, multi-turn behavior,
device isolation, or exact build identity. Record the running binary, model,
hardware, and isolation separately when using the result as performance evidence.

## RTX 2080 Ti tiered tuning measurements

These same-binary comparisons used Qwen3.6-35B-A3B Q4_K_M, Q8 KV, a
65,536-token context limit, eight CPU workers, and fresh prompts. Prompt
rates include the short completed answer. Each row measures a distinct
switch or setting against its own control.

| Change | Earlier build | Measured result |
| --- | --- | --- |
| Coalesce tiered prefill through 8K | `52a85f4a` | At 7,904 prompt tokens, 318.21 to 369.81 tokens/s (+16.2%). The short and medium cases ranged from −0.3% to +1.4%. |
| Tensor prefill attention on Turing, with wide prefill enabled in both arms | `70f81539` | At 7,904 prompt tokens, 364.19 to 684.39 tokens/s (+87.9%). The 4,277-token case gained 42.9%. In the 1,024-row attention microbenchmark, 18.50 to 0.61 ms. |
| Stage an expert only after at least 16 routed rows in the piece | `9ae8611f` | The 1,977- and 4,277-token cases improved 9.3% and 19.2%. The 7,904-token result was within the control spread. |
| Limit split CPU decode dispatch to four misses per layer | `a8b90af1` | Cold technical decode went from 45.05 to 46.21 tokens/s (+2.6%). Warm technical decode lost 1.3%. |
| Admit 4,096 rows with a 64-expert staging cap | `08aa8242` | At 4,277 tokens, 809.05 to 917.28 tokens/s (+13.4%). At 7,904 tokens, 714.25 to 767.75 tokens/s (+7.5%). |
| Raise the staging cap from 64 to 78 with 4,096 rows in both arms | `08aa8242` | At 7,904 tokens, 762.13 to 890.67 tokens/s (+16.9%). The 1,977-token case lost 4.6%. |
| Queue expert warm-up uploads through pinned memory | `47f3c96a` | Median warm-up upload time fell from 124.6 to 87.4 ms. First token arrived 15 to 73 ms sooner across three matched prompts. |

The [warm-up and upload record](../../scripts/benchmarks/qwen36-tiered-warmup-2080.json)
includes the earlier builds, matched controls, route counts, and factual checks.
The older 4,096-row trials also enabled their build's asynchronous decode
fills. The short-piece staging cap has host coverage and awaits a GPU
measurement. Repeat matched model-quality and concurrent long-context runs
on the current tip before enabling these controls by default.

For the fixed short-answer corpus, run
`python3 scripts/bench_corpus_accuracy.py BASE_URL API_KEY MODEL OUT.json`
once per configuration and compare the two reports with
`python3 scripts/bench_corpus_accuracy.py --compare A.json B.json`.

An earlier standalone RTX 2080 Ti run with Q8 KV and one saved
conversation took 834.27 seconds to first token for a 65,351-token prompt.
An immediate repeat reused 65,344 tokens and took 0.87 seconds.
After another conversation displaced that checkpoint, the original prompt
took 836.37 seconds. Each figure is one observation. Separate 128-token
continuations decoded at 80.19 tokens/s near 2K and 66.77 tokens/s near 64K.
[Original workload and measurements](https://github.com/zbrdc/fifi/pull/89)

The coding benchmark accepts `--reuse-mode disabled` to require zero reused
tokens on the second turn. Its default `--reuse-mode required` checks the
minimum reuse in addition to validating the requested edit.

## Earlier peer CUDA measurements

The older peer measurements used separate builds and workloads. Use each
linked report for its exact device order, source revision, and settings.

| Earlier measurement | Result | Source |
| --- | --- | --- |
| 512-row prefill and recall | Passed on 8,311 and 32,884 counted prompt tokens. Server prefill took 17.87 and 70.68 seconds. | [Upstream PR #42](https://github.com/zbrdc/fifi/pull/42), [fork PR #4](https://github.com/BColsey/fifi/pull/4) |
| 1,024-row long recall | Passed on 131,195 counted prompt tokens. Server prefill took 157.78 seconds. | [Upstream PR #43](https://github.com/zbrdc/fifi/pull/43), [fork PR #10](https://github.com/BColsey/fifi/pull/10) |
| 2,048-row build and host checks | Ten host checks were recorded and peer-device tests were enumerated. | [Upstream PR #44](https://github.com/zbrdc/fifi/pull/44), [fork PR #11](https://github.com/BColsey/fifi/pull/11) |
| Peer prefix diagnostic | Separately checked aligned state and logits, and logged natural continuation comparisons. | [Upstream PR #46](https://github.com/zbrdc/fifi/pull/46), [fork PR #6](https://github.com/BColsey/fifi/pull/6) |
| Cross-device event barriers | Long-prompt speed ratios were 1.0003 and 0.9989 across the two run orders. The separate resource records include CPU and GPU sampling. | [Upstream PR #47](https://github.com/zbrdc/fifi/pull/47), [fork PR #8](https://github.com/BColsey/fifi/pull/8), [resource observer PR](https://github.com/BColsey/fifi/pull/7) |
| Incremental KV staging | Short- and long-prompt time-to-first-token ratios were 1.010 and 1.036. Median decode was 38.66 and 38.64 tokens/s without and with staging. | [Upstream PR #59](https://github.com/zbrdc/fifi/pull/59), [fork PR #1](https://github.com/BColsey/fifi/pull/1) |
| Multi-block ring, two coding turns | Both edits passed at 261,696 and 261,740 prompt tokens, with zero reuse. First token arrived in 633.08 and 633.21 seconds. | [Fork PR #9](https://github.com/BColsey/fifi/pull/9) |
| Peer expert placement | Records the older RTX 3080 placement experiment. The upstream placement code now charges each card's resident KV layout. | [Fork PR #3](https://github.com/BColsey/fifi/pull/3), [upstream PR #90](https://github.com/zbrdc/fifi/pull/90) |

The paired host observer, bounded client, and scoped workload tests are
documented in the [Peer-pair host observer](../explanation/p2p-pair-host-observer.md).
