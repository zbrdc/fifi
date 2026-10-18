# AGENT-1 acceptance harness

This directory holds the acceptance harness for AGENT-1a. It runs a tool-calling client against
`enginesrv` on the Mac. A pass is a structural result. It makes no speed claim and it does not
qualify Phase 0B or Phase 1B.

## Files

| File | Use |
| --- | --- |
| `agent_client.py` | Minimal chat client with timings, standard library only. Refuses port 8788. |
| `sandbox.py` | The only way the model's code runs: `docker run` with the pinned image, no network, read-only root. Never pulls. |
| `acc_a.py` | ACC-A: the scripted tool loop (`write_file`, `run_python`) and its checks. |
| `smoke.py` | One short request, for the engine smoke run. |
| `check_pi_run.py` | ACC-B checker for the JSON event stream of `pi`; needs `--server-log`. |
| `reqlog_check.py`, `check_server_log.py` | Checks of the server request log (one line per request, status 200, outcome `ok`). |
| `fake_server.py`, `fake_docker.py` | Test doubles of the chat server and of `docker`. The fake docker runs nothing. |
| `test_*.py`, `testdata/` | Python self-tests. `test_sandbox_real.py` needs a real image and is skipped without one. |
| `mac-acc-a.sh` | Mac side, mode `serve` only: build and start `enginesrv`, wait for `STOP`, stop that server, write evidence. |
| `kubdan-acc-a.sh` | kubdan side of the CUDA run (`--engine q36cuda`): GPU preflight, model pins, build with `--features native-cuda`, serve, `vram.tsv`, evidence. |
| `run-acc-a.sh` | Linux launcher for ACC-A: sandbox gate, lock, serve session, tunnel, client here, copy back, seal. |
| `run-acc-b.sh` | Linux launcher for ACC-B: the container of design 4.3, the unix-socket forward of 4.2, the export extractor, the checker, the seal. |
| `accb-entry.sh`, `accb-bridge.js` | The container entry (PID 1) and the in-container bridge (127.0.0.1:18791 to the socket, at most 8 connections). |
| `accb-relay.py` | The rehearsal relay: a unix-socket stand-in for `ssh -L` (used by `run-acc-b.sh --rehearse`). |
| `accb_export.py`, `test_accb_export.py` | The export extractor of design 4.6 (untrusted tar) and its tests. |
| `selftest-accb.sh` | The pinu self-tests of the launcher: stage-1a flag tests and the stage-1b/2 launcher tests. |
| `mutate-accb.sh` | One mutation per control of design 6; `selftest-accb.sh` must kill each. |
| `driver-lib.sh` | Functions shared by the launchers, including the abort-safe session stop. |
| `q36-pins.sha256` | Expected SHA-256 of four Q36 checkpoint files; `mac-acc-a.sh` checks them for `--engine q36`. |
| `q36-cuda-pins.sha256` | Expected SHA-256 of the GGUF and the four tokenizer files of the CUDA model dir; `kubdan-acc-a.sh` checks them. |
| `selftest.sh` | Rehearses the scripts on this host against the fake server and the fake docker. |
| `selftest-cuda.sh` | CUDA part of the self-test (sourced by `selftest.sh`, also runs alone). |
| `mutate-q36-cuda.sh` | Mutation check of the CUDA refusals (run it after `selftest.sh`). |

## Sandbox for model-written code

Model-written code never runs on a host, and the Mac never runs the client. The ACC-A tool loop
runs on the launcher host (pinu). It writes the model's `hello.py` into a scratch directory, copies
the file into a private staging directory, and runs it only in `docker run` with:
`--network none`, `--read-only`, `--tmpfs /tmp:rw,noexec,nosuid,size=16m`, `--user 65534:65534`,
`--memory 256m --memory-swap 256m`, `--cpus 1`, `--pids-limit 64`, `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--shm-size 16m`, the staging directory mounted read-only at
`/work`, and `timeout -s KILL 30` inside the container. The key is in no variable of the container.
The container runs with `--pull never --log-driver none`. The harness captures at most 1 MiB of
the program's output through a capped reader and removes the container when the cap or the
timeout is hit, so a print loop fills neither memory nor disk. A docker failure (exit 125 to 127)
is a sandbox error, not a quality miss; the program cannot use those exit codes.
pinu runs a root `dockerd` and a rootless one, and the user is in group `docker`. The harness uses
the rootless daemon only: every docker call is `docker -H unix:///run/user/<uid>/docker.sock`
(override with `SANDBOX_DOCKER_HOST`), and `docker info` must report the security option
`rootless`. Otherwise the run stops with status 69 before any lock, server, or model call. The
evidence file `sandbox-daemon.txt` records the pinned host, the runc, containerd, and engine
versions, and the image id.
`SANDBOX_IMAGE` names the image by digest, `name@sha256:<64 hex>`. The approved image is
`python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`
(python:3.12-slim, amd64, created 2026-10-06T02:04:40Z; record at
`/home/dan/fifi/scrch/orchestrator/evidence/acc-sandbox-image.txt`). If `SANDBOX_IMAGE` is unset,
not pinned by digest, or not present locally (`docker image inspect`), the run stops with
"sandbox image missing; needs the user's approval to pull" and status 67, before the lock, the
server, or any model call. No script pulls an image.`SANDBOX_REAL_IMAGE=<that name@sha256> python3 -I -m unittest discover -s . -p test_sandbox_real.py`
checks the isolation against the real docker (network, read-only root, uid, key, timeout).

## Self-tests (no Mac, no model)

```
cd qualification/agent-acceptance
python3 -I -m unittest discover -s . -p 'test_*.py'
bash selftest.sh            # about 2 minutes; fake server and fake docker, no container
bash selftest-accb.sh       # pinu only: the real rootless daemon, the pinned image, containers
bash mutate-accb.sh         # pinu only: every mutation of design 6 must be killed
```
The client files use only Python 3.9 syntax, because the Mac has Python 3.9.6.

## ACC-A (the coordinator runs it)

The launcher takes `--engine flash|q36`. `flash` (the default) serves Qwen3.8-Flash-Next as the
model id `flash`. `q36` serves Qwen3.6-35B-A3B (family `qwen3_6_35b_a3b`, Q36-F3c) as the model id
`fifi-q36`. Both engines run the same checks and the same client.

Prerequisites: H6, H8, and H8c are merged; no T0b cold-read run is active; the Mac lock is free;
`SANDBOX_IMAGE` is set to the approved image and present.
```
SANDBOX_IMAGE=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 \
  bash qualification/agent-acceptance/run-acc-a.sh <new evidence dir> --branch main
```
The same run against the Q36 engine:
```
SANDBOX_IMAGE=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 \
  bash qualification/agent-acceptance/run-acc-a.sh <new evidence dir> --branch <branch> --engine q36
```
The launcher checks the sandbox, holds the Mac lock `mac.lock`, bundles the branch, clones it on the
Mac, and starts `mac-acc-a.sh serve`. That script builds `enginesrv` with `--features native-mlx`,
writes the config, starts the server on `127.0.0.1:18791`, waits for `/ready` (up to 1,800 s), and
waits for the `STOP` file. The launcher opens `ssh -N -L 18791:127.0.0.1:18791`, runs `acc_a.py` on
this host, then touches `STOP`. The Mac script stops the server by the PID it recorded (SIGTERM,
never SIGKILL). The client sees the server through an ssh tunnel, so its timings include the tunnel.
An abort (SIGTERM, SIGINT) of the launcher stops the client, the tunnel, and the Mac session
(`STOP` is checked during the build, the load, and the run), and the lock is released because no
background child keeps its file descriptor.
The bearer key is generated on the host, goes to the Mac on stdin, and is never written to disk. It is in no process argument: the final key scan gives it to `grep -f -` on standard input, and `selftest.sh` logs the argv of the tools and samples `ps -eo args` during a rehearsal.
Evidence: `acc-a/{timings.tsv,checks.tsv,transcript.jsonl,sse-*.txt,result.json}` and `acc-a-work/`
(the model's files) on this host; `evidence/{identity.txt,system-before.txt,system-after.txt,
rss.tsv,server.log,summary.txt}` from the Mac; `SHA256SUMS` in `evidence/` and at the top (the top
file covers the inner one). After the server stops, `check_server_log.py` checks one log line per
request and status 200 with outcome `ok` on every line, and turns a `pass` into a pipeline defect
when a check fails. `timings.tsv` holds prompt tokens, time to headers, time to the first delta,
total time, and decode tokens per second of the streamed request.
`timings.tsv` writes `decode_tok_s` only for the streamed request, and only when all three hold:
at least 2 text deltas arrived (a delta with `content` or `reasoning_content`; a delta with only
`tool_calls` does not count), the reply has at least 2 completion tokens, and
`total_seconds - first_text_delta_seconds` is at least 0.1 s (`MIN_RATE_INTERVAL_SECONDS` in
`agent_client.py`). The value is `(completion_tokens - 1) / (total_seconds - first_text_delta_seconds)`.
Otherwise the field is empty, never a number. The reason: the server sends a complete tool call as
a burst of chunks within milliseconds, so the streamed tool-call request of ACC-A normally has an
empty rate. A quotient over such a burst is not a decode speed. The 0.1 s floor is above the jitter
of one local burst. The rate is a client-side figure through the ssh tunnel, not an engine speed claim.
Exit status: 0 pass, 2 pipeline defect, 3 model quality miss, 4 prompt budget, 64 to 95 setup
(67 sandbox image missing, 69 docker daemon not rootless, 86 the server ended during the run, 88 the
Q36 checkpoint does not match its pins, 95 the server did not stop).
The stream rule that `acc_a.py` enforces on the streamed request: role in the first chunk, each
tool call in its own chunk, then a final chunk with an empty delta, the finish reason, and the
usage; content type `text/event-stream`; `[DONE]` last. The final reply needs finish reason `stop`.
Read `swap_grew` in `evidence/summary.txt`: it does not change the exit status, and the plan
requires it to be `no`.

Each run uses a unique scratch and worktree name on the Mac (`runs/agent1-<mode>-<UTC time>-<pid>`), so
a later run never deletes the directory of a server that is still running. The coordinator removes old
run directories.

## ACC-A on CUDA (the coordinator runs it)

`--engine q36cuda` runs the same ACC-A loop and the same checks against `enginesrv` built with
`--features native-cuda` on kubdan's RTX 3090 (CUDA-SERVE gate G2). The model id is `fifi-q36`
(family `qwen3_6_35b_a3b`) served from the pinned GGUF checkpoint, so the same `acc_a.py`, the same
`check_server_log.py`, and the same client run; only the serve side and the guard rails differ.

The baton rule: the GPU is shared with the user's gaming and with other lanes' baton-gated holds,
so the launcher refuses to start (status 66, before the lock, the build, and any model call)
unless the environment variable `GPU_BATON` names an existing file that the coordinator created
after git-db granted the 3090. The launcher copies that file into the evidence (`gpu-baton.txt`)
and never creates it. The launcher also holds the new `kubdan-gpu.lock` (flock, exclusive, beside
`mac.lock`) and needs a working `ssh -o BatchMode=yes kubdan` (status 64 otherwise). The bundle
goes to kubdan by `scp`, the ssh tunnel forwards the port (default 18792; 8788 and the Mac port
18791 are refused), and the Mac modes are untouched.

The kubdan side is `kubdan-acc-a.sh serve <scratch dir> <worktree>`, started by the launcher over
ssh with the bearer key on stdin only (a fourth argument is refused, and so is an
`ENGINESRV_API_KEY` that is already in the environment). It waits for `/ready` (default up to
1,800 s, `READY_TIMEOUT`), follows the `STOP` file protocol, stops the server it started by its
recorded PID (SIGTERM, never SIGKILL), and writes `evidence/{identity.txt, system-before.txt,
system-after.txt, rss.tsv, vram.tsv, server.log, server.out, config.json, summary.txt,
SHA256SUMS}`; `vram.tsv` samples `nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,
clocks.sm --format=csv,noheader` every 5 s. Its preflight stops with a distinct status and starts
nothing: 71 the GPU is not free (nvidia-smi unusable or no GPU, `memory.used` at or above
2,048 MiB, or a compute process present), 72 another `enginesrv` of this user listens on the port
(70 any other listener), 73 `nvcc` is missing (the CUDA toolkit comes from the environment of
kubdan: `CUDA_PATH`, else `/usr/local/cuda`, else `nvcc` on `PATH`; the RD build image ships
CUDA 12.2), 74 the free disk of the local NVMe is below 30 GiB, and 75 the worktree or the
scratch is not under `/media/dan/nvme/fifi/wrktr/runs` resp. `/media/dan/nvme/fifi/scrch/runs`
(the build, its target dir, and its `TMPDIR` never run on the NFS share, which is pinu's small
root disk). The build is `cargo build --release --features native-cuda --bin enginesrv`. The
model dir is the pinned constant `/media/dan/nvme/models/qwen3.6-35b-a3b`: the GGUF (name,
size 20,419,565,568, sha256, as the gate pins it) and the four tokenizer files are hashed against
`q36-cuda-pins.sha256` before the build; a mismatch stops with status 88. The engine config holds
one entry: family `qwen3_6_35b_a3b`, model id `fifi-q36`, `capture` from `CAPTURE` (0 default, 1)
and `context_limit` from `CONTEXT_LIMIT` (default 16384) in the `qwen36` object, and no MLX-only
key (`cache_limit`, `session_cap`, and the `prefill_*` terms do not apply to the CUDA arm;
`prefix_reuse` is refused and not set). The evidence copy of the config holds no key, only the
`api_key_env` variable name.

The exact commands, capture off and capture on (two separate runs; the evidence records which):

```
export SANDBOX_IMAGE=python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
export GPU_BATON=<the baton file the coordinator created for the 3090>
bash qualification/agent-acceptance/run-acc-a.sh <new evidence dir> --branch <branch> --engine q36cuda
CAPTURE=1 bash qualification/agent-acceptance/run-acc-a.sh <new evidence dir 2> --branch <branch> --engine q36cuda
```

Timeouts: the CUDA prefill runs token by token (about 36 ms per token, so a 6,000-token tool
prompt takes minutes). The client timeout is `ACC_REQUEST_TIMEOUT` per request (default 1,800 s
for this engine; it is passed to `acc_a.py --timeout`, written into `result.json` as
`timeout_seconds`, and into `launcher.txt` as `request_timeout`). The serve-side ready wait is
`READY_TIMEOUT` (default 1,800 s, recorded in `identity.txt`); `STOP_TIMEOUT` (900 s) bounds the
SIGTERM wait. `CONTEXT_LIMIT` (default 16384) sets the engine's context limit.

Status codes added by this mode: 66 the GPU baton is missing, 71 the GPU is not free, 72 another
`enginesrv` of this user listens on the port, 73 `nvcc` is missing, 74 the local NVMe has less
than 30 GiB free, 75 a build path is outside the local NVMe roots (88 pin mismatch and the rest
as the Mac run).

## ACC-B (the coordinator runs it)

`pi` has the `write` and `bash` tools, so a model controls a shell. That shell runs only inside a
sandbox container (design `acc-b-design/`, option (a), approved 2026-10-10). In one sentence, the
topology of design 4.2: an `ssh -N -L` forward on the pinu host turns the Mac loopback port 18791
into a unix socket in `/run/user/1000/acc-b/<run>` (mode 0666, directory 0755, parent 0700), and
the container, with `--network none`, reaches that socket read-only through `accb-bridge.js`, so
no TCP port ever opens on pinu and no other address is reachable at all.

The container of design 4.3 (the flag list is verbatim in `run-acc-b.sh` and printed by
`run-acc-b.sh --print-spec <dir>`): `docker --host unix:///run/user/1000/docker.sock run -i
--pull never --network none --read-only --user 65534:65534 --cap-drop ALL --security-opt
no-new-privileges --memory 1024m --memory-swap 1024m --cpus 1 --pids-limit 128 --shm-size 16m
--ulimit nofile=512:512 --ulimit core=0 --log-driver none`, tmpfs `/tmp` (16m), `/work` (16m),
`/home/acc` (32m), all `rw,noexec,nosuid,nodev`, and exactly seven read-only bind mounts: the
fixture, the reflink copies of the pi 1.0.0 release and the node v22.22.3 binary, the scratch
agent directory, `accb-entry.sh`, `accb-bridge.js`, and the IPC directory. The launcher refuses a
mount source that holds a colon or a comma. The pinned image id, the file hashes of the copies,
and a tree hash over the pi copy are recorded before and after the run; a mismatch fails with
89. `accb-entry.sh` is PID 1: it reads the bearer key as the first stdin line (never an argument,
an environment option, or a file), starts the bridge, runs pi under `timeout -s KILL $PI_TIMEOUT`,
ends every other process, and prints the capped `ACCB-EXPORT` tar of `/work` on stdout. The
status comes from Docker only (`docker run`, `docker inspect`, `docker wait`; a disagreement is
a pipeline defect): the evidence stream is untrusted data (design 4.5), a pass needs the
server-log cross-check, and `check_pi_run.py` requires `--server-log` and reruns `hello.py` only
in a second sandbox container. The export is unpacked by `accb_export.py` (design 4.6) with every
name `repr`-escaped and cut at 200 characters in the manifest.

`--rehearse` needs no Mac and no model: `fake_server.py --mode pi-tools` serves on pinu loopback
and `accb-relay.py` stands in for the ssh forward. The fixed test key comes from `ACC_TEST_KEY`.
The rehearsal's pinu-local TCP port comes from `ACCB_TEST_PORT` (default 18791); the self-tests
pick a free one at start, because a real run opens no TCP port on pinu at all.
A launcher deadline (`ACCB_DEADLINE`, default 6600 s) wraps the run: on the deadline the launcher
runs `docker kill` and exits 91. The former status 68 "container mode pending approval" refusal
is gone: the container mode is approved and built.

Exit status of `run-acc-b.sh` (`selftest.sh` compares this table with the script's own exit-status
comment and with every literal `die`/`exit` status of the script and `driver-lib.sh`):

| Status | Meaning |
| --- | --- |
| 0 | pass (the status of `check_pi_run.py`) |
| 2 | pipeline defect |
| 3 | quality miss |
| 64 | usage (including a fixture path or an `ACCB_SCRATCH` with `:`, `,`, or a newline) |
| 66 | the GPU baton is missing (ACC-A `q36cuda` only, `driver-lib.sh` `require_baton`; `run-acc-b.sh` itself never exits it) |
| 67 | sandbox image missing (`sandbox.py`) |
| 69 | docker daemon not rootless (`sandbox.py`) |
| 70 | local port busy (rehearsal only) |
| 81 | prepare/tunnel |
| 82 | prepare/tunnel |
| 83 | server not ready |
| 84 | forward not ready |
| 86 | the serve session's own status when the checker passed (the server ended early) |
| 89 | pin/hash/flag mismatch |
| 90 | key found in the evidence (the evidence is removed) |
| 91 | launcher deadline |
| 92 | ssh staging failure |
| 93 | pre-start resource guard (4 GiB free memory, 20 GiB free disk) |
| 143 | ended by SIGTERM, SIGINT, or SIGHUP |

The container output readers are bounded byte readers (round-5 review): the stdout stamper reads
64 KiB blocks, stamps each complete line with `<epoch seconds><TAB>`, and enforces the 48 MiB
cap on the stored payload bytes (a line that cannot fit, including a newline-free stream, hits
the cap: `docker kill` is spawned without waiting and the reader keeps draining); the docker
client stderr goes to `pi/accb-peak.txt` capped at 16 MiB with a truncation note (the design
states no stderr number; 16 MiB is the fix5 packet's value). The key scan for a leak runs on
every exit path after the container ran, before the seal: on the normal paths a leak still
removes the evidence and exits 90; an exit that is already abnormal (91, 143, ...) keeps its
stronger status and records `key_leak yes` in `sandbox/teardown.txt`. The Docker status must be
an integer and `docker run`, `docker inspect`, and `docker wait` must agree, or the run is a
pipeline defect that reaches the exit status. The S12 field-by-field compare runs on the probe
container and again on the real run container's saved inspect, covering the mount sources, the
tmpfs option strings, `CapAdd`, `PidMode`, `IpcMode`, `UsernsMode`, `Ulimits`, `MemorySwap`, and
`ShmSize`, and rejecting any `SecurityOpt` beyond `no-new-privileges`. At the start the launcher
reaps stale containers of its own earlier runs only, matched by its naming patterns
(`accb-<UTC time>-<4 hex>` and `accb-t-s12-<4 hex>`); the design 4.3 flag list is exact and
carries no label, so nothing else is ever touched. The entry prints the export line through a
reopened blocking descriptor (`> /proc/self/fd/1`): the runtime holds the container stdout fifo
with `O_NONBLOCK`, and one big `printf` of the multi-MiB export would fail with `EAGAIN`.

Residual risks, accepted (user approvals U4 and the design limits):
- The per-run bearer key is readable by the model's shell by design (U4: the key is random,
  valid for one server run, and authorizes only this server and run). The bridge comment said
  the key stays in the pi process; the truth is that any process in the container can read it
  from the environment, and the launcher's answer is the evidence key scan (C15), not secrecy
  inside the container.
- A connection flood against `/ipc/fifi.sock` is a residual denial-of-service risk. The bridge
caps its own concurrent connections at 8, but the model can connect to the socket directly
(mode 0666 by design) and hold up to the container `--ulimit nofile=512:512` connections
(minus the standard descriptors), each an ssh channel on the host forward and a connection on
the Mac server, within the run deadline (`ACCB_DEADLINE` and `SSH_DEADLINE`, both 6600 s).
Accepted because it damages only availability of the same run: no second route, no data
exposure, no other port.
- `sandbox.py` cannot reliably tell docker's own exit codes 125 to 127 from a program that
itself exits 125 to 127: the docker client passes the container's exit code through, and its
own error marker on stderr ("docker: ...") is model-forgeable. The rerun therefore keeps
treating 125 to 127 as a sandbox error (exit 67); a hostile `hello.py` exiting 126 forces a
sandbox error instead of a quality miss. Documented and left as is.

How `pi` finds the server without editing `~/.pi` (checked on pi 1.0.0, installed docs under
`~/.pi/agent/install/releases/1.0.0/node_modules/@earendil-works/pi-coding-agent/docs/`):
- `environment-variables.md` line 81: `PI_CODING_AGENT_DIR` overrides the agent directory.
  `configuration.md` ("Agent directory") lists `models.json` and `settings.json` there. The launcher
  sets it to `<evidence dir>/pi/agent`. `pi --list-models` then lists only the provider `fifi`.
- `models.md` ("Configure a compatible endpoint"): a provider with `baseUrl`, `api`
  `openai-completions`, and `apiKey`. The key can be `$NAME`, so `models.json` holds
  `"apiKey": "$ENGINESRV_API_KEY"` and never the value.
- `docs/settings.md` ("Network and retries"): `httpIdleTimeoutMs` defaults to 300,000 ms and
  `retry.enabled` to true. The scratch `settings.json` sets `httpIdleTimeoutMs` 0 and retries off,
  because a prefill can last minutes and a retry would add a request.
- `pi-ai/dist/api/simple-options.js`: pi sends `max_tokens = min(maxTokens, contextWindow -
  estimated prompt - 4096)` with a minimum of 1. With `contextWindow` 2051 pi sends `max_tokens`
  1. The scratch `models.json` therefore declares `contextWindow` 6400 and `maxTokens` 256. The
  engine still enforces its own 2,051 positions and returns 409 `context_limit` above them.
- `pi` reads standard input in print mode. Run it with `< /dev/null`, or it waits.
- The command: `pi --offline -nc -ne -ns -np -na --no-session --mode json -p --thinking off
  --model fifi/<id> --system-prompt <text> --tools write,bash "<task>"`. `-nc` ignores context
  files, `-ne` ignores extensions, `-na` ignores project files. Provider traffic: `stream: true`
  and `stream_options.include_usage`, so the server accepts and ignores `stream_options`.
These points were exercised on this host with `fake_server.py --mode pi-tools` (`selftest.sh`).


## Server log contract

`enginesrv` writes one JSON line on standard error for each chat request that has a prompt
(`docs/design/server.md`, "Request log"; `engine/src/server/reqlog.rs`). The scripts send standard
error to `server.log` and standard output to `server.out`, and copy both into the evidence. The
checks read `server.log`. They need the keys `finish_reason` (ACC-A counts the lines that hold it)
and `prompt_tokens`, and every line must have `status` 200 and `outcome` `ok`. `testdata/reqlog-ok.jsonl` has the real shape, and `test_reqlog.py` compares its keys with `reqlog.rs`. Lines that are not JSON, such as startup errors, are skipped.

## Server config

`mac-acc-a.sh` writes one `config.json` (no key; only `api_key_env`). The engine entry uses
`id` and `model` `flash`, `family` `qwen3_8_flash_next`, `model_dir` and `tokenizer_dir` (the
artifact directory), and the object `flash_next` with `manifest` (`download-manifest.json`) and
`cache_limit_bytes`. The prefill chunk keeps its default. `EngineConfig` rejects any other key.
`GET /v1/models` then lists the id `flash`, which the client and `pi` use.

Sampling defaults come from the engine entry, not from a global constant: at start the server
reads `temperature`, `top_p`, and `top_k` from `generation_config.json` in the entry's
`model_dir` (an absent file or key falls back to 0.7, 1.0, and 0) and logs one line per engine
that quotes the file path and the effective values. `acc_a.py` and `smoke.py` set `temperature`
0.2 but not `top_p` or `top_k`, so the file's `top_p` 0.95 and `top_k` 20 now apply to the ACC
runs for the q36 engine. The ACC evidence records the effective defaults from the engine start
log line.

With `--engine q36` the engine entry has `id` `q36`, `family` `qwen3_6_35b_a3b`, `model`
`fifi-q36`, `model_dir` and `tokenizer_dir` `/Users/dan/edge0/models/edge0-35b`, and the object
`qwen36` copied as is from `engine/config/qwen36-mac-studio-36gb.json` in the cloned source tree of
the run (see "What the q36 engine serves"). `GET /v1/models` lists `fifi-q36`. The client gets
`--model fifi-q36 --engine q36`: it stops with a pipeline defect, before any chat request, when
`/v1/models` does not list that id. `result.json` and `launcher.txt` record the engine.

### What the q36 engine serves

`--engine q36` serves the configuration the user will run. `mac-acc-a.sh` reads the `qwen36` object of
`engine/config/qwen36-mac-studio-36gb.json` in the worktree that the run builds (`python3 -I`, never a
host copy) and writes it unchanged: chunked prefill on, `cache_limit_bytes` 4294967296,
`context_limit` 262144, `session_cap_bytes` 4294967296, `prefill_budget_bytes` 32212254720,
`prefill_session_reserve_bytes` 8286699520, `kv_dtype` q8, `compile` true, and no `prefill_chunk`,
so the engine default 256 applies. The script stops with status 65 when a key is missing or
`prefill_chunk` is present. Only `listen`, `api_key_env`, `id`, `model`, `model_dir` and `tokenizer_dir`
come from the harness. `Q36_CACHE_LIMIT` cannot change the served object: a value that differs from
the shipped file stops the run with status 64, from the environment or from `env.sh`. `evidence/identity.txt`
records `shipped_config_sha256` and the object, and `evidence/config.json` is the served file. ACC-A stays
structural. `timings.tsv` reports total seconds per request and the first delta of the streamed request,
and no gate uses them. `selftest-q36-config.sh` checks these rules and `mutate-q36-config.sh` kills
seven mutations of them.

### Q36 checkpoint pins

`mac-acc-a.sh` holds the checkpoint path as the constant `Q36_PINNED_DIR`
(`/Users/dan/edge0/models/edge0-35b`) and reads the expected hashes from `q36-pins.sha256`
(`config.json`, `model.safetensors.index.json`, `tokenizer.json`, `chat_template.jinja`). It hashes
the four files on the Mac before the build and before the server starts. A mismatch or a missing
file prints `Q36 PIN MISMATCH: <file>` and stops with status 88; the launcher passes 88 on. On a
match, `evidence/identity.txt` records `engine q36`, `model_id`, `model_dir`, the four
`artifact_file <name> <sha256>` lines, and `cache_limit_bytes`. The pin file holds hashes of files
that are not in this repository; `selftest.sh` checks the four values and the path. Only a
rehearsal reads `Q36_DIR_OVERRIDE` and `Q36_PINS_OVERRIDE`; a real run ignores them. `mac-acc-a.sh` also repeats the port 8788 refusal and the `ENGINE` check after it sources `env.sh`, and keeps `REHEARSE` as it was before, so `env.sh` cannot reach port 8788, pick an unknown engine, or switch on the overrides. A Q36 pass is a
stepping-stone result, not the primary requirement (Flash-Next on the 36 GB Mac Studio).

## Limits

- The fake server and the unit tests prove the client, not the engine.
- The Mac runs no model code: it builds and serves `enginesrv` only. The abort paths of the real
  (non-rehearsal) launchers share their code with the rehearsal, but no Mac, ssh, or flock run
  tested them.
- The fake docker proves the command line and the flow. `test_sandbox_real.py` proves the
  isolation of the real docker on the pinned image.
- Timings are measured facts of one run. They are not benchmarks.
- A model quality miss with a clean parse is recorded as exit status 3, not hidden.
- Port 8788 (laya3) is refused by every script and by the client.
