# Guarded 2080 Ti handoff (host-specific)

This cooperative command acquires the established `gpu.lock` before it can stop
`fifi-qwen36-2080ti.service`, runs one trusted command on the 2080 Ti in a
nondelegated systemd scope, and verifies the serving process before releasing
the lock. **It is not installed or enabled by this change.** Its live path is
not qualified until the service owner adopts and tests its startup attestation.
The current owner-launched service is unattested, so this tool refuses it before
any stop. Do not use it for a live GPU handoff until the owner and coordinator
have completed that adoption and a separately admitted smoke test.

Before considering adoption, the service owner must review the exact fixed host
paths and the launcher and sourced `build-env.sh` hashes in
`scripts/gpu_guarded_handoff/{host.py,attested_launch.py}`. The owner may then
configure the user unit to execute `/usr/bin/python3` followed by the absolute
path of **the same checkout's** `scripts/gpu_guarded_handoff/attested_launch.py`.
That wrapper records startup hashes and its PID/start time before it execs the
existing `start.sh`. A launcher or sourced environment change needs a new review
and updated approved hashes; the guard refuses an unknown version. Any change
to the live unit or restart remains the service owner's coordinated action.

At preparation time, the owner's `start.sh` had already changed from the
reviewed hash compiled into this tool. The launcher would therefore reject
that file. **Do not point the live unit at this wrapper or restart it** until
the owner has reviewed the current launcher and its complete sourced inputs,
updated the approved hashes, and validated a separate adoption plan. Simply
replacing a hash without reviewing the changed script is not an adoption.

From the same checkout, after the owner has adopted the attested launcher,
choose an existing task-owned directory under `/home/ben/fifi-scratch`, declare
free disk and peak GPU memory, and provide a trusted test command after `--`:

```sh
python3 -m scripts.gpu_guarded_handoff \
  --scratch-dir /home/ben/fifi-scratch/BASE/TASK/tmp \
  --required-free-mib 2048 --min-disk-mib 20480 --timeout-seconds 600 \
  -- /absolute/path/to/test-binary --test-threads=1
```

Replace `BASE`, `TASK`, the memory/disk bounds, timeout, and test command with
actual task values. The command sets a UUID-based `CUDA_VISIBLE_DEVICES`,
task-local temporary paths, and `INFERENCE_CUDA_REQUIRE_DEVICE=1`. It writes
`outcome.json` under a newly created campaign directory. A zero workload exit
and a restored service do **not** prove that any tests ran or that required
artifacts or numerical checks passed; the caller must verify those separately.

The owner and coordinator must establish a quiet window first. The guard rejects
all established 2080 service connections, including router polling, and checks
service and GPU ownership, both pair GPUs' utilization, foreign builders, and
tracked assistant children. These checks are sampled: a new client or job can
arrive between observations, and another Codex action may not create a directly
observable child process. They do not fence incoming requests. The old
user-confirmed router-poll exception is not implemented here.
The monitor reads process and device state repeatedly and can perturb a timing
run. Use the same guard settings in each comparison arm and measure monitoring
overhead separately before attributing small performance differences.

The current adapter supports only units without `EnvironmentFiles` or inline
`Environment`, `PassEnvironment`, or `UnsetEnvironment` overrides. At startup
the attestor checks that the inherited manager environment matches a single
manager snapshot, then records its fingerprint. The handoff checks that the
manager and loaded-unit environment have not changed before a stop or restart.
These checks are sampled and still cannot prevent a same-user change after the
last observation. Unsupported or unverifiable environment inputs fail closed.

At startup the owner’s attested launcher records its input hashes and process
identity. The handoff checks those against the running process, pins the live
executable inode, unit files and relevant launch inputs, and compares the
process environment privately after restart. Environment values and the API key
are not written to the receipt. This is a cooperative pre-exec attestation: it
does not prove the bytes later parsed in an adversarial same-user file race,
attest the model/tokenizer contents or dynamic libraries, or preserve in-memory
sessions and prefix caches. Restarting the service discards that in-memory
state. It verifies only the pinned binary and launch inputs, model/context
readiness, device ownership, and environment under the documented assumptions.

If workload cleanup or restoration cannot be verified, the supervisor reports
`inconclusive` and keeps the lease while it waits for the scope to empty and
the pinned service to become ready. Do not kill the supervisor while it holds
an unresolved lease; coordinate recovery with the service owner. `SIGKILL`,
process death, system failure, or another same-user caller using `systemctl`
directly can bypass or end this protection. The lease is advisory, not an
access-control boundary. A reliable hard fence requires a separately designed
service-control broker or permissions separation.
