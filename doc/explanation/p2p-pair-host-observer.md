# P2P pair host observer: experimental read-only slice

`LinuxPairObserver` samples the running pair service without starting or
stopping it. It pins the loaded unit and process identity, fingerprints
explicitly supplied launcher/configuration/environment/key and unit files,
checks the process `--config` argument, and compares two snapshots around
three idle GPU samples. It refuses active client sockets and requires the
pair service process to own the reported listening socket and all reported
compute-app contexts on the specified pair. Its test inputs are synthetic;
there is no executable entry point or live admission integration.

This slice **cannot authorize a service stop or a concurrent campaign**. The
caller must prove the canonical held lease, engine readiness and request queue,
all GPU contexts including graphics, companion and scope identity, drain,
restoration inputs and lifecycle containment. Fingerprinted environment and
key paths are supplied by the caller and are not proof that the launcher reads
those files. The observer does not enumerate model or restoration dependencies.
Samples do not fence future requests or direct same-user service changes, and
the code intentionally refuses router-poll connections. Because a snapshot
repeatedly hashes the running executable and waits between utilization samples,
it is not a per-second performance monitor.

An independent read-only check found that this host's `systemctl show --all`
omits several empty hook/environment properties, while direct systemd D-Bus
queries report typed empty arrays. For those known omitted fields the observer
resolves the unit with `GetUnit` and requires the exact expected empty D-Bus
signature; it rejects nonempty, unsupported, missing or changing proofs. This
does not treat missing fields as empty. No live observer, service handoff, GPU
workload or 2080 integration has been run. CPU-only fake-command/proc tests do
not qualify live behavior.

The companion `PairLifecycle` module is a **test-gated sequence**, not a live
service controller. It accepts only explicitly marked fake backends and
runners, never closes the supplied lease, and does not make systemd calls on
its own. It plans only the fixed pair service's stop and start commands, marks
attempts before invoking the fake runner, and requires external proof of the
unchanged secondary, queue idleness, contexts, sockets, restoration inputs and
replacement start provenance. Each injected operation receives an absolute
monotonic deadline and must enforce it internally; post-call checks cannot
interrupt a blocking implementation. A requested start is not evidence that
the resulting process came from that start.

For the fake restoration admission the two supplied UUIDs must match the
baseline, total memory must accommodate the earlier used amount plus 1,024 MiB,
and sampled free memory must be at least the larger of 8,192 MiB and earlier
used plus 1,024 MiB. This is a conservative sampled gate, not a guarantee of
future allocation or a claim about fragmentation. Uncertainty must retain the
actual lease under an independently supervised recovery process; this module
cannot do that on its own. No production or live behavior is qualified.

A separate ctypes NVML reader takes an already loaded library object and never
loads the driver implicitly. It queries the two requested UUID handles and
verifies the returned UUIDs, requires successful v3 compute, graphics and MPS
process lists, and reads byte-valued memory totals/free/used. It rejects
missing symbols, unsupported/permission errors, inconsistent memory, unstable
or oversized process lists and aliased handles as inconclusive. There is no
MPS-disabled fallback: unsupported MPS also fails closed. The optional pure
headroom check requires a prior matching active pair and both later sampled
GPUs to show no listed contexts and at least the configured 1-GiB restoration
margin. The injected boot-ID reader must be trusted; its value is checked
before and after sampling and bound to both compared samples. The supervisor
must separately revalidate that the numeric PID has the pinned process start
ticks and service identity before and after the sample; matching a PID alone
does not establish process identity.

The ctypes reader and helpers have been exercised **only against a fake Python
library**. They have not loaded the real NVML library or queried any GPU; ABI,
library provenance, platform support and actual integration remain unverified.
The categories are sampled sequentially and cannot exclude contexts appearing
between calls, unreported driver allocations, or same-user lock bypass. They do
not establish an exclusive device reservation or guarantee later restoration.

`ScopedUnit` is an unintegrated scope-identity and drain observation slice for
a future guarded pair runner. It pins a unique nondelegated scope's invocation
and cgroup; signals are preceded by an exact identity sample. It treats cgroup
permission and device errors as inconclusive, and considers a collected scope
drained only with explicit unit-not-found, stat-confirmed cgroup absence and a
matching second query. Omitted properties are never treated as evidence for a
loaded scope. Unit identity checks are sampled rather than a same-UID fence;
filesystem reads also lack an independently enforced absolute deadline.

Only fake-manager/temp-filesystem tests have run. This host's actual
`systemctl show` output and exit behavior for an auto-collected scope have not
been captured. The slice has no service controller, child launcher, result
drain, lease keeper or live entry point. It must not be used to qualify a real
handoff without a reviewed, bounded controller and disposable scope tests.
