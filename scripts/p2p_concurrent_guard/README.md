# Experimental concurrent supervisor core

This is a CPU-only sequencing prototype. It has no real host backend, runnable
supervisor entry point, service integration, or permission to use GPUs. It does not
attest any actual process, endpoint, lease, scope, configuration, or service,
and it does not prove that a server-side request has drained. It cannot be used
to run a campaign.

The intended future adapter must retain the existing shared lease through
verified cleanup, and pin the secondary service identity (including process
start time, executable, unit, configuration and GPU ownership). It must contain
the companion client, prove request completion and server-side drain, prevent
unapproved service handoffs, and restore and verify the pair before release.
The adapter must enforce finite waits and keep the lease alive during manual
recovery. The core cannot establish any of those host properties itself.

An opt-in, harmless scope test checks descriptor inheritance and independent
contention on a fresh temporary lock. A future host adapter must still inspect
the inherited descriptor's inode and prove independent contention on the
canonical lock before sending any HTTP request. It must not unlock the
inherited descriptor. CPU fakes and a real, harmless scope test are both
required; a successful parent `pass_fds` call alone does not show that the
descriptor survived the service-manager scope.

The core only tests whether two supplied request time intervals overlap. This
does not prove that GPU kernels were simultaneously active, that requests
completed correctly, or that a performance measurement is uncontaminated.
Any future concurrent result must be labeled and compared against separately
measured isolated workloads.

The unit tests inject fake operations and failures. They verify sequencing and
lease-retention decisions only; they are not service, GPU, or end-to-end tests.

## Host components under development

The gate checks a held lease and sealed, digest-pinned companion image, waits
for an explicit parent pipe signal before exec, and rechecks the lease and
image. It accepts typed bounded options and a private task-owned output root;
it does not accept key, model, endpoint, lease-descriptor, or prompt overrides.
The gate itself is pinned and sealed. The parent still must attest its scope,
PID/start time, inherited descriptors and waiting state before signalling.
The gate samples directory identity and permissions before and after that
wait. This is not an openat-based fence against later path replacement; the
future host adapter must provide independently reviewed output containment.
An opt-in disposable-scope CPU test exercises the sealed gate and a sealed
no-network fixture with a temporary lock; it checks that the fixture does not
execute until the parent signals. It does not attest any live service or GPU.

The adapter currently accepts only an explicitly designated test backend.
There is no real pair stop/restore, client ownership or request-drain backend,
and no runnable supervisor entry point. A fake backend is not a security
boundary or live attestation. In particular, an inconclusive core result must
not let a real supervisor exit and close its lease without verified recovery.

The pair request-window helper records monotonic nanoseconds around a completed
request and requires exact synthetic response, expected token count, fixed
payload digest and explicit zero reuse. A caller must ensure the provided send
function actually sends the supplied immutable bytes and bind the producer,
boot, clock, scopes and artifacts. Client-observed request windows may overlap
while HTTP transfer or GPU activity does not. The companion uses nonce prompts,
so its response hashes do not establish matched inputs or outputs. The
separate reviewed overlap validator remains external to this prototype.

System Python `/usr/bin/python3` is needed on this host for memfd sealing;
the default Conda interpreter does not expose `os.memfd_create`. No adapter
component should be used against a live service until the missing backend,
provenance, drain, error/interrupt recovery and independent review are complete.
