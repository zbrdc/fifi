"""NVML ctypes tests against a Python fake; never load a driver or query a GPU."""

import ctypes
from dataclasses import replace
import unittest

import p2p_test_support  # noqa: F401 -- make repository packages visible under python -I

from scripts.p2p_pair_host import nvml

GPU0 = "GPU-704fef60-5280-9b88-fa3d-d11135caf576"
GPU1 = "GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a"
MIB = 1024 * 1024
BOOT = "7b2415a1-fab1-4aa0-b52b-c1bce142f145"
OTHER_BOOT = "9b2415a1-fab1-4aa0-b52b-c1bce142f145"


class Function:
    def __init__(self, body):
        self.body = body
        self.argtypes = self.restype = None

    def __call__(self, *args):
        return self.body(*args)


class FakeNvml:
    def __init__(self):
        self.uuids = {1: GPU0, 2: GPU1}
        self.handles = {GPU0: 1, GPU1: 2}
        self.compute = {1: [111], 2: [111]}
        self.graphics = {1: [], 2: []}
        self.mps = {1: [], 2: []}
        self.memory = {1: (12288, 3288, 9000), 2: (10240, 2240, 8000)}
        self.errors = {}
        self.shutdowns = 0
        self.queries = []
        self.nvmlInit_v2 = Function(lambda: self.errors.get("init", 0))
        self.nvmlShutdown = Function(self.shutdown)
        self.nvmlDeviceGetHandleByUUID = Function(self.handle)
        self.nvmlDeviceGetUUID = Function(self.uuid)
        self.nvmlDeviceGetMemoryInfo = Function(self.mem)
        self.nvmlDeviceGetComputeRunningProcesses_v3 = Function(
            lambda handle, count, array: self.processes("compute", handle, count, array)
        )
        self.nvmlDeviceGetGraphicsRunningProcesses_v3 = Function(
            lambda handle, count, array: self.processes(
                "graphics", handle, count, array
            )
        )
        self.nvmlDeviceGetMPSComputeRunningProcesses_v3 = Function(
            lambda handle, count, array: self.processes("mps", handle, count, array)
        )

    def shutdown(self):
        self.shutdowns += 1
        return self.errors.get("shutdown", 0)

    def handle(self, uuid, pointer):
        if self.errors.get("handle"):
            return self.errors["handle"]
        value = self.handles.get(uuid.decode(), 0)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            value
        )
        return 0

    def uuid(self, handle, output, size):
        if self.errors.get("uuid"):
            return self.errors["uuid"]
        value = self.uuids[handle.value].encode() + b"\x00"
        self.assert_capacity(size, len(value))
        ctypes.memmove(output, value, len(value))
        return 0

    @staticmethod
    def assert_capacity(size, needed):
        if size < needed:
            raise AssertionError("fake UUID output buffer too small")

    def mem(self, handle, pointer):
        if self.errors.get("memory"):
            return self.errors["memory"]
        total, free, used = self.memory[handle.value]
        data = ctypes.cast(pointer, ctypes.POINTER(nvml._Memory)).contents
        data.total, data.free, data.used = total * MIB, free * MIB, used * MIB
        return 0

    def processes(self, kind, handle, pointer, array):
        self.queries.append((kind, handle.value, array is None))
        if self.errors.get(kind):
            return self.errors[kind]
        pids = getattr(self, kind)[handle.value]
        count = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))
        if array is None:
            count.contents.value = len(pids)
            return 7 if pids else 0
        if count.contents.value < len(pids):
            count.contents.value = len(pids)
            return 7
        for index, pid in enumerate(pids):
            array[index].pid = pid
        count.contents.value = len(pids)
        return 0


def fake_sample(lib=None):
    lib = lib or FakeNvml()
    clock = iter((100, 200))
    return nvml.NvmlReader(
        lib, (GPU0, GPU1), boot_id_reader=lambda: BOOT, clock_ns=lambda: next(clock)
    ).sample()


def stopped_from(base, *, started=201):
    def empty(item):
        return replace(
            item,
            compute_pids=(),
            graphics_pids=(),
            mps_pids=(),
            free_bytes=item.total_bytes,
            used_bytes=0,
        )

    return nvml.PairSample(
        base.boot_id, started, started + 1, tuple(empty(x) for x in base.devices)
    )


class NvmlReaderTests(unittest.TestCase):
    def test_complete_fake_sample_queries_all_classes_and_shutdowns(self):
        lib = FakeNvml()
        sample = fake_sample(lib)
        self.assertEqual((sample.started_ns, sample.finished_ns), (100, 200))
        self.assertEqual([item.uuid for item in sample.devices], [GPU0, GPU1])
        self.assertEqual(sample.devices[0].compute_pids, (111,))
        self.assertEqual(sample.devices[1].total_bytes, 10240 * MIB)
        self.assertEqual(lib.shutdowns, 1)
        self.assertTrue(nvml.active_pair_matches(sample, (GPU0, GPU1), 111, BOOT))
        self.assertEqual(
            {kind for kind, _, _ in lib.queries}, {"compute", "graphics", "mps"}
        )

    def test_unknown_permission_unsupported_and_query_errors_are_inconclusive(self):
        for kind in (
            "handle",
            "uuid",
            "compute",
            "graphics",
            "mps",
            "memory",
            "shutdown",
        ):
            for code in (3, 4, 999):
                with self.subTest(kind=kind, code=code):
                    lib = FakeNvml()
                    lib.errors[kind] = code
                    with self.assertRaises(nvml.NvmlInconclusive):
                        fake_sample(lib)
                    self.assertEqual(lib.shutdowns, 1)
        lib = FakeNvml()
        lib.errors["init"] = 4
        with self.assertRaises(nvml.NvmlInconclusive):
            fake_sample(lib)
        self.assertEqual(lib.shutdowns, 0)

    def test_missing_v3_abi_is_not_silently_downgraded(self):
        lib = FakeNvml()
        del lib.nvmlDeviceGetMPSComputeRunningProcesses_v3
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.NvmlReader(lib, (GPU0, GPU1), boot_id_reader=lambda: BOOT)
        self.assertEqual(lib.shutdowns, 0)

    def test_binding_exceptions_are_inconclusive_and_hide_details(self):
        lib = FakeNvml()

        def explode(*_args):
            raise OSError("sensitive fake driver text")

        lib.nvmlDeviceGetMemoryInfo = Function(explode)
        with self.assertRaises(nvml.NvmlInconclusive) as failure:
            fake_sample(lib)
        self.assertNotIn("sensitive", str(failure.exception))
        self.assertEqual(lib.shutdowns, 1)
        lib = FakeNvml()
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.NvmlReader(lib, (GPU0, GPU1), boot_id_reader=explode).sample()
        self.assertEqual(lib.shutdowns, 0)

    def test_bad_uuid_missing_and_aliased_handles_rejected(self):
        lib = FakeNvml()
        lib.uuids[1] = GPU1
        with self.assertRaises(nvml.NvmlInconclusive):
            fake_sample(lib)
        lib = FakeNvml()
        lib.handles[GPU1] = 1
        with self.assertRaises(nvml.NvmlInconclusive):
            fake_sample(lib)
        lib = FakeNvml()
        lib.handles[GPU0] = 0
        with self.assertRaises(nvml.NvmlInconclusive):
            fake_sample(lib)
        with self.assertRaises(ValueError):
            nvml.NvmlReader(FakeNvml(), (GPU0, GPU0), boot_id_reader=lambda: BOOT)

    def test_invalid_memory_and_process_pids_rejected(self):
        for memory in ((0, 0, 0), (100, 101, 0), (100, 50, 49)):
            lib = FakeNvml()
            lib.memory[1] = memory
            with self.assertRaises(nvml.NvmlInconclusive):
                fake_sample(lib)
        for pids in ([0], [111, 111], list(range(1, 4098))):
            lib = FakeNvml()
            lib.compute[1] = pids
            with self.assertRaises(nvml.NvmlInconclusive):
                fake_sample(lib)

    def test_process_list_growth_is_retried_without_truncation(self):
        lib = FakeNvml()
        original = lib.nvmlDeviceGetComputeRunningProcesses_v3.body
        filled = 0

        def grow(handle, pointer, array):
            nonlocal filled
            if handle.value != 1 or array is None:
                return original(handle, pointer, array)
            filled += 1
            if filled == 1:
                count = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))
                count.contents.value += 1
                return 7
            return original(handle, pointer, array)

        lib.nvmlDeviceGetComputeRunningProcesses_v3 = Function(grow)
        result = fake_sample(lib)
        self.assertEqual(result.devices[0].compute_pids, (111,))
        self.assertEqual(filled, 2)

    def test_unsupported_or_unbounded_process_count_never_becomes_empty(self):
        lib = FakeNvml()

        def huge(_handle, pointer, _array):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint)).contents.value = 4097
            return 7

        lib.nvmlDeviceGetMPSComputeRunningProcesses_v3 = Function(huge)
        with self.assertRaises(nvml.NvmlInconclusive):
            fake_sample(lib)
        self.assertEqual(lib.shutdowns, 1)

    def test_context_change_and_unexpected_owners_fail(self):
        for kind in ("compute", "graphics", "mps"):
            lib = FakeNvml()
            getattr(lib, kind)[1].append(222)
            with self.assertRaises(nvml.NvmlInconclusive):
                nvml.active_pair_matches(fake_sample(lib), (GPU0, GPU1), 111, BOOT)
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.active_pair_matches(fake_sample(), (GPU1, GPU0), 111, BOOT)

    def test_stopped_headroom_boundaries_and_uuid_identity(self):
        baseline = fake_sample()
        empty = stopped_from(baseline)
        self.assertTrue(
            nvml.stopped_pair_has_headroom(empty, baseline, (GPU0, GPU1), 111, BOOT)
        )
        for index, minimum_mib in ((0, 10024), (1, 9024)):
            items = list(empty.devices)
            total = items[index].total_bytes
            free = minimum_mib * MIB
            items[index] = replace(
                items[index], free_bytes=free, used_bytes=total - free
            )
            self.assertTrue(
                nvml.stopped_pair_has_headroom(
                    replace(empty, devices=tuple(items)),
                    baseline,
                    (GPU0, GPU1),
                    111,
                    BOOT,
                )
            )
            items[index] = replace(
                items[index], free_bytes=free - 1, used_bytes=total - free + 1
            )
            with self.assertRaises(nvml.NvmlInconclusive):
                nvml.stopped_pair_has_headroom(
                    replace(empty, devices=tuple(items)),
                    baseline,
                    (GPU0, GPU1),
                    111,
                    BOOT,
                )
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.stopped_pair_has_headroom(
                replace(empty, devices=empty.devices[::-1]),
                baseline,
                (GPU0, GPU1),
                111,
                BOOT,
            )
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.stopped_pair_has_headroom(
                replace(empty, started_ns=100), baseline, (GPU0, GPU1), 111, BOOT
            )
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.stopped_pair_has_headroom(empty, baseline, (GPU0, GPU1), 222, BOOT)
        item = baseline.devices[0]
        no_usage = replace(item, used_bytes=0, free_bytes=item.total_bytes)
        invalid_baseline = replace(baseline, devices=(no_usage, baseline.devices[1]))
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.stopped_pair_has_headroom(
                empty, invalid_baseline, (GPU0, GPU1), 111, BOOT
            )

    def test_directly_constructed_samples_do_not_bypass_invariants(self):
        baseline = fake_sample()
        item = baseline.devices[0]
        for bad in (
            replace(item, compute_pids=(111, 111)),
            replace(item, free_bytes=-1),
            replace(item, total_bytes=0),
            replace(item, graphics_pids=(True,)),
        ):
            sample = replace(baseline, devices=(bad, baseline.devices[1]))
            with self.assertRaises(nvml.NvmlInconclusive):
                nvml.active_pair_matches(sample, (GPU0, GPU1), 111, BOOT)
        with self.assertRaises(nvml.NvmlInconclusive):
            nvml.active_pair_matches(
                replace(baseline, started_ns=True), (GPU0, GPU1), 111, BOOT
            )

    def test_clock_must_be_valid_and_monotonic(self):
        for values in ((True, 2), (10, 9), (-1, 2)):
            lib = FakeNvml()
            clock = iter(values)
            with self.assertRaises(nvml.NvmlInconclusive):
                nvml.NvmlReader(
                    lib,
                    (GPU0, GPU1),
                    boot_id_reader=lambda: BOOT,
                    clock_ns=lambda: next(clock),
                ).sample()

    def test_boot_identity_must_remain_stable_and_match_baseline(self):
        lib = FakeNvml()
        boots = iter((BOOT, OTHER_BOOT))
        clock = iter((100, 200))
        with self.assertRaisesRegex(nvml.NvmlInconclusive, "boot identity changed"):
            nvml.NvmlReader(
                lib,
                (GPU0, GPU1),
                boot_id_reader=lambda: next(boots),
                clock_ns=lambda: next(clock),
            ).sample()
        self.assertEqual(lib.shutdowns, 1)
        baseline = fake_sample()
        stopped = replace(stopped_from(baseline), boot_id=OTHER_BOOT)
        with self.assertRaisesRegex(nvml.NvmlInconclusive, "expected boot"):
            nvml.active_pair_matches(baseline, (GPU0, GPU1), 111, OTHER_BOOT)
        with self.assertRaisesRegex(nvml.NvmlInconclusive, "different boots"):
            nvml.stopped_pair_has_headroom(stopped, baseline, (GPU0, GPU1), 111, BOOT)
        with self.assertRaisesRegex(nvml.NvmlInconclusive, "boot identity"):
            nvml.NvmlReader(
                FakeNvml(), (GPU0, GPU1), boot_id_reader=lambda: "bad"
            ).sample()


if __name__ == "__main__":
    unittest.main()
