import ctypes
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_host_gpu_resources as observer


UUID = 'GPU-704fef60-5280-9b88-fa3d-d11135caf576'


def proc_stat(user, system, start, name='worker (main)'):
    fields = ['S'] + ['0'] * 19
    fields[11], fields[12], fields[19] = str(user), str(system), str(start)
    return '123 (' + name + ') ' + ' '.join(fields)


class FakeNvml:
    def __init__(self, uuids):
        self.uuids, self.closed = uuids, False

    def sample(self, uuid):
        return {'uuid': uuid, 'pcie_tx_kb_per_s': 10, 'errors': {}}

    def close(self):
        self.closed = True


class ResourceObserverTests(unittest.TestCase):
    def test_proc_parser_handles_parentheses_and_rejects_bad_fields(self):
        self.assertEqual(observer.parse_proc_stat(proc_stat(17, 9, 55)),
                         observer.CpuTicks(55, 17, 9, 'worker (main)'))
        for text in ('no pid', '123 (x) S', proc_stat(-1, 2, 3)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                observer.parse_proc_stat(text)

    def test_system_cpu_excludes_guest_double_count(self):
        self.assertEqual(observer.parse_system_stat('cpu 100 10 20 30 40 5 2 3 900 900\n'), (210, 70))

    def test_cpu_rates_match_process_and_persistent_threads(self):
        old = observer.CpuSnapshot(observer.CpuTicks(1, 100, 50),
            {10: observer.CpuTicks(10, 50, 20), 11: observer.CpuTicks(11, 5, 5)}, 1000, 500, 1_000_000_000)
        new = observer.CpuSnapshot(observer.CpuTicks(1, 300, 100),
            {10: observer.CpuTicks(10, 210, 20), 11: observer.CpuTicks(99, 200, 20),
             12: observer.CpuTicks(12, 100, 20)}, 2000, 1000, 3_000_000_000)
        result = observer.cpu_delta(old, new, 100)
        self.assertEqual(result['process_user_cores'], 1.0)
        self.assertEqual(result['process_system_cores'], 0.25)
        self.assertEqual(result['process_total_cores'], 1.25)
        self.assertEqual(result['busiest_thread_cores'], 0.8)
        self.assertEqual(result['busiest_thread_id'], 10)
        self.assertEqual(result['matched_threads'], 1)
        self.assertEqual(result['system_busy_fraction'], 0.5)
        with self.assertRaises(ValueError):
            observer.cpu_delta(old, observer.CpuSnapshot(observer.CpuTicks(2, 300, 100), {}, 2000, 1000, 3_000_000_000), 100)

    def test_unsupported_nvml_counter_is_not_reported_as_zero(self):
        class Library:
            def nvmlDeviceGetUtilizationRates(self, handle, value):
                value._obj.gpu, value._obj.memory = 40, 20
                return 0

            def nvmlDeviceGetMemoryInfo(self, handle, value):
                value._obj.total, value._obj.free, value._obj.used = 100, 40, 60
                return 0

            def nvmlDeviceGetPcieThroughput(self, handle, direction, value):
                value._obj.value = 123
                return 3 if direction == 0 else 0

            def nvmlDeviceGetCurrPcieLinkGeneration(self, handle, value):
                value._obj.value = 4
                return 0

            def nvmlDeviceGetCurrPcieLinkWidth(self, handle, value):
                value._obj.value = 8
                return 0

        nvml = observer.Nvml.__new__(observer.Nvml)
        nvml.lib, nvml.handles = Library(), {UUID: ctypes.c_void_p(1)}
        sample = nvml.sample(UUID)
        self.assertIsNone(sample['pcie_tx_kb_per_s'])
        self.assertEqual(sample['errors']['pcie_tx_kb_per_s'], 3)
        self.assertEqual(sample['pcie_rx_kb_per_s'], 123)
        self.assertLessEqual(sample['pcie_rx_kb_per_s_started_monotonic_ns'],
                             sample['pcie_rx_kb_per_s_finished_monotonic_ns'])
        self.assertEqual(sample['gpu_busy_percent'], 40)

    def test_cpu_only_driver_writes_valid_samples_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as td, patch.object(observer, 'Nvml', FakeNvml):
            output = Path(td) / 'resources.jsonl'
            self.assertEqual(observer.run(os.getpid(), [UUID], 0.25, 2, output), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row['type'] for row in rows], ['metadata', 'sample', 'sample', 'end'])
            self.assertIsNone(rows[1]['cpu'])
            self.assertGreater(rows[2]['cpu']['interval_seconds'], 0)
            self.assertEqual(rows[-1]['reason'], 'sample_limit')
            original = output.read_text()
            with self.assertRaises(FileExistsError):
                observer.run(os.getpid(), [UUID], 0.25, 1, output)
            self.assertEqual(output.read_text(), original)

    def test_invalid_arguments_are_rejected_before_nvml(self):
        with patch.object(observer, 'Nvml') as nvml:
            for args in ((0, [UUID], 1.0, 1), (1, [UUID, UUID], 1.0, 1),
                         (1, ['GPU-invalid'], 1.0, 1), (1, [UUID], 0.01, 1)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    observer.run(*args, Path('/not-created'))
            nvml.assert_not_called()

    def test_signal_flushes_a_termination_event(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'resources.jsonl'
            code = '''import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import bench_host_gpu_resources as observer
class Fake:
    def __init__(self, uuids): pass
    def sample(self, uuid): return {"uuid": uuid, "errors": {}}
    def close(self): pass
observer.Nvml = Fake
raise SystemExit(observer.run(os.getpid(), [sys.argv[2]], 0.25, 0, Path(sys.argv[3])))
'''
            process = subprocess.Popen([sys.executable, '-c', code, str(Path(observer.__file__).parent),
                                        UUID, str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and (not output.exists() or output.stat().st_size == 0):
                time.sleep(0.02)
            self.assertTrue(output.exists(), 'observer did not start')
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[-1]['type'], 'end')
            self.assertEqual(rows[-1]['reason'], 'signal')

    def test_write_event_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            observer.write_event(io.StringIO(), {'bad': float('nan')})


if __name__ == '__main__':
    unittest.main()
