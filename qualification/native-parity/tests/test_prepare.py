import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if not os.environ.get('TMPDIR'):
    sys.exit('set TMPDIR to the task scratch directory')

PREPARE = Path(__file__).resolve().parents[1] / 'prepare.py'

def checkpoint_fixture(root, defect=None):
    import ml_dtypes
    import numpy as np
    from safetensors.numpy import save_file
    config = {'quantization': {'mode': 'affine', 'bits': 4, 'group_size': 64,
                              'gate': {'mode': 'affine', 'bits': 8, 'group_size': 64}}}
    (root/'config.json').write_text(json.dumps(config))
    tensors = {}
    for module, shape, sidecar in [('dense', (2, 16), (2, 2)), ('experts', (2, 2, 16), (2, 2, 2)), ('gate', (1, 16), (1, 1))]:
        tensors[module+'.weight'] = np.zeros(shape, dtype=np.uint32)
        tensors[module+'.scales'] = np.zeros(sidecar, dtype=ml_dtypes.bfloat16)
        tensors[module+'.biases'] = np.full(sidecar, -1, dtype=ml_dtypes.bfloat16)
    if defect == 'shape':
        tensors['dense.scales'] = np.zeros((1, 2), dtype=ml_dtypes.bfloat16)
    if defect == 'nonfinite':
        tensors['dense.scales'][:] = float('inf')
    if defect == 'negative':
        for key in tensors:
            if key.endswith('.biases'):
                tensors[key][:] = 0
    save_file(tensors, str(root/'model.safetensors'))
    if defect == 'duplicate':
        (root/'config.json').write_text('{"quantization":{},"quantization":{}}')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key:'model.safetensors' for key in tensors}}))
    files = []
    for name in ['config.json', 'model.safetensors.index.json', 'model.safetensors']:
        data = (root/name).read_bytes()
        files.append({'path':name, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
    (root/'fixture-source.json').write_text(json.dumps({'files':files, 'group_count':1000 if defect == 'insufficient' else 4}))


class PreparationCommand(unittest.TestCase):
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'owner.txt'
            marker.write_text('previous run')
            result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root/'absent'), '--output', str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('output already exists', result.stderr)
            self.assertEqual(marker.read_text(), 'previous run')

    def test_changed_source_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_bytes(b'{}')
            (root / 'fixture-source.json').write_text(json.dumps({'files': [{'path': 'config.json', 'bytes': 2, 'sha256': '0'*64}], 'group_count': 1}))
            output = root / 'prepared'
            result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root), '--output', str(output), '--fixture'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('source digest mismatch: config.json', result.stderr)
            self.assertFalse(output.exists())

    @unittest.skipUnless(sys.platform == 'darwin', 'positive preparation requires pinned Mac MLX')
    def test_prepares_distinct_real_slices_with_independent_reference_outputs(self):
        import ml_dtypes
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root)
            output = root/'prepared'
            result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root), '--output', str(output), '--fixture'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            capture = json.loads(result.stdout)
            data = (output/'manifest.json').read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), capture['manifest_sha256'])
            manifest = json.loads(data)
            self.assertTrue(manifest['fixture'])
            self.assertFalse(capture['qualified'])
            self.assertEqual(manifest['negative_biases'], 4)
            self.assertEqual(manifest['excluded_int8'], ['gate'])
            expected = [('dense', [1, 1]), ('experts', [0, 0, 1]), ('dense', [0, 0]), ('experts', [1, 0, 0])]
            self.assertEqual([(g['module'], g['coordinates']) for g in manifest['groups']], expected)
            for group in manifest['groups']:
                with safe_open(str(output/group['weight']['path']), framework='numpy') as weights:
                    self.assertEqual(weights.get_tensor('weight').tobytes(), bytes(32))
                    self.assertEqual(weights.get_tensor('biases').tobytes(), bytes.fromhex('80bf'))
                with safe_open(str(output/group['reference']['path']), framework='numpy') as reference:
                    self.assertEqual(reference.get_tensor('basis').tolist(), [[-1.0]]*64)
                    self.assertEqual(reference.get_tensor('dense').tolist(), [[-5.0]])

    @unittest.skipUnless(sys.platform == 'darwin', 'preparation inventory uses pinned packages')
    def test_count_must_cover_the_eligible_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root)
            path = root/'fixture-source.json'
            pins = json.loads(path.read_text())
            pins['group_count'] = 1
            path.write_text(json.dumps(pins))
            result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root), '--output', str(root/'prepared'), '--fixture'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn('cover eligible modules', result.stderr)
            self.assertFalse((root/'prepared'/'manifest.json').exists())

    @unittest.skipUnless(sys.platform == 'darwin', 'preparation requires pinned Mac packages')
    def test_invalid_checkpoint_cannot_publish_a_manifest(self):
        for defect, message in [('shape', 'sidecar storage'), ('nonfinite', 'nonfinite'),
                                ('negative', 'no negative bias'), ('duplicate', 'duplicate JSON'),
                                ('insufficient', 'insufficient groups')]:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint_fixture(root, defect)
                result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root), '--output', str(root/'prepared'), '--fixture'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse((root/'prepared'/'manifest.json').exists())

    @unittest.skipUnless(sys.platform == 'darwin', 'preparation requires pinned Mac packages')
    def test_repeated_preparation_preserves_the_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root)
            for destination in ['first', 'second']:
                result = subprocess.run([sys.executable, str(PREPARE), '--checkpoint', str(root), '--output', str(root/destination), '--fixture'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'first'/'manifest.json').read_bytes(), (root/'second'/'manifest.json').read_bytes())

if __name__ == '__main__':
    unittest.main()
