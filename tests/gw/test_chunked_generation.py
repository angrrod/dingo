"""Incremental writing correctness and compatibility with the Dingo dataset reader."""
from pathlib import Path
import copy
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from dingo.core.dataset import DingoDataset
from dingo.gw.dataset.chunked_generation import (
    cached_basis, completed_output, partial_path, write_chunks,
)


class TestChunkedGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'waveforms.hdf5'
        self.settings = {'num_samples': 11, 'domain': {'f_max': 8}}
        self.basis = {'V': np.eye(5, 3, dtype=np.complex128)}
        self.template = {'settings': self.settings, 'svd': self.basis}
        self.next_row = 0
        self.requests = []

    def generate(self, count):
        self.requests.append(count)
        rows = np.arange(self.next_row, self.next_row + count)
        self.next_row += count
        return pd.DataFrame({'row': rows}), {
            name: np.repeat(rows[:, None], 3, axis=1).astype(np.complex128)
            for name in ('h_plus_A', 'h_cross_A', 'h_plus_B', 'h_cross_B')}

    def test_normal_reader_and_exact_final_count(self):
        write_chunks(self.output, self.template, self.settings, 2, self.generate, 4)
        self.assertEqual(self.requests, [4, 4, 3])
        self.assertFalse(partial_path(self.output).exists())
        dataset = DingoDataset(file_name=self.output,
                               data_keys=['parameters', 'polarizations', 'svd'])
        np.testing.assert_array_equal(dataset.parameters['row'], np.arange(11))
        for values in dataset.polarizations.values():
            np.testing.assert_array_equal(values[:, 0], np.arange(11))
        np.testing.assert_array_equal(dataset.svd['V'], self.basis['V'])
        self.assertEqual(dataset.settings['signal_suffixes'], ['_A', '_B'])
        self.assertTrue(completed_output(self.output, self.settings, 2))

    def test_resume_preserves_rows_and_reuses_basis(self):
        def interrupted(count):
            if self.next_row:
                raise RuntimeError('simulated interruption')
            return self.generate(count)
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            write_chunks(self.output, self.template, self.settings, 2, interrupted, 4)
        self.assertFalse(self.output.exists())
        with h5py.File(partial_path(self.output), 'r+') as handle:
            self.assertEqual(handle.attrs['committed_rows'], 4)
            np.testing.assert_array_equal(handle['parameters']['row'], np.arange(4))
            for array in handle['polarizations'].values():
                self.assertEqual(len(array), 4)
            # Append interrupted before commit: discard this tail on restart.
            handle['parameters'].resize(7, axis=0)
            handle['polarizations/h_plus_A'].resize(6, axis=0)
        changed = copy.deepcopy(self.settings)
        changed['generation'] = {'chunk_size': 6}
        np.testing.assert_array_equal(
            cached_basis(self.output, changed, 2)['V'], self.basis['V'])
        write_chunks(self.output, self.template, changed, 2, self.generate, 6)
        with h5py.File(self.output, 'r') as handle:
            np.testing.assert_array_equal(handle['parameters']['row'], np.arange(11))
            for array in handle['polarizations'].values():
                np.testing.assert_array_equal(array[:, 0], np.arange(11))

    def test_incompatible_resume_rejected(self):
        def fail(count):
            raise RuntimeError('interruption')
        with self.assertRaises(RuntimeError):
            write_chunks(self.output, self.template, self.settings, 2, fail, 4)
        changed = {**self.settings, 'num_samples': 12}
        with self.assertRaises(ValueError):
            cached_basis(self.output, changed, 2)
        with self.assertRaises(ValueError):
            write_chunks(self.output, self.template, changed, 2, self.generate, 4)

    def test_existing_output_is_not_overwritten(self):
        with h5py.File(self.output, 'w') as handle:
            handle.attrs['sentinel'] = 'existing dataset'
        with self.assertRaises(FileExistsError):
            write_chunks(self.output, self.template, self.settings, 2, self.generate, 4)
        with h5py.File(self.output, 'r') as handle:
            self.assertEqual(handle.attrs['sentinel'], 'existing dataset')


if __name__ == '__main__':
    unittest.main()
