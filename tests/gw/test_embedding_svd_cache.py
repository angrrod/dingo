"""Checkpoint state transitions, restart consistency, and detector ordering."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dingo.gw.SVD import SVDBasis
from dingo.gw.training.embedding_svd_cache import EmbeddingSVDCache


class TestEmbeddingSVDCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recipe = {'data': {'detectors': ['L1', 'H1']}, 'svd': {'size': 2}, 'seed': 42}
        self.cache = EmbeddingSVDCache(self.tmp.name, self.recipe)
        self.basis = SVDBasis(dictionary={
            'V': np.vstack([np.zeros((2, 2)), np.eye(4, 2)]).astype(complex)})

    def test_per_detector_resume_and_validation_states(self):
        with self.cache.preparing():
            self.cache.save_basis('L1', self.basis)
        restarted = EmbeddingSVDCache(self.tmp.name, self.recipe)
        self.assertFalse(restarted.load_basis('L1')[1])
        self.assertIsNone(restarted.load_basis('H1'))
        with self.assertRaises(FileNotFoundError):
            restarted.load_ready()
        with restarted.preparing():
            with self.assertRaises(RuntimeError):
                restarted.publish(2)
            restarted.save_basis('L1', self.basis, validated=True)
            other = SVDBasis(dictionary={'V': self.basis.V * 1j})
            restarted.save_basis('H1', other, validated=True)
            restarted.publish(2)
        matrices = restarted.load_ready()
        np.testing.assert_array_equal(matrices[0], self.basis.V[2:])
        np.testing.assert_array_equal(matrices[1], other.V[2:])

    def test_examples_persist_exactly_and_load_as_mapped_arrays(self):
        examples = {ifo: np.arange(24).reshape(4, 6).astype(complex)
                    for ifo in self.recipe['data']['detectors']}
        parameters = pd.DataFrame({'row': np.arange(4)})
        with self.cache.preparing():
            self.cache.save_examples(examples, parameters)
        restored, labels = self.cache.load_examples()
        for ifo in examples:
            self.assertIsInstance(restored[ifo], np.memmap)
            np.testing.assert_array_equal(restored[ifo], examples[ifo])
        pd.testing.assert_frame_equal(labels, parameters)

    def test_failed_atomic_write_preserves_previous_basis(self):
        with self.cache.preparing():
            self.cache.save_basis('L1', self.basis)
            with patch.object(self.basis, 'to_file', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    self.cache.save_basis('L1', self.basis, validated=True)
        saved, validated = self.cache.load_basis('L1')
        np.testing.assert_array_equal(saved.V, self.basis.V)
        self.assertFalse(validated)
        self.assertFalse(list(self.cache.directory.glob('*.tmp')))

    def test_changed_recipe_uses_new_cache_and_concurrent_writer_is_rejected(self):
        changed = EmbeddingSVDCache(self.tmp.name, {**self.recipe, 'seed': 43})
        self.assertNotEqual(self.cache.directory, changed.directory)
        with self.cache.preparing():
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with EmbeddingSVDCache(self.tmp.name, self.recipe).preparing():
                    pass


if __name__ == '__main__':
    unittest.main()
