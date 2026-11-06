"""Atomic per-detector embedding checkpoints and persisted initialization examples."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import h5py
import numpy as np
import pandas as pd

from dingo.gw.SVD import SVDBasis


class EmbeddingSVDCache:
    def __init__(self, root, recipe):
        self.recipe = recipe
        serialized = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        self.signature = hashlib.sha256(serialized.encode()).hexdigest()
        self.directory = Path(root) / self.signature
        self.ifos = recipe['data']['detectors']
        self.rank = recipe['svd']['size']

    @contextmanager
    def preparing(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / 'prepare.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Embedding preparation already running: {self.directory}')
            try:
                self._save('recipe.json', lambda path: path.write_text(
                    json.dumps(self.recipe, sort_keys=True, indent=2)))
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _save(self, filename, write):
        destination = self.directory / filename
        temporary = destination.with_name(destination.name + f'.{uuid.uuid4().hex}.tmp')
        try:
            write(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_examples(self, waveforms, parameters):
        # Publish the manifest last, so incomplete example files are never reused.
        def save_array(filename, array):
            def write(path):
                with open(path, 'wb') as stream:
                    np.save(stream, array, allow_pickle=False)
            self._save(filename, write)
        for ifo in self.ifos:
            save_array(f'examples_{ifo}.npy', waveforms[ifo])
        save_array('example_parameters.npy', parameters.to_records(index=False))
        self._save('examples.json', lambda path: path.write_text(json.dumps({
            'signature': self.signature, 'ifos': self.ifos,
            'shapes': {ifo: list(waveforms[ifo].shape) for ifo in self.ifos}})))

    def load_examples(self):
        manifest = self.directory / 'examples.json'
        if not manifest.exists():
            return None
        info = json.loads(manifest.read_text())
        if info['signature'] != self.signature or info['ifos'] != self.ifos:
            raise ValueError(f'Incompatible initialization examples: {manifest}')
        waveforms = {ifo: np.load(self.directory / f'examples_{ifo}.npy',
                                  mmap_mode='r', allow_pickle=False) for ifo in self.ifos}
        for ifo, array in waveforms.items():
            if list(array.shape) != info['shapes'][ifo]:
                raise ValueError(f'Invalid initialization array: {ifo}')
        parameters = pd.DataFrame(np.load(self.directory / 'example_parameters.npy',
                                          allow_pickle=False))
        if any(len(array) != len(parameters) for array in waveforms.values()):
            raise ValueError('Initialization examples and parameters differ in length.')
        print(f'Reusing prepared clean, whitened examples: {self.directory}', flush=True)
        return waveforms, parameters

    def save_basis(self, ifo, basis, validated=False):
        def write(path):
            basis.to_file(path)
            with h5py.File(path, 'r+') as handle:
                handle.attrs['embedding_signature'] = self.signature
                handle.attrs['validated'] = validated
        self._save(f'svd_{ifo}.hdf5', write)

    def load_basis(self, ifo):
        path = self.directory / f'svd_{ifo}.hdf5'
        if not path.exists():
            return None
        with h5py.File(path, 'r') as handle:
            if handle.attrs.get('embedding_signature') != self.signature:
                raise ValueError(f'Incompatible embedding checkpoint: {path}')
            validated = bool(handle.attrs.get('validated', False))
        basis = SVDBasis(file_name=path)
        if basis.V.ndim != 2 or basis.V.shape[1] != self.rank:
            raise ValueError(f'Invalid embedding checkpoint shape: {path}')
        print(f'Reusing {ifo} basis (validated={validated}): {path}', flush=True)
        return basis, validated

    def publish(self, first_index):
        for ifo in self.ifos:
            saved = self.load_basis(ifo)
            if saved is None or not saved[1]:
                raise RuntimeError(f'Cannot publish incomplete {ifo} basis.')
        self._save('ready.json', lambda path: path.write_text(json.dumps({
            'signature': self.signature, 'ifos': self.ifos, 'first_index': first_index})))

    def load_ready(self, required=True):
        path = self.directory / 'ready.json'
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f'Completed embedding SVD cache missing: {self.directory}. '
                    'Run scripts/prepare_embedding_svd.py with the matching training '
                    'configuration on the CPU partition before GPU training.')
            return None
        info = json.loads(path.read_text())
        if info['signature'] != self.signature or info['ifos'] != self.ifos:
            raise ValueError(f'Incompatible ready manifest: {path}')
        result = []
        for ifo in self.ifos:
            saved = self.load_basis(ifo)
            if saved is None or not saved[1]:
                raise RuntimeError(f'Completed cache has unfinished detector {ifo}.')
            basis = saved[0]
            start = info['first_index']
            if not 0 <= start < len(basis.V) or not np.allclose(basis.V[:start], 0):
                raise ValueError(f'Invalid low-frequency truncation for {ifo}.')
            result.append(basis.V[start:])
        return result
