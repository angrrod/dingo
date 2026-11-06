"""Bounded generation into the ordinary Dingo HDF5 layout.

Only one process writes HDF5. A partial file is closed while workers generate
the next chunk; committed_rows is advanced only after all arrays are flushed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import resource

import h5py
import numpy as np

from dingo.core.dataset import recursive_hdf5_load, recursive_hdf5_save
from dingo.core.utils.misc import call_func_strict_output_dim


def partial_path(out_file):
    return Path(str(out_file) + ".partial")


def signature(settings, num_signals):
    content = copy.deepcopy(settings)
    # Execution chunk size can change on restart without changing the data model.
    content.pop("generation", None)
    payload = json.dumps({"settings": content, "num_signals": num_signals},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def completed_output(out_file, settings, num_signals):
    if not Path(out_file).exists():
        return False
    with h5py.File(out_file, "r") as handle:
        if (handle.attrs.get("generation_signature") == signature(settings, num_signals)
                and handle.attrs.get("generation_complete", False)):
            print(f'Generation already complete: {out_file}', flush=True)
            return True
    raise FileExistsError(f'Refusing to overwrite existing dataset: {out_file}')


def cached_basis(out_file, settings, num_signals):
    path = partial_path(out_file)
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("generation_signature") != signature(settings, num_signals):
            raise ValueError(f'Generation settings differ from partial dataset: {path}')
        return recursive_hdf5_load(handle["svd"]) if "svd" in handle else None


def memory_summary():
    # Linux ru_maxrss is KiB. This is parent-only, not shared/forked worker RSS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return f'parent peak RSS={peak:.2f} GiB'


def write_chunks(out_file, template, settings, num_signals, generate, chunk_size):
    if chunk_size <= 0:
        raise ValueError('Generation chunk_size must be positive.')
    target = int(settings["num_samples"])
    if target <= 0:
        raise ValueError('Chunked generation requires num_samples > 0.')
    path = partial_path(out_file)
    if completed_output(out_file, settings, num_signals):
        return
    metadata = copy.deepcopy(template)
    stored_settings = copy.deepcopy(settings)
    if num_signals > 1:
        from dingo.gw.dataset.generate_dataset import make_signal_suffixes
        stored_settings.update(num_signals=num_signals,
                               signal_suffixes=make_signal_suffixes(num_signals),
                               overlapping_signals=True)
    metadata.pop("settings", None)
    expected = signature(settings, num_signals)
    if not path.exists():
        with h5py.File(path, "x") as handle:
            recursive_hdf5_save(handle, metadata)
            handle.attrs["settings"] = str(stored_settings)
            handle.attrs["generation_signature"] = expected
            handle.attrs["committed_rows"] = 0
            handle.attrs["generation_complete"] = False
            handle.flush()
    with h5py.File(path, "r+") as handle:
        if handle.attrs.get("generation_signature") != expected:
            raise ValueError(f'Generation settings differ from partial dataset: {path}')
        completed = int(handle.attrs["committed_rows"])
        if not 0 <= completed <= target:
            raise ValueError(f'Invalid checkpoint row count: {completed}')
        if "parameters" in handle:
            for array in [handle["parameters"], *handle["polarizations"].values()]:
                if len(array) < completed:
                    raise ValueError(f'Partial dataset shorter than its checkpoint: {path}')
                array.resize(completed, axis=0)
        elif completed:
            raise ValueError(f'Partial dataset missing committed rows: {path}')
    initial = completed
    started = time.monotonic()
    print(f'Chunked generation: saved {completed:,}/{target:,} rows; '
          f'chunk size {chunk_size:,}; output={path}; {memory_summary()}', flush=True)
    while completed < target:
        count = min(chunk_size, target - completed)
        chunk_started = time.monotonic()
        print(f'Starting chunk: rows {completed:,} to {completed + count:,} '
              f'(exclusive); {memory_summary()}', flush=True)
        # No HDF5 handles are inherited by forked generation workers.
        parameters, polarizations = call_func_strict_output_dim(generate, count)
        generation_seconds = time.monotonic() - chunk_started
        payload = sum(values.nbytes for values in polarizations.values())
        print(f'Chunk generation finished in {generation_seconds:.1f} s; '
              f'coefficient payload={payload / 2**30:.2f} GiB; '
              f'{memory_summary()}; appending to HDF5', flush=True)
        records = parameters.to_records(index=False)
        write_started = time.monotonic()
        with h5py.File(path, "r+") as handle:
            if "parameters" not in handle:
                handle.create_dataset("parameters", shape=(0,), maxshape=(None,),
                                      dtype=records.dtype, chunks=(min(count, 1024),))
                group = handle.create_group("polarizations")
                for name, values in polarizations.items():
                    # Approximately 1 MiB HDF5 chunks; generation chunks are larger.
                    rows = max(1, min(count, 1024 * 1024 // max(1, values[0].nbytes)))
                    group.create_dataset(name, shape=(0, *values.shape[1:]),
                                         maxshape=(None, *values.shape[1:]),
                                         dtype=values.dtype,
                                         chunks=(rows, *values.shape[1:]))
            if records.dtype != handle["parameters"].dtype:
                raise ValueError('Parameter schema changed between chunks.')
            if set(polarizations) != set(handle["polarizations"]):
                raise ValueError('Polarization keys changed between chunks.')
            end = completed + count
            arrays = [(handle["parameters"], records)] + [
                (handle["polarizations"][name], values)
                for name, values in polarizations.items()]
            for array, values in arrays:
                if len(values) != count or values.shape[1:] != array.shape[1:]:
                    raise ValueError('Generated chunk has unexpected shape.')
                if values.dtype != array.dtype:
                    raise ValueError('Generated chunk dtype changed.')
                array.resize(end, axis=0)
                array[completed:end] = values
            handle.flush()
            handle.attrs.modify("committed_rows", end)
            handle.flush()
        completed = end
        elapsed = time.monotonic() - started
        rate = (completed - initial) / max(elapsed, 1e-9)
        print(f'Committed {completed:,}/{target:,} joint samples '
              f'({100 * completed / target:.2f}%), {rate:.2f} samples/s, '
              f'ETA {(target - completed) / rate / 3600:.2f} h; '
              f'write={time.monotonic() - write_started:.1f} s; '
              f'file={path.stat().st_size / 2**30:.2f} GiB; '
              f'{memory_summary()}', flush=True)
        # Release the chunk before constructing the next one.
        del parameters, polarizations, records, arrays, values
    with h5py.File(path, "r+") as handle:
        handle.attrs.modify("generation_complete", True)
        handle.flush()
    os.replace(path, out_file)
    print(f'Completed dataset: {out_file}', flush=True)
