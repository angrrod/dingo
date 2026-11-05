# Optional GPU waveform preprocessing

Enable in the training YAML:

```yaml
local:
  device: cuda
  preprocessing_device: cuda
```

Set `preprocessing_device: cpu` (or omit it) to use the original preprocessing.
This setting is independent of `num_workers` and the BLAS thread limits.

## Implementation

The executable logic is in `dingo/gw/training/gpu_preprocessing.py`. Its module,
class and method docstrings explain the CPU/GPU split, batch shapes, geometry
keys, reconstruction formula, whitening, noise and network input channels.

Supported: uniform frequency domains, SVD plus/cross polarizations, single and
suffixed overlapping signals, optional compression whitening, context parameters,
and zero-noise training. Other decompression/training transforms (including
heterodyning, GNPE, chirp conditioning, tokenization and cropping) raise errors.

## Verification on the remote server

From the repository root, using the environment containing this checkout of DINGO:

```bash
python -c 'import torch; assert torch.cuda.is_available()'
python External/dingo/tests/gw/test_gpu_preprocessing.py
```

The test uses synthetic deterministic inputs, so it needs no waveform dataset.
It compares the original CPU transforms with the new path for single and A/B
signals, batch sizes 1 and 4, optional inverse compression whitening, target/context
packaging and nonzero arrival times. It exercises the real dataset's deferred
read/unbatch path. A separate test checks the noise component variance.
CUDA comparisons are skipped if CUDA is unavailable. CPU tests establish numerical
agreement of the formulas, not CUDA correctness or a performance improvement.

For the first real-data run, use a separate output directory and a short run limit.
Compare wall-clock throughput against the CPU path with identical workers, BLAS
limits, batch size and cache state. GPU preprocessing is included in the loop's
network-side timing, so that field no longer represents only the neural network.
CUDA execution is asynchronous: use synchronization or CUDA events for a precise
component benchmark. This implementation does not yet overlap next-batch transfer
with current-batch computation on a separate CUDA stream.
