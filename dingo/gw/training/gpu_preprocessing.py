"""Prepare training waveforms on the GPU while keeping sampling on the CPU.

Execution order
---------------
1. ``initialize_stage()`` builds the normal training transforms, then calls
   ``enable_gpu_preprocessing(wfd)`` before creating the DataLoaders.
2. Dataset reads skip decompression. ``PrepareCompressedBatch`` samples extrinsic
   parameters/ASDs and computes small geometry arrays in CPU workers.
3. The DataLoader collates those compressed samples into a nested tensor dict.
4. ``_prepare_training_batch()`` calls ``GPUWaveformPreprocessor`` in the main
   training process, for both training and validation. It returns the usual list
   of network inputs, so the loss and optimizer need no special handling.

The dataset/worker transform holds no CUDA tensors. The device module belongs to
``pm.batch_preprocessor``; its fixed buffers move to the GPU once per stage.
Enable with ``local.preprocessing_device: cuda``; ``cpu`` keeps the original path.

Notation: B = batch size, K = retained SVD coefficients, F = full frequency bins,
D = detectors, P = inference parameters, C = optional context parameters.
Supported: uniform frequency domains, SVD plus/cross polarizations, single or
suffixed overlapping signals, optional compression whitening, and zero noise.
Unsupported decompression/training transforms raise errors during stage setup.
"""
import numpy as np
import torch

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.SVD import ApplySVD
from dingo.gw.transforms.detector_transforms import ProjectOntoDetectors
from dingo.gw.transforms.noise_transforms import WhitenFixedASD


class PrepareProjection(ProjectOntoDetectors):
    """Collect CPU geometry while reusing the original parameter bookkeeping.

    The parent handles source suffixes and moves sampled extrinsic parameters
    into the parameter dictionary. Its calls to ``_project_one_signal`` dispatch
    to our override, which records distance ratios, antenna factors and times
    instead of expanding polarizations into detector strains.

    Geometry keys use source indices (0, 1, ...) in the parent's sorted suffix
    order, matching the device loop. Each value has shape (B,).
    """

    def __call__(self, sample):
        # Reset the collector so one batch cannot retain another batch's geometry.
        self.geometry = {}
        result = super().__call__(sample)
        result['geometry'] = self.geometry
        # The parent replaces waveform with detector strains; retain coefficients.
        result['waveform'] = sample['waveform']
        return result

    def _project_one_signal(self, *, h_cross, h_plus, d_ref, d_new,
                            ra, dec, psi, tc_ref, detector_times):
        """Record one source's geometry; h_plus/h_cross are still coefficients.

        The empty return lets the parent finish updating parameters without
        allocating any frequency-domain detector strains.
        """
        if not np.allclose(tc_ref, 0):
            raise ValueError('GPU preprocessing requires reference geocent_time = 0.')
        # Each source contributes one distance and three values per detector.
        index = len(self.geometry) // (1 + 3 * len(self.ifo_list))
        prefix = str(index)
        self.geometry[f'{prefix}:distance'] = np.broadcast_to(
            np.asarray(d_ref / d_new), h_plus.shape[:-1])
        for ifo in self.ifo_list:
            fp, fc = self._antenna_patterns(ifo, ra, dec, psi)
            for name, value in [('plus', fp), ('cross', fc),
                                ('time', detector_times[ifo.name] - tc_ref)]:
                self.geometry[f'{prefix}:{ifo.name}:{name}'] = np.broadcast_to(
                    np.asarray(value).reshape(-1), h_plus.shape[:-1])
        return {}  # The caller still performs the original parameter updates.

class PrepareCompressedBatch:
    """Replace the dataset's transform with CPU-only batch preparation.

    ``prefix`` contains the original transforms before detector projection
    (parameter sampling and detector-time calculation). ``projection`` supplies
    detector geometry, ``sample_asd`` draws noise curves, and ``standardize``
    produces the same targets/context as the CPU pipeline.

    Returns a two-level dictionary of NumPy arrays, which the dataset unbatches
    into samples and the DataLoader collates into tensors:

    * waveform: h_plus/h_cross (or h_plus_A, etc.), each (B, K), complex.
    * geometry: distance ratio, plus/cross response and time per source/detector,
      each (B,), real. Example key: ``0:H1:time``.
    * asds: detector name -> (B, F), real; omitted when sample_asd=None
      because the preprocessor holds fixed ASD buffers.
    * targets: inference_parameters (B, P), optional context_parameters (B, C).

    The key ``waveform`` deliberately retains the dataset's naming convention,
    but its values are compressed coefficients at this point.
    """

    def __init__(self, prefix, projection, sample_asd, standardize):
        self.prefix = prefix
        self.projection = PrepareProjection(
            projection.ifo_list, projection.domain, projection.ref_time)
        self.sample_asd = sample_asd
        self.standardize = standardize

    def __call__(self, sample):
        for transform in self.prefix:
            sample = transform(sample)
        sample = self.projection(sample)
        if self.sample_asd is not None:
            sample = self.sample_asd(sample)
        sample = self.standardize(sample)
        targets = {'inference_parameters': sample['inference_parameters']}
        if 'context_parameters' in sample:
            targets['context_parameters'] = sample['context_parameters']
        batch = {'waveform': sample['waveform'], 'geometry': sample['geometry'],
                 'targets': targets}
        if self.sample_asd is not None:
            batch['asds'] = sample['asds']
        return batch

class GPUWaveformPreprocessor(torch.nn.Module):
    """Reconstruct, project, whiten, add noise and package a collated batch.

    Construct once per stage from the dataset and projection transform, then move
    with ``.to(pm.device)``. The registered buffers are fixed data, not trainable
    parameters: Vh (K, F), the frequency grid (F,), and inverse compression
    whitening (F,), plus optional fixed detector ASDs (1, F). Vh is the dataset basis, separate from the embedding basis.

    ``add_noise=False`` implements the existing zero-noise option. This module
    also runs on CPU for numerical comparisons. The training-loop hook disables
    autocast; float64 time phases preserve precision for long arrival offsets.
    The final network input is float32.
    """

    def __init__(self, wfd, projection, add_noise=True, asd_dataset=None):
        super().__init__()
        if not isinstance(wfd.domain, UniformFrequencyDomain):
            raise NotImplementedError('GPU preprocessing requires a uniform frequency domain.')
        if wfd.svd is None:
            raise ValueError('GPU preprocessing requires an SVD-compressed dataset.')
        self.ifos = [ifo.name for ifo in projection.ifo_list]
        self.first_index = wfd.domain.min_idx
        self.scale = wfd.domain.noise_std
        self.add_noise = add_noise
        # Cache only when ALL detectors have exactly one curve. Otherwise retain
        # the original per-example CPU sampling, including mixed fixed/variable banks.
        self.cache_fixed_asds = asd_dataset is not None and all(
            np.asarray(asd_dataset.asds[ifo]).shape == (1, len(wfd.domain))
            for ifo in self.ifos)
        if self.cache_fixed_asds:
            for index, ifo in enumerate(self.ifos):
                curve = np.array(asd_dataset.asds[ifo], dtype=np.float32, copy=True)
                self.register_buffer(f'fixed_asd_{index}', torch.from_numpy(curve))
        # NumPy decompression is coefficients @ V.conj().T; preserve its dtype.
        self.register_buffer('Vh', torch.as_tensor(wfd.svd['V'].T.conj().copy()))
        self.register_buffer('frequencies', torch.as_tensor(
            wfd.domain.sample_frequencies, dtype=torch.float64))
        # Undo storage-time whitening before applying sampled detector ASDs.
        # An unwhitened dataset gets the identity multiplier instead.
        undo_whitening = np.ones(len(wfd.domain), dtype=np.float32)
        for transform in wfd.decompression_transform.transforms:
            if isinstance(transform, ApplySVD):
                continue
            if isinstance(transform, WhitenFixedASD) and transform.inverse:
                undo_whitening *= transform.asd_array
            else:
                raise NotImplementedError(
                    f'GPU decompression does not support {type(transform).__name__}.')
        self.register_buffer('undo_whitening', torch.as_tensor(undo_whitening))

    @torch.no_grad()
    def forward(self, batch):
        """Convert a collated compressed batch into the existing loss inputs.

        Input schema is documented in ``PrepareCompressedBatch``. Return:
        ``[targets, waveform, optional_context]``, where waveform has shape
        (B, D, 3, F - domain.min_idx). Its channels are real strain, imaginary
        strain and 1 / (ASD * 1e23), in the configured detector order.

        All returned tensors are on the buffer device. No autograd graph is
        needed: preprocessing generates inputs rather than fitting parameters.
        Noise has independent N(0, 1) real/imaginary components; CUDA draws need
        not match CPU draws sample by sample.
        """
        # Transfer nested tensors, including targets, once in the main process.
        device = self.Vh.device
        batch = {key: {k: v.to(device, non_blocking=True) for k, v in values.items()}
                 for key, values in batch.items()}
        coefficients = batch['waveform']
        suffixes = sorted(k[len('h_plus'):] for k in coefficients if k.startswith('h_plus'))
        # Sorted suffixes match the CPU collector: "" for single, or _A/_B.
        strains = {}
        for index, suffix in enumerate(suffixes):
            # Batched reconstruction: (B, K) @ (K, F) -> (B, F).
            hp = coefficients['h_plus' + suffix].to(self.Vh.dtype) @ self.Vh
            hc = coefficients['h_cross' + suffix].to(self.Vh.dtype) @ self.Vh
            hp = hp * self.undo_whitening
            hc = hc * self.undo_whitening
            geometry = batch['geometry']
            # [:, None] broadcasts each sample's scalar over frequency bins.
            distance = geometry[f'{index}:distance'][:, None]
            for ifo in self.ifos:
                fp = geometry[f'{index}:{ifo}:plus'][:, None]
                fc = geometry[f'{index}:{ifo}:cross'][:, None]
                dt = geometry[f'{index}:{ifo}:time'].double()[:, None]
                # h_detector(f) = (Fp * hp + Fc * hc) * distance * exp(-2πifΔt).
                phase = torch.exp(-2j * torch.pi * dt * self.frequencies).to(hp.dtype)
                strain = (fp * (hp * distance) + fc * (hc * distance)) * phase
                strains[ifo] = strains.get(ifo, 0) + strain
        # Sum physical source strains before detector whitening and noise.
        channels = []
        for index, ifo in enumerate(self.ifos):
            asd = (getattr(self, f'fixed_asd_{index}') if self.cache_fixed_asds
                   else batch['asds'][ifo])
            strain = strains[ifo] / (asd * self.scale)
            if self.add_noise:
                strain = strain + torch.complex(
                    torch.randn(strain.shape, device=device),
                    torch.randn(strain.shape, device=device))
            # Remove bins below f_min only when packaging, as in the CPU path.
            start = self.first_index
            channels.append(torch.stack((strain[..., start:].real,
                                         strain[..., start:].imag,
                                         (1 / (asd[..., start:] * 1e23)).expand(
                                             strain.shape[0], -1)), dim=-2))
        waveform = torch.stack(channels, dim=1).float()
        result = [batch['targets']['inference_parameters'], waveform]
        if 'context_parameters' in batch['targets']:
            result.append(batch['targets']['context_parameters'])
        return result


def enable_gpu_preprocessing(wfd):
    """Split a freshly built, ordinary training transform chain in place.

    Call only after ``set_train_transforms`` and any CPU stage diagnostics, and
    before creating DataLoader workers. Replaces ``wfd.transform`` with
    ``PrepareCompressedBatch`` and sets ``wfd.defer_decompression`` so dataset
    reads return coefficients. Returns a device module initially on CPU; the
    caller attaches it to the posterior model and moves it to the training GPU.

    The original transform instances supply geometry, ASD sampling and target
    standardization settings. Unknown transform types are rejected rather than
    omitted. Calling ``set_train_transforms`` again rebuilds the CPU chain and
    clears the deferral flag before a later stage is configured.
    """
    from dingo.gw.transforms import (
        SampleExtrinsicParameters, GetDetectorTimes, SampleNoiseASD,
        WhitenAndScaleStrain, AddWhiteNoiseComplex,
        SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict,
    )
    transforms = wfd.transform.transforms
    allowed = (SampleExtrinsicParameters, GetDetectorTimes, ProjectOntoDetectors,
               SampleNoiseASD, WhitenAndScaleStrain, AddWhiteNoiseComplex,
               SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict)
    for transform in transforms:
        if type(transform) not in allowed:
            raise NotImplementedError(
                f'GPU preprocessing does not support {type(transform).__name__}.')
    def get(cls):
        return next(t for t in transforms if type(t) is cls)
    projection = get(ProjectOntoDetectors)
    sample_asd = get(SampleNoiseASD)
    module = GPUWaveformPreprocessor(
        wfd, projection, any(type(t) is AddWhiteNoiseComplex for t in transforms),
        asd_dataset=sample_asd.asd_dataset)
    split = transforms.index(projection)
    wfd.transform = PrepareCompressedBatch(
        transforms[:split], projection,
        None if module.cache_fixed_asds else sample_asd,
        get(SelectStandardizeRepackageParameters))
    wfd.defer_decompression = True
    return module
