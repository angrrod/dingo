"""Deterministic parity of CPU and device preprocessing, including A/B geometry."""
from types import SimpleNamespace
import copy

import numpy as np
import pandas as pd
import unittest
import torch
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose
from torch.utils.data import default_collate

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.SVD import SVDBasis, ApplySVD
from dingo.gw.transforms import (
    ProjectOntoDetectors, GetDetectorTimes, WhitenAndScaleStrain,
    RepackageStrainsAndASDS, SelectStandardizeRepackageParameters, WhitenFixedASD,
)
from dingo.gw.training.gpu_preprocessing import (
    GPUWaveformPreprocessor, PrepareCompressedBatch,
)


class FixedExtrinsics:
    def __init__(self, values):
        self.values = values

    def __call__(self, sample):
        return {**sample, 'extrinsic_parameters': self.values}


class FixedASDs:
    def __init__(self, asds):
        self.asds = asds

    def __call__(self, sample):
        return {**sample, 'asds': self.asds}


def check_projection_packaging_parity(batch_size, suffixes, device, whiten=False, cached=False):
    rng = np.random.default_rng(13)
    domain = UniformFrequencyDomain(f_min=2, f_max=16, delta_f=0.25)
    V = (rng.normal(size=(len(domain), 5)) +
         1j * rng.normal(size=(len(domain), 5))).astype(np.complex64)
    V[:domain.min_idx] = 0
    basis = SVDBasis(dictionary={'V': V})
    wfd = SimpleNamespace(domain=domain, svd={'V': V},
                          decompression_transform=Compose([ApplySVD(basis, inverse=True)]))
    if whiten:
        # Use a fixed array to isolate inversion from PSD file loading/interpolation.
        undo = WhitenFixedASD.__new__(WhitenFixedASD)
        undo.inverse = True
        undo.asd_array = np.linspace(0.5, 2, len(domain), dtype=np.float32)
        wfd.decompression_transform.transforms.append(undo)
    parameters, extrinsics, coefficients = {}, {}, {}
    for suffix in suffixes:
        parameters['luminosity_distance' + suffix] = np.full(batch_size, 100.)
        parameters['geocent_time' + suffix] = np.zeros(batch_size)
        for key, low, high in [('luminosity_distance', 200, 1000), ('ra', 0, 6),
                               ('dec', -1, 1), ('psi', 0, 3), ('geocent_time', 20, 58)]:
            extrinsics[key + suffix] = rng.uniform(low, high, batch_size)
        for pol in ['h_plus', 'h_cross']:
            coefficients[pol + suffix] = (rng.normal(size=(batch_size, 5)) +
                1j * rng.normal(size=(batch_size, 5))).astype(np.complex64) * 1e-22
    ifos = InterferometerList(['H1', 'L1'])
    sample = GetDetectorTimes(ifos, 0, isSingle=suffixes == ('',))(
        {'parameters': parameters, 'extrinsic_parameters': extrinsics,
         'waveform': coefficients})
    projection = ProjectOntoDetectors(ifos, domain, 0)
    asds = {ifo.name: np.full((batch_size, len(domain)), 1e-23, dtype=np.float32)
            for ifo in ifos}
    keys = ['luminosity_distance' + suffixes[0], 'geocent_time' + suffixes[-1]]
    standardize = SelectStandardizeRepackageParameters(
        {'inference_parameters': keys, 'context_parameters': [keys[1]]},
        {'mean': dict.fromkeys(keys, 0), 'std': dict.fromkeys(keys, 1)})
    cpu = wfd.decompression_transform(copy.deepcopy(sample))
    cpu = projection(cpu)
    cpu = FixedASDs(asds)(cpu)
    cpu = WhitenAndScaleStrain(domain.noise_std)(cpu)
    cpu = RepackageStrainsAndASDS(['H1', 'L1'], domain.min_idx)(cpu)
    cpu = standardize(cpu)
    # Exercise the actual dataset's deferred-decompression and unbatch boundary.
    dataset = WaveformDataset.__new__(WaveformDataset)
    dataset.file_handle = None
    dataset._leave_on_disk_keys = []
    dataset.parameters = pd.DataFrame(parameters)
    dataset.polarizations = coefficients
    dataset.decompression_transform = wfd.decompression_transform
    dataset.defer_decompression = True
    dataset.transform = PrepareCompressedBatch(
        [FixedExtrinsics(sample['extrinsic_parameters'])], projection,
        None if cached else FixedASDs(asds), standardize)
    batch = default_collate(dataset.__getitems__(list(range(batch_size))))
    asd_dataset = SimpleNamespace(asds={k: v[:1] for k, v in asds.items()})
    module = GPUWaveformPreprocessor(
        wfd, projection, add_noise=False,
        asd_dataset=asd_dataset if cached else None).to(device)
    if cached:
        assert 'asds' not in batch
        assert module.cache_fixed_asds
        assert module.fixed_asd_0.device == module.Vh.device
        assert module.fixed_asd_0.shape == (1, len(domain))
    actual = module(batch)
    for observed, key in zip(actual, ['inference_parameters', 'waveform', 'context_parameters']):
        np.testing.assert_allclose(observed.cpu().numpy(), cpu[key], atol=2e-5, rtol=2e-5)
        assert not observed.requires_grad
    return module, batch


class TestGPUPreprocessing(unittest.TestCase):
    def test_cpu_parity(self):
        for size in (1, 4):
            for suffixes in (("",), ("_A", "_B")):
                with self.subTest(batch_size=size, suffixes=suffixes):
                    for whiten in (False, True):
                        check_projection_packaging_parity(size, suffixes, "cpu", whiten)

    def test_fixed_asd_cache_parity(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for size in (1, 4):
                for suffixes in (("",), ("_A", "_B")):
                    for whiten in (False, True):
                        with self.subTest(device=device, size=size, suffixes=suffixes,
                                          whiten=whiten):
                            check_projection_packaging_parity(
                                size, suffixes, device, whiten, cached=True)

    def test_multiple_asds_preserve_batch_sampling(self):
        module, batch = check_projection_packaging_parity(4, ("_A", "_B"), 'cpu')
        domain = UniformFrequencyDomain(f_min=2, f_max=16, delta_f=0.25)
        wfd = SimpleNamespace(domain=domain, svd={'V': module.Vh.numpy().conj().T},
                             decompression_transform=Compose([]))
        projection = SimpleNamespace(ifo_list=InterferometerList(['H1', 'L1']))
        # Mixed fixed/variable banks must also use the original sampled path.
        bank = SimpleNamespace(asds={'H1': batch['asds']['H1'][:1].numpy(),
                                     'L1': batch['asds']['L1'][:2].numpy()})
        fallback = GPUWaveformPreprocessor(wfd, projection, add_noise=False,
                                           asd_dataset=bank)
        self.assertFalse(fallback.cache_fixed_asds)
        batch['asds']['L1'][1] *= 2
        torch.testing.assert_close(fallback(batch)[1], module(batch)[1])

    def test_noise_components_have_unit_variance(self):
        module, batch = check_projection_packaging_parity(256, ("_A", "_B"), "cpu")
        clean = module(batch)[1]
        module.add_noise = True
        noisy = module(batch)[1]
        noise = (noisy - clean)[:, :, :2]
        self.assertLess(abs(noise.mean().item()), 0.03)
        self.assertLess(abs(noise.std().item() - 1), 0.03)
        torch.testing.assert_close(noisy[:, :, 2], clean[:, :, 2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_parity(self):
        for size in (1, 4):
            for suffixes in (("",), ("_A", "_B")):
                with self.subTest(batch_size=size, suffixes=suffixes):
                    for whiten in (False, True):
                        check_projection_packaging_parity(size, suffixes, "cuda", whiten)


if __name__ == "__main__":
    unittest.main()
