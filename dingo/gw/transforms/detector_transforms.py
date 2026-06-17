import math
import numpy as np
import torch
import pandas as pd
from bilby.gw.detector.interferometer import Interferometer
from lal import GreenwichMeanSiderealTime
from typing import Union
from bilby.gw.detector import calibration
from bilby.gw.prior import CalibrationPriorDict
from bilby_pipe.utils import CALIBRATION_CORRECTION_TYPE_LOOKUP

CC = 299792458.0


def time_delay_from_geocenter(
    ifo: Interferometer,
    ra: Union[float, np.ndarray, torch.Tensor],
    dec: Union[float, np.ndarray, torch.Tensor],
    time: float,
):
    """
    Calculate time delay between ifo and geocenter. Identical to method
    ifo.time_delay_from_geocenter(ra, dec, time), but the present implementation allows
    for batched computation, i.e., it also accepts arrays and tensors for ra and dec.

    Implementation analogous to bilby-cython implementation
    https://git.ligo.org/colm.talbot/bilby-cython/-/blob/main/bilby_cython/geometry.pyx,
    which is in turn based on XLALArrivaTimeDiff in TimeDelay.c.

    Parameters
    ----------
    ifo: bilby.gw.detector.interferometer.Interferometer
        bilby interferometer object.
    ra: Union[float, np.array, torch.Tensor]
        Right ascension of the source in radians. Either float, or float array/tensor.
    dec: Union[float, np.array, torch.Tensor]
        Declination of the source in radians. Either float, or float array/tensor.
    time: float
        GPS time in the geocentric frame.

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame
    """
    # check that ra and dec are of same type and length
    if type(ra) != type(dec):
        raise ValueError(
            f"ra type ({type(ra)}) and dec type ({type(dec)}) don't match."
        )
    if isinstance(ra, (np.ndarray, torch.Tensor)):
        if len(ra.shape) != 1:
            raise ValueError(f"Only one axis expected for ra and dec, got multiple.")
        if ra.shape != dec.shape:
            raise ValueError(
                f"Shapes of ra ({ra.shape}) and dec ({dec.shape}) don't match."
            )

    if isinstance(ra, (float, np.float32, np.float64)):
        return ifo.time_delay_from_geocenter(ra, dec, time)

    elif isinstance(ra, (np.ndarray, torch.Tensor)) and len(ra) == 1:
        return ifo.time_delay_from_geocenter(ra[0], dec[0], time)

    else:
        if isinstance(ra, np.ndarray):
            sin = np.sin
            cos = np.cos
        elif isinstance(ra, torch.Tensor):
            sin = torch.sin
            cos = torch.cos
        else:
            raise NotImplementedError(
                "ra, dec must be either float, np.ndarray, or torch.Tensor."
            )

        gmst = math.fmod(GreenwichMeanSiderealTime(float(time)), 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec
        sintheta = sin(theta)
        costheta = cos(theta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        detector_1 = ifo.vertex
        detector_2 = np.zeros(3)
        return (
            (detector_2[0] - detector_1[0]) * sintheta * cosphi
            + (detector_2[1] - detector_1[1]) * sintheta * sinphi
            + (detector_2[2] - detector_1[2]) * costheta
        ) / CC


class GetDetectorTimes(object):
    """
    Compute the time shifts in the individual detectors based on the sky
    position (ra, dec), the geocent_time and the ref_time.
    """

    def __init__(self, ifo_list, ref_time, isSingle:bool = True):
        self.ifo_list = ifo_list
        self.ref_time = ref_time
        self.isSingle = isSingle
        print(f'$$$ GetDetectorTimes: isSingle {isSingle}')

    def __call__(self, input_sample):
        if self.isSingle:
            return self._call_single(input_sample)
        else:
            return self._call_joint(input_sample)

    def _call_single(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        ra = extrinsic_parameters["ra"]
        dec = extrinsic_parameters["dec"]
        geocent_time = extrinsic_parameters["geocent_time"]
        for ifo in self.ifo_list:
            extrinsic_parameters[f"{ifo.name}_time"] = self._compute_ifo_time(ra,dec,ifo,geocent_time)
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
    
    def _call_joint(self, input_sample):
        sample = input_sample.copy()

        # Shallow-copy protection.
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        suffixes = ["_A","_B"]
        if suffixes is None:
            suffixes = self._infer_signal_suffixes(extrinsic_parameters)

        for suffix in suffixes:
            ra_key = f"ra{suffix}"
            dec_key = f"dec{suffix}"
            time_key = f"geocent_time{suffix}"

            missing = [
                key for key in [ra_key, dec_key, time_key]
                if key not in extrinsic_parameters
            ]

            if missing:
                raise KeyError(
                    f"Missing keys for joint detector-time computation "
                    f"for suffix {suffix}: {missing}. "
                    f"Available keys are: {list(extrinsic_parameters.keys())}"
                )

            ra = extrinsic_parameters[ra_key]
            dec = extrinsic_parameters[dec_key]
            geocent_time = extrinsic_parameters[time_key]

            for ifo in self.ifo_list:
                extrinsic_parameters[f"{ifo.name}_time{suffix}"] = self._compute_ifo_time(ra,dec,ifo,geocent_time)

        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

    def _compute_ifo_time(self,ra,dec,ifo,geocent_time):
        if type(ra) == torch.Tensor:
            # computation does not work on gpu, so do it on cpu
            ra = ra.cpu()
            dec = dec.cpu()
        dt = time_delay_from_geocenter(ifo, ra, dec, self.ref_time)
        if type(dt) == torch.Tensor:
            dt = dt.to(geocent_time.device)
        ifo_time = geocent_time + dt
        return ifo_time

class ProjectOntoDetectors(object):
    """
    Project the GW polarizations onto the detectors in ifo_list. This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:

    (1) Rescale polarizations to account for sampled luminosity distance
    (2) Project polarizations onto the antenna patterns using the ref_time and
        the extrinsic parameters (ra, dec, psi)
    (3) Time shift the strains in the individual detectors according to the
        times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain, ref_time):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time

    def __call__(self, input_sample):
        if any(key.endswith("_A") for key in input_sample["parameters"]):  #switch between multiple or single signal output, ensure backwards compatibility
            return self._call_overlapping_signals(input_sample)
        return self._call_single_signal(input_sample)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_ratio(d_ref, d_new):
        """
        Compute d_ref / d_new with the same broadcasting convention as the
        original DINGO implementation.
        """
        # (1) rescale polarizations and set distance parameter to sampled value
        if np.isscalar(d_ref) or np.isscalar(d_new):
            return d_ref / d_new

        if isinstance(d_ref, np.ndarray) and isinstance(d_new, np.ndarray):
            return (d_ref / d_new)[:, np.newaxis]

        raise ValueError("luminosity_distance should be a float or a numpy array.")
    
    def _antenna_patterns(self, ifo, ra, dec, psi):
        """
        Compute plus/cross antenna responses.

        This keeps the original non-vectorized loop for batched parameters.
        """
        if any(np.isscalar(x) for x in [ra, dec, psi]):
            fp = ifo.antenna_response(
                ra, dec, self.ref_time, psi, mode="plus"
            )
            fc = ifo.antenna_response(
                ra, dec, self.ref_time, psi, mode="cross"
            )
        else:
            fp = np.array(
                [
                    ifo.antenna_response(
                        ra_i, dec_i, self.ref_time, psi_i, mode="plus"
                    )
                    for ra_i, dec_i, psi_i in zip(ra, dec, psi)
                ],
                dtype=np.float32,
            )
            fc = np.array(
                [
                    ifo.antenna_response(
                        ra_i, dec_i, self.ref_time, psi_i, mode="cross"
                    )
                    for ra_i, dec_i, psi_i in zip(ra, dec, psi)
                ],
                dtype=np.float32,
            )
            fp = fp[..., np.newaxis]
            fc = fc[..., np.newaxis]

        return fp, fc
    
    def _project_one_signal(
        self,
        *,
        h_cross,
        h_plus,
        d_ref,
        d_new,
        ra,
        dec,
        psi,
        tc_ref,
        detector_times,
    ):
        """
        Project one plus/cross polarization pair onto all detectors.

        Returns
        -------
        dict
            Mapping detector name -> frequency-domain detector strain.
        """
        assert np.allclose(tc_ref, 0.0), (
            "This should always be 0. If for some reason "
            "you want to save time shifted polarizations, "
            "then remove this assert statement."
        )

        d_ratio = self._distance_ratio(d_ref, d_new)

        hc = h_cross * d_ratio
        hp = h_plus * d_ratio

        strains = {}

        for ifo in self.ifo_list:
            # (2) project strains onto the different detectors
            # TODO the Bilby cython functions are not vectorized, so for now
            # we just loop over the extrinsic parameters. This is not ideal
            # and eventually one should also vectorize these functions to
            # achieve optimal batching capabilities.
            fp, fc = self._antenna_patterns(ifo, ra, dec, psi)

            strain = fp * hp + fc * hc

            # (3) time shift the strain. If polarizations are timeshifted by
            #     tc_ref != 0, undo this here by subtracting it from dt.
            dt = detector_times[ifo.name] - tc_ref
            strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        return strains
    
    @staticmethod
    def _sum_strains(total_strains, signal_strains):
        """
        Add detector-strain dictionaries.

        Used for overlapping signals:
            total = signal_A + signal_B + ...
        """
        if not total_strains:
            return dict(signal_strains)

        for ifo_name, strain in signal_strains.items():
            if ifo_name in total_strains:
                total_strains[ifo_name] = total_strains[ifo_name] + strain
            else:
                total_strains[ifo_name] = strain

        return total_strains
    
    @staticmethod
    def _infer_suffixes_from_parameters(parameters):
        """
        Infer signal suffixes from parameter names.

        Example:
            mass_1_A, mass_1_B -> ["_A", "_B"]

        This intentionally looks only for two-character suffixes of the form
        _<capital letter>, matching your current naming scheme.
        """
        suffixes = sorted(
            {
                key[-2:]
                for key in parameters
                if len(key) >= 2
                and key[-2] == "_"
                and key[-1].isalpha()
                and key[-1].isupper()
            }
        )

        if not suffixes:
            raise ValueError(
                "Could not infer overlapping-signal suffixes from parameters. "
                f"Available parameter keys: {list(parameters.keys())}"
            )

        return suffixes

    @staticmethod
    def _get_suffixed(mapping, base_name, suffix, *, allow_unsuffixed_fallback=True):
        """
        Get a parameter from a mapping.

        In overlap mode:
            first try f"{base_name}{suffix}",
            then optionally fall back to base_name.

        Returns
        -------
        value, key_used
        """
        suffixed_name = f"{base_name}{suffix}"

        if suffixed_name in mapping:
            return mapping[suffixed_name], suffixed_name

        if allow_unsuffixed_fallback and base_name in mapping:
            return mapping[base_name], base_name

        raise KeyError(
            f"Missing parameter {suffixed_name!r}. "
            f"Fallback key {base_name!r} "
            f"{'was also absent' if allow_unsuffixed_fallback else 'was disabled'}."
        )
    
    # ------------------------------------------------------------------
    # signal projection path
    # ------------------------------------------------------------------

    def _call_single_signal(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        try:
            d_ref = parameters["luminosity_distance"]
            d_new = extrinsic_parameters.pop("luminosity_distance")
            ra = extrinsic_parameters.pop("ra")
            dec = extrinsic_parameters.pop("dec")
            psi = extrinsic_parameters.pop("psi")
            tc_ref = parameters["geocent_time"]
            assert np.allclose(tc_ref, 0.0), (
                "This should always be 0. If for some reason "
                "you want to save time shifted polarizations,"
                " then remove this assert statement."
            )
            tc_new = extrinsic_parameters.pop("geocent_time")
            detector_times = {
                ifo.name: extrinsic_parameters[f"{ifo.name}_time"]
                for ifo in self.ifo_list
            }
        except:
            raise ValueError("Missing parameters.")

        strains = self._project_one_signal(
            h_cross=sample["waveform"]["h_cross"],
            h_plus=sample["waveform"]["h_plus"],
            d_ref=d_ref,
            d_new=d_new,
            ra=ra,
            dec=dec,
            psi=psi,
            tc_ref=tc_ref,
            detector_times=detector_times,
        )
        # Add extrinsic parameters corresponding to the transformations
        # applied in the loop above to parameters. These have all been popped off of
        # extrinsic_parameters, so they only live one place.
        parameters["ra"] = ra
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["geocent_time"] = tc_new
        for ifo in self.ifo_list:
            param_name = f"{ifo.name}_time"
            parameters[param_name] = extrinsic_parameters.pop(param_name)

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample

    def _call_overlapping_signals(self, input_sample):
        sample = input_sample.copy()

        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        waveform = sample["waveform"]

        suffixes = self._infer_suffixes_from_parameters(parameters)

        total_strains = {}

        # We delay popping extrinsic parameters until all signals have been
        # processed. This matters when unsuffixed extrinsics are shared by A/B.
        consumed_extrinsic_keys = set()

        for suffix in suffixes:
            try:
                d_ref = parameters[f"luminosity_distance{suffix}"]
                tc_ref = parameters[f"geocent_time{suffix}"]

                d_new, d_new_key = self._get_suffixed(
                    extrinsic_parameters,
                    "luminosity_distance",
                    suffix,
                    allow_unsuffixed_fallback=True,
                )
                ra, ra_key = self._get_suffixed(
                    extrinsic_parameters,
                    "ra",
                    suffix,
                    allow_unsuffixed_fallback=True,
                )
                dec, dec_key = self._get_suffixed(
                    extrinsic_parameters,
                    "dec",
                    suffix,
                    allow_unsuffixed_fallback=True,
                )
                psi, psi_key = self._get_suffixed(
                    extrinsic_parameters,
                    "psi",
                    suffix,
                    allow_unsuffixed_fallback=True,
                )
                tc_new, tc_new_key = self._get_suffixed(
                    extrinsic_parameters,
                    "geocent_time",
                    suffix,
                    allow_unsuffixed_fallback=True,
                )

                detector_times = {}
                detector_time_keys = {}

                for ifo in self.ifo_list:
                    base_name = f"{ifo.name}_time"
                    detector_times[ifo.name], detector_time_keys[ifo.name] = (
                        self._get_suffixed(
                            extrinsic_parameters,
                            base_name,
                            suffix,
                            allow_unsuffixed_fallback=True,
                        )
                    )

                h_cross = waveform[f"h_cross{suffix}"]
                h_plus = waveform[f"h_plus{suffix}"]

            except Exception as exc:
                raise ValueError(
                    f"Missing parameters for overlapping signal {suffix!r}. "
                    f"Parameter keys: {list(parameters.keys())}. "
                    f"Extrinsic keys: {list(extrinsic_parameters.keys())}. "
                    f"Waveform keys: {list(waveform.keys())}."
                ) from exc

            signal_strains = self._project_one_signal(
                h_cross=h_cross,
                h_plus=h_plus,
                d_ref=d_ref,
                d_new=d_new,
                ra=ra,
                dec=dec,
                psi=psi,
                tc_ref=tc_ref,
                detector_times=detector_times,
            )

            total_strains = self._sum_strains(total_strains, signal_strains)

            # Store the actually used values under suffixed names, so downstream
            # code sees a coherent overlap parameterization.
            parameters[f"luminosity_distance{suffix}"] = d_new
            parameters[f"ra{suffix}"] = ra
            parameters[f"dec{suffix}"] = dec
            parameters[f"psi{suffix}"] = psi
            parameters[f"geocent_time{suffix}"] = tc_new

            for ifo in self.ifo_list:
                parameters[f"{ifo.name}_time{suffix}"] = detector_times[ifo.name]

            consumed_extrinsic_keys.update(
                [d_new_key, ra_key, dec_key, psi_key, tc_new_key]
            )
            consumed_extrinsic_keys.update(detector_time_keys.values())

        # Remove extrinsic parameters corresponding to the transformations
        # applied above. This mirrors the single-signal behavior where these
        # values are moved from extrinsic_parameters into parameters.
        for key in consumed_extrinsic_keys:
            extrinsic_parameters.pop(key, None)

        sample["waveform"] = total_strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample
        
class TimeShiftStrain(object):
    """
    Time shift the strains in the individual detectors according to the
    times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain):
        self.ifo_list = ifo_list
        self.domain = domain

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = input_sample["extrinsic_parameters"].copy()

        strains = {}

        if isinstance(input_sample["waveform"], dict):
            for ifo in self.ifo_list:
                # time shift the strain
                strain = input_sample["waveform"][ifo.name]
                dt = extrinsic_parameters.pop(f"{ifo.name}_time")
                strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        elif isinstance(input_sample["waveform"], torch.Tensor):
            strains = input_sample["waveform"]
            dt = [extrinsic_parameters.pop(f"{ifo.name}_time") for ifo in self.ifo_list]
            dt = torch.stack(dt, 1)
            strains = self.domain.time_translate_data(strains, dt)

        else:
            raise NotImplementedError(
                f"Unexpected type {type(input_sample['waveform'])}, expected dict or "
                f"torch.Tensor"
            )

        sample["waveform"] = strains
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class SampleCalibrationParameters(object):
    r"""
    Expand out a waveform using several detector calibration draws. These multiple
    draws are intended to be used for marginalizing over calibration uncertainty.

    The calibration parameters are not known precisely, rather they are assumed to be
    normally distributed, with mean 0 and standard deviation  determined by the
    "calibration envelope", which varies from event to event.

    Therefore, for each detector waveform, this transform draws a collection of $N$
    calibration curves $\{(\delta A^n(f), \delta \phi^n(f))\}_{n=1}^N$ according to a
    calibration envelope, and applies them to generate $N$ observed waveforms $\{h^n_{
    obs}(f)\}$. This is intended to be used for marginalizing over the calibration
    uncertainty when evaluating the likelihood for importance sampling.
    
    This transform should be followed by ApplyCalibrationToWaveform to apply the
    sampled calibration curves to the waveform.
    """

    def __init__(
        self,
        ifo_list,
        data_domain,
        calibration_envelope,
        num_calibration_curves,
        num_calibration_nodes,
        correction_type="data",
    ):
        r"""
        Parameters
        ----------
        ifo_list : InterferometerList
            List of Interferometers present in the analysis.
        data_domain : Domain
            Domain on which data is defined.
        calibration_envelope : dict
            Dictionary of the form ``{"H1": filepath, "L1": filepath}``,
            where the filepaths are strings pointing to ".txt" files containing
            calibration envelopes.
        num_calibration_curves : int
            Number of calibration curves to sample.
        num_calibration_nodes : int
            Number of log-spaced frequency nodes for the spline.
        correction_type : str = "data"
            Whether envelopes are over eta ("data") or alpha ("template").
        """
        self.ifo_list = ifo_list
        self.num_calibration_curves = num_calibration_curves
        self.data_domain = data_domain

        if correction_type is None:
            correction_type_dict = {
                ifo.name: CALIBRATION_CORRECTION_TYPE_LOOKUP[ifo.name] for ifo in self.ifo_list
            }
        elif correction_type == "data" or correction_type == "template":
            correction_type_dict = {ifo.name: correction_type for ifo in self.ifo_list}
        elif isinstance(correction_type, dict):
            correction_type_dict = correction_type
        else:
            raise Exception(f"{correction_type} not understood")

        self.calibration_prior = {}
        if all([s.endswith(".txt") for s in calibration_envelope.values()]):
            self.calibration_envelope = calibration_envelope
            for ifo in self.ifo_list:
                # Setting a calibration prior. 
                # Take the calibration envelope and use it to set a spline on
                # the median and sigma of the amplitude and phase. Then in log
                # frequency it will setup node points at frequency points, f_i
                # where i is given from 0...num_calibration_nodes and f_i are log
                # spaced between f_min and f_max. Then for each node point f_i,
                # it will create a gaussian prior according to the spline of
                # the median and sigma found earlier
                self.calibration_prior[
                    ifo.name
                ] = CalibrationPriorDict.from_envelope_file(
                    self.calibration_envelope[ifo.name],
                    self.data_domain.f_min,
                    self.data_domain.f_max,
                    num_calibration_nodes,
                    ifo.name,
                    correction_type=correction_type_dict[ifo.name],
                )
        else:
            raise Exception("Calibration envelope must be specified in a .txt file!")

    def __call__(self, input_sample):
        sample = input_sample.copy()

        if "extrinsic_parameters" not in sample:
            sample["extrinsic_parameters"] = {}

        for ifo in self.ifo_list:
            # Sample calibration parameters from prior
            # Returns dict like {"recalib_H1_amplitude_0": array, ...}
            draws = self.calibration_prior[ifo.name].sample(self.num_calibration_curves)

            # Add each parameter to extrinsic_parameters
            for param_name, values in draws.items():
                sample["extrinsic_parameters"][param_name] = np.array(values)

        return sample


class ApplyCalibrationToWaveform(object):
    r"""
    Apply calibration correction to the waveform based on calibration parameters
    in extrinsic_parameters.

    Detector calibration uncertainty is modeled as described in
    https://dcc.ligo.org/LIGO-T1400682/public

    Gravitational wave data $d$ is assumed to be of the form

    $$d(f) = h_{obs}(f) + n(f),$$

    where $h_{obs}$ is the observed waveform and $n$ is the noise. Since the detector
    is not perfectly calibrated, the observed waveform is not identical to the true
    waveform $h(f)$. Rather, it is assumed to have corrections of the form

    $$h_{obs}(f) = h(f) * (1 + \delta A(f)) * \exp(i \delta \phi(f)) = h(f) * \alpha(f),$$

    where $\delta A(f)$ and $\delta \phi(f)$ are frequency-dependent amplitude and
    phase errors. Under the calibration model, these are parametrized with cubic
    splines, defined in terms of calibration parameters $A_i$ and $\phi_i$, defined
    at log-spaced frequency nodes,

    $$
    \delta A(f) &= \mathrm{spline}(f; {f_i, \delta A_i}), \\
    \delta \phi(f) &= \mathrm{spline}(f; {f_i, \delta \phi_i}).
    $$

    Calibration parameters (\delta A, \delta \phi) should be in
    sample["extrinsic_parameters"] with keys like "recalib_H1_amplitude_0",
    "recalib_H1_phase_0", etc.

    - If values are arrays of shape (N,), applies N calibration curves, adding a
      leading dimension to the waveform.
    - If values are scalars, applies a single calibration curve.
    - If no calibration parameters found, passes through unchanged.

    The calibration spline model is lazily initialized on the first call, inferring
    num_calibration_nodes from the number of calibration parameters present.
    """

    def __init__(
        self,
        ifo_list,
        data_domain,
    ):
        r"""
        Parameters
        ----------
        ifo_list : InterferometerList
            List of Interferometers present in the analysis.
        data_domain : Domain
            Domain on which data is defined.
        """
        self.ifo_list = ifo_list
        self.data_domain = data_domain

    def _ensure_calibration_model(self, ifo, num_calibration_nodes):
        """
        Ensure the calibration model is set up on the ifo. Creates it if not present
        or if it has a different number of nodes.
        """
        if not hasattr(ifo, "calibration_model") or ifo.calibration_model is None or isinstance(ifo.calibration_model, calibration.Recalibrate):
            # using https://dcc.ligo.org/LIGO-T2300140 
            ifo.calibration_model = calibration.CubicSpline(
                f"recalib_{ifo.name}_",
                minimum_frequency=self.data_domain.f_min,
                maximum_frequency=self.data_domain.f_max,
                n_points=num_calibration_nodes,
            )

    def __call__(self, input_sample):
        sample = input_sample.copy()

        extrinsic = sample.get("extrinsic_parameters", {})

        # Check if any recalib parameters are present
        if not any(k.startswith("recalib_") for k in extrinsic.keys()):
            return sample

        for ifo in self.ifo_list:
            prefix = f"recalib_{ifo.name}_"

            # Extract calibration parameters for this ifo
            calib_params = {
                k: v for k, v in extrinsic.items() if k.startswith(prefix)
            }

            if not calib_params:
                continue

            # Infer num_calibration_nodes from the number of amplitude parameters
            # Parameters are named like recalib_H1_amplitude_0, recalib_H1_amplitude_1, ...
            amplitude_params = [k for k in calib_params if "amplitude" in k]
            num_calibration_nodes = len(amplitude_params)

            # Ensure calibration model is set up (lazy initialization)
            self._ensure_calibration_model(ifo, num_calibration_nodes)

            # Check if parameters are scalars and promote to arrays if so
            first_param = next(iter(calib_params.values()))
            is_scalar = np.isscalar(first_param)
            if is_scalar:
                calib_params = {k: np.array([v]) for k, v in calib_params.items()}

            num_curves = len(next(iter(calib_params.values())))

            # Initialize calibration draws array
            calibration_draws = np.zeros(
                (num_curves, len(self.data_domain.sample_frequencies)),
                dtype=complex,
            )

            # Compute calibration curve for each parameter set
            for i in range(num_curves):
                params_i = {k: v[i] for k, v in calib_params.items()}
                calibration_draws[
                    i, self.data_domain.frequency_mask
                ] = ifo.calibration_model.get_calibration_factor(
                    self.data_domain.sample_frequencies[self.data_domain.frequency_mask],
                    prefix=prefix,
                    **params_i,
                )

            # Squeeze out leading dimension if input was scalar
            if is_scalar:
                calibration_draws = calibration_draws.squeeze(axis=0)

            # Multiplying the sample waveform in the interferometer according to
            # the calibration curve.  This is done by following the perscription
            # here:
            #
            # https://dcc.ligo.org/LIGO-T1400682 Eq 3 and 4
            #
            # We take the waveform h(f) and multiply it by C = (1 + \delta A(f))
            # \exp(i \delta \psi) i.e. h_obs(f) = C * h(f)
            # Here C is "calibration_draws"

            sample["waveform"][ifo.name] = (
                sample["waveform"][ifo.name] * calibration_draws
            )

        return sample
