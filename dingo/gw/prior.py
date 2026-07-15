from copy import deepcopy

from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import (
    fill_from_fixed_priors,
    convert_to_lal_binary_black_hole_parameters,
)
from bilby.core.prior import Uniform, Sine, Cosine

import numpy as np
from typing import Dict, Set, Any
import warnings
from bilby.core.prior import PriorDict
# Silence INFO and WARNING messages from bilby
import logging

logging.getLogger("bilby").setLevel("ERROR")


class BBHExtrinsicPriorDict(BBHPriorDict):   
    """
    This class is the same as BBHPriorDict except that it does not require mass parameters.

    It also contains a method for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """
    def __init__(
        self,
        dictionary=None,
        filename=None,
        conversion_function=None,
        modeIsSingle = True,
        reconstruct_geocent_time_B=True,
        time_reference="A_minus_delta",
    ):
        self.modeIsSingle = modeIsSingle
        print(f"$$$ BBHExtrinsicPriorDict modeIsSingle: {modeIsSingle}")
        self.reconstruct_geocent_time_B = reconstruct_geocent_time_B
        self.time_reference = time_reference

        if modeIsSingle:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )

        else:
            # Important: bypass BBHPriorDict because it assumes ordinary BBH names.
            PriorDict.__init__(
                self,
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
            
    def _default_conversion_function_joint(self, sample):
        out = dict(sample)

        for suffix in ["_A", "_B"]:
            sub = self._strip_suffix(sample, suffix)

            if not sub:
                continue

            # Build a single-signal prior-like object for fixed-prior filling.
            # In the simplest version, you can first skip fill_from_fixed_priors.
            sub_converted, _ = convert_to_lal_binary_black_hole_parameters(sub)

            # Remove parameters DINGO does not want, matching the single-signal path.
            sub_converted.pop("phi_jl", None)
            sub_converted.pop("phi_12", None)

            out.update(self._add_suffix(sub_converted, suffix))

        # If you use delta_t_AB, preserve it.
        if "delta_t_AB" in sample:
            out["delta_t_AB"] = sample["delta_t_AB"]

        # If geocent_time_B was reconstructed, preserve it.
        if "geocent_time_B" in sample:
            out["geocent_time_B"] = sample["geocent_time_B"]

        return out
    def _default_conversion_function_single(self, sample):
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        if "phi_jl" in out_sample:
            del out_sample["phi_jl"]
        if "phi_12" in out_sample:
            del out_sample["phi_12"]

        return out_sample
    def default_conversion_function(self, sample):
        if self.modeIsSingle:
            return self._default_conversion_function_single(sample)
        else:
            return self._default_conversion_function_joint(sample)
    
    def sample(self, size=None):
        """
        Sample from the prior.

        In single mode, this behaves like the original BBHExtrinsicPriorDict.

        In joint mode, this additionally reconstructs derived joint parameters,
        e.g. geocent_time_B from geocent_time_A and delta_t_AB.
        """
        sample = super().sample(size=size)

        if not self.modeIsSingle:
            sample = self._postprocess_joint_sample(sample)

        return sample


    def _postprocess_joint_sample(self, sample):
        """
        Postprocess samples in joint mode.

        Currently supports the convention:

            geocent_time_B = geocent_time_A - delta_t_AB

        """

        if not self.reconstruct_geocent_time_B:
            return sample

        if "delta_t_AB" not in sample:
            return sample

        if "geocent_time_B" in sample:
            raise ValueError(
                "Both geocent_time_B and delta_t_AB are present. "
                "This is a redundant time parameterization. Use either "
                "geocent_time_B directly, or reconstruct it from delta_t_AB."
            )

        if "geocent_time_A" not in sample:
            raise KeyError(
                "Cannot reconstruct geocent_time_B because geocent_time_A "
                "is missing from the sampled joint prior."
            )
            
        #extend to allow different definiton
        if self.time_reference == "A_minus_delta":
            sample["geocent_time_B"] = (
                sample["geocent_time_A"] - sample["delta_t_AB"]
            )

        else:
            raise ValueError(
                f"Unknown time_reference={self.time_reference!r}. "
                "Use 'A_minus_delta' or 'A_plus_delta'."
            )

        return sample
    @staticmethod
    def _strip_suffix(sample, suffix):
        """
        Extract one signal's parameters and remove the suffix.
        """
        n = len(suffix)

        return {
            key[:-n]: value
            for key, value in sample.items()
            if key.endswith(suffix)
        }

    @staticmethod
    def _add_suffix(sample, suffix):
        """
        Reattach a suffix to a single-signal parameter dictionary.
        """
        return {
            f"{key}{suffix}": value
            for key, value in sample.items()
        }
    def mean_std(self, keys=([]), sample_size=50000, force_numerical=False, signal_suffixes=None):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys: list(str)
            A list of desired parameter names
        sample_size: int
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical: bool (False)
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns dictionaries for the means and standard deviations.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean = {}
        std = {}

        if not force_numerical:
            # First try to calculate analytically (works for standard priors)
            estimation_keys = []
            for key in keys:
                p = self[key]
                # A few analytic cases
                if isinstance(p, Uniform):
                    mean[key] = (p.maximum + p.minimum) / 2.0
                    std[key] = np.sqrt((p.maximum - p.minimum) ** 2.0 / 12.0).item()
                elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                    mean[key] = np.pi / 2.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample(size=sample_size)
            samples = self.default_conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key]).item()
                    std[key] = np.std(samples[key]).item()
        
        return mean, std

default_extrinsic_dict = {
    "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2, name='dec')",
    "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic", name="ra")',
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1, name='geocent_time')",
    "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic", name="psi")',
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0, name='luminosity_distance')",
}

default_intrinsic_dict = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_1')",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_2')",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0, name='mass_ratio')",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0, name='chirp_mass')",
    "luminosity_distance": 1000.0,
    "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='theta_jn')",
    "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phase")',
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_1')",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_2')",
    "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_1')",
    "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_2')",
    "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_12")',
    "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_jl")',
    "geocent_time": 0.0,
}

default_inference_parameters = [
    "chirp_mass",
    "mass_ratio",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]


def build_prior_with_defaults(prior_settings: Dict[str, str]):
    """
    Generate BBHPriorDict based on dictionary of prior settings,
    allowing for default values.

    Parameters
    ----------
    prior_settings: Dict
        A dictionary containing prior definitions for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a string for a custom prior, e.g.,
               "Uniform(minimum=10.0, maximum=80.0, name=None, latex_label=None, unit=None, boundary=None)"

    Depending on the particular prior choices the dimensionality of a
    parameter sample obtained from the returned GWPriorDict will vary.
    """

    full_prior_settings = deepcopy(prior_settings)
    for k, v in prior_settings.items():
        if v == "default":
            full_prior_settings[k] = default_intrinsic_dict[k]

    return BBHPriorDict(full_prior_settings)


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters (includes calibration parameters).
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters or "recalib" in k:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic
