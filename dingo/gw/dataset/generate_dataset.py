import argparse
import copy
import textwrap
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple
from functools import partial

import numpy as np
import pandas as pd
import yaml
from bilby.gw.prior import BBHPriorDict
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.SVD import ApplySVD, SVDBasis
from dingo.gw.transforms import WhitenFixedASD
from dingo.gw.waveform_generator import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator,
    generate_waveforms_parallel,
)
from dingo.core.utils.misc import call_func_strict_output_dim
import string
import re

def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int,
    num_signals:int = 1
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int
    num_signals : int the amount of signals, default = 1

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    print("Generating dataset of size " + str(num_samples))
    parameters      = pd.DataFrame(prior.sample(num_samples))
    if num_signals > 1:
        parameters      = _break_symmetry(parameters)
    suffixes        = _extract_signal_suffixes(parameters.columns,num_signals = num_signals)
    splitted_params = _split_by_suffix(parameters,suffixes,num_signals = num_signals)
    
    polarization_blocks = {}
    failed_masks = []
    for suffix,parameters_split in splitted_params.items():
        if num_processes > 1:
            with threadpool_limits(limits=1, user_api="blas"):
                with Pool(processes=num_processes) as pool:
                    polarizations_split  = generate_waveforms_parallel(
                        waveform_generator, parameters_split, pool
                    )
        else:
            polarizations_split  = generate_waveforms_parallel(waveform_generator, parameters_split)
        
        polarization_blocks[suffix] = polarizations_split
        
        # Find cases where waveform generation failed and only return data for successful ones
        wf_failed_s = np.any(np.isnan(polarizations_split["h_plus"]), axis=1)
        failed_masks.append(wf_failed_s)

    # A joint sample fails if any of its component waveforms failed.
    wf_failed           = np.logical_or.reduce(failed_masks)
    polarizations_joint = _recombine_polarizations_with_suffix(polarization_blocks,num_signals = num_signals)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations_joint.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        print(
            f"{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate."
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(parameters.iloc[idx_failed])
        print(
            f"Only returning the {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok
    return parameters, polarizations_joint

def _recombine_polarizations_with_suffix(
    polarization_blocks: dict[str, dict[str, np.ndarray]],
    num_signals = 1
) -> dict[str, np.ndarray]:
    if num_signals == 1:
       return polarization_blocks.get("")
    out = {}

    for suffix, polarizations in polarization_blocks.items():
        for key, value in polarizations.items():
            out[f"{key}{suffix}"] = value

    return out

def _break_symmetry(parameters):
    #break symmetry for overlapping signals
    parameters["geocent_time_B"] = (
        parameters["geocent_time_A"] + parameters["delta_t_AB"]
    )
    parameters = parameters.drop(columns=["delta_t_AB"]) #remove old column
    return parameters

def _extract_signal_suffixes(columns, pattern=r"_[A-Z]$",num_signals = 1):
    """
    Extract signal suffixes from parameter names.

    Example:
        ["mass_1_A", "mass_1_B", "delta_t_AB"]
        -> ["_A", "_B"]

    Joint parameters like delta_t_AB are ignored because they do not end
    in a single signal suffix.
    default to set of empty suffix str if number of signals  = 1
    """
    if num_signals == 1:
        return sorted(set(""))
    suffixes = set()

    for col in columns:
        match = re.search(pattern, col)
        if match is not None:
            suffixes.add(match.group(0))

    return sorted(suffixes)

def _split_by_suffix(df: pd.DataFrame, suffixes=None,num_signals = 1) -> dict[str, pd.DataFrame]:
    """
    Split a joint suffixed DataFrame into unsuffixed sub-DataFrames.

    Example:
        mass_1_A, mass_2_A -> mass_1, mass_2
        mass_1_B, mass_2_B -> mass_1, mass_2

    Returns:
        {
            "_A": df_A_unsuffixed,
            "_B": df_B_unsuffixed,
        }
    """
    
    if num_signals == 1:
        return {df}
    
    if suffixes is None:
        suffixes = _extract_signal_suffixes(df.columns)
    
    out = {}

    for suffix in suffixes:
        cols = [c for c in df.columns if c.endswith(suffix)]

        if not cols:
            raise ValueError(f"No columns found for suffix {suffix!r}.")

        rename = {
            c: c[:-len(suffix)]
            for c in cols
        }

        out[suffix] = df[cols].rename(columns=rename).copy()

    return out

def train_svd_basis(dataset: WaveformDataset, size: int, n_train: int):
    """
    Train (and optionally validate) an SVD basis.

    Parameters
    ----------
    dataset : WaveformDataset
        Contains waveforms to be used for building SVD.
    size : int
        Number of elements to keep for the SVD basis.
    n_train : int
        Number of training waveforms to use. Remaining are used for validation. Note
        that the actual number of training waveforms is n_train * len(polarizations),
        since there is one waveform used for each polarization.

    Returns
    -------
    SVDBasis, n_train, n_test
        Since EOB waveforms can fail to generate, provide also the number used in
        training and validation.
    """
    # Prepare data for training and validation.
    train_data = np.vstack([val[:n_train] for val in dataset.polarizations.values()])
    test_data = np.vstack([val[n_train:] for val in dataset.polarizations.values()])
    test_parameters = pd.concat(
        [
            # I would like to save the polarization, but saving the dataframe with
            # string columns causes problems. Fix this later.
            # dataset.parameters.iloc[n_train:].assign(polarization=pol)
            dataset.parameters.iloc[n_train:]
            for pol in dataset.polarizations
        ]
    )
    test_parameters.reset_index(drop=True, inplace=True)

    print("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, size)

    assert np.allclose(basis.V[: dataset.domain.min_idx], 0)

    # Since there is a possibility that the size of the dataset returned by
    # generate_parameters_and_polarizations is smaller than requested, we don't assume
    # that there are n_test samples. Instead we just look at the size of the test
    # dataset.
    if test_data.size != 0:
        basis.compute_test_mismatches(
            test_data, parameters=test_parameters, verbose=True
        )

    # Return also the true number of samples. Some EOB waveforms may have failed to
    # generate, so this could be smaller than the number requested.
    n_ifos = len(dataset.polarizations)
    n_train = len(train_data) // n_ifos
    n_test = len(test_data) // n_ifos

    return basis, n_train, n_test

def generate_dataset(settings: Dict, num_processes: int, num_signals:int = 1) -> WaveformDataset:
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int
    num_signals : int

    Returns
    -------
    A WaveformDataset based on the settings.
    """

    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    domain = build_domain(settings["domain"])

    new_interface_flag = settings["waveform_generator"].get("new_interface", False)
    if new_interface_flag:
        waveform_generator = NewInterfaceWaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
    else:
        waveform_generator = WaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )

    dataset_dict = {"settings": settings}

    if "compression" in settings:
        compression_transforms = []

        if "whitening" in settings["compression"]:
            compression_transforms.append(
                WhitenFixedASD(
                    domain,
                    asd_file=settings["compression"]["whitening"],
                    inverse=False,
                )
            )

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]

            # Load an SVD basis from file, if specified.
            if "file" in svd_settings:
                basis = SVDBasis(file_name=svd_settings["file"])

            # Otherwise, generate the basis based on simulated waveforms.
            else:
                # If using whitened waveforms, then the SVD should be based on these.
                waveform_generator.transform = Compose(compression_transforms)

                n_train = svd_settings["num_training_samples"]
                n_test = svd_settings.get("num_validation_samples", 0)

                func = partial(
                    generate_parameters_and_polarizations,
                    waveform_generator,
                    prior,
                    num_processes=num_processes,
                )
                parameters, polarizations = call_func_strict_output_dim(
                    func, n_train + n_test
                )
                svd_dataset_settings = copy.deepcopy(settings)
                svd_dataset_settings["num_samples"] = len(parameters)
                del svd_dataset_settings["compression"]["svd"]

                # We build a WaveformDataset containing the SVD-training waveforms
                # because when constructed, it will automatically zero the waveforms
                # below f_min. This is useful for EOB waveforms, which are Fourier
                # transformed from time domain, and hence are nonzero below f_min. The
                # waveforms need to be zeroed below f_min because this corresponds to
                # setting the lower bound of the likelihood integral.

                svd_dataset = WaveformDataset(
                    dictionary={
                        "parameters": parameters,
                        "polarizations": polarizations,
                        "settings": svd_dataset_settings,
                    }
                )
                basis, n_train, n_test = train_svd_basis(
                    svd_dataset, svd_settings["size"], n_train
                )

            compression_transforms.append(ApplySVD(basis))
            dataset_dict["svd"] = basis.to_dictionary()

        waveform_generator.transform = Compose(compression_transforms)

    func = partial(
        generate_parameters_and_polarizations,
        waveform_generator,
        prior,
        num_processes=num_processes,
        num_signals=num_signals
    )
    
    parameters, polarizations = call_func_strict_output_dim(
        func,
        settings["num_samples"],
    )
    
    # ------------------------------------------------------------------
    # Standard DINGO path: keep this exactly backwards compatible.
    # ------------------------------------------------------------------
    if num_signals == 1:
        
        dataset_dict["parameters"] = parameters
        dataset_dict["polarizations"] = polarizations
        dataset_dict[settings["num_samples"]] = len(parameters)
        dataset = WaveformDataset(dictionary=dataset_dict)
        return dataset
    
    # ------------------------------------------------------------------
    # Multi-signal / overlapping-source path.
    # ------------------------------------------------------------------
    
    else:         
        overlap_settings = copy.deepcopy(settings)
        overlap_settings["num_signals"] = num_signals
        overlap_settings["signal_suffixes"] = make_signal_suffixes(num_signals)
        overlap_settings["overlapping_signals"] = True    
        
        dataset_dict["settings"] = overlap_settings
        dataset_dict["parameters"] = parameters
        dataset_dict["polarizations"] = polarizations
        dataset_dict["num_samples"] = len(parameters)

        dataset = WaveformDataset(dictionary=dataset_dict)
            
        return dataset

def make_signal_suffixes(num_signals: int):
    """
    Helper function used for generating suffixes for the naming of multiple signals
    Args:
        num_signals (int):  the number of signals

    Raises:
        ValueError: num_signals  must be strictly larger than 1

    Returns:
        list: suffixes to be used
    """
    if num_signals <= 1:
        raise ValueError("num_signals must be >= 1")

    if num_signals <= 26:
        return [f"_{letter}" for letter in string.ascii_uppercase[:num_signals]]

    # Fallback for more than 26 signals.
    return [f"_S{i}" for i in range(num_signals)]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a waveform dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--num_signals",
        type=int,
        default=1,
        help="Number of overlapping signals to generate per dataset item. Default: 1.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    return parser.parse_args()

def _generate_dataset_main(
    settings_file: str, out_file: str, num_processes: int, num_signals: int = 1
) -> None:
    if not Path(settings_file).is_file():
        raise FileNotFoundError(f"dataset generation, failed to find {settings_file}")
    if not Path(out_file).parent.is_dir():
        raise FileNotFoundError(
            f"dataset generation: can not create {out_file}: "
            f"{Path(out_file).parent} does not exist"
        )
    # Load settings
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(settings, num_processes, num_signals = num_signals)
    dataset.to_file(str(out_file))

def main() -> None:
    args = parse_args()
    _generate_dataset_main(args.settings_file, args.out_file, args.num_processes,args.num_signals)

if __name__ == "__main__":
    main()
