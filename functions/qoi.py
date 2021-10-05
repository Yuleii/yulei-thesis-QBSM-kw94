"""Functions that compute quantity of interest."""
import numpy as np
import respy as rp
from joblib import Parallel, delayed

import contextlib
import os
import shutil
from pathlib import Path
from time import time
from os import getpid


def unconditional_quantile_y(x, alpha_grid, func):
    """Return quantiles of outputs with unconditional input.
    Parameters
    ----------
    x : np.ndarray
        Draws from a joint distribution. Shape has the form (n_draws, n_params).

    alpha_grid : np.ndarray
        A sequence of evenly spaced values on the interval (0, 1).

    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    Returns
    -------
    quantile_y_x :  np.ndarray
        Quantiles of outputs corresponding to alpha with unconditional inputs.
        Shape has the form (len(alpha_grid),).
    """
    n_draws = len(x)

    # Equation 21a
    y_x = Parallel(n_jobs=8)(delayed(func)(i) for i in x)
    y_x_asc = np.sort(y_x)
    q_index = (np.floor(alpha_grid * n_draws)).astype(int)
    quantile_y_x = y_x_asc[q_index]

    return quantile_y_x


def conditional_quantile_y(n_samples, input_x_mix_respy, func, alpha_grid):

    """Return quantiles of outputs with conditional input.

    Parameters
    ----------
    x_mix : np.ndarray
        Mixed draws. Shape has the form (m, n_params, n_draws, n_params).
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.
    alpha_grid : np.ndarray
        A sequence of evenly spaced values on the interval (0, 1).

    Returns
    -------
    quantile_y_x_mix  :  np.ndarray
        Quantiles of output corresponding to alpha with conditional inputs. Shape has the form
        (m, n_params, len(alpha_grid), 1), where m is the number of conditional bins.

    """

    n_params = 3
    len_input = len(input_x_mix_respy)
    m = int(len_input / (n_params * n_samples))

    #     y_x_mix = np.zeros((m, n_params, n_draws, 1))
    y_x_mix_asc = np.zeros((m, n_params, n_samples, 1))
    quantile_y_x_mix = np.zeros((m, n_params, len(alpha_grid), 1))

    y_x_mix = np.array(
        Parallel(n_jobs=-1)(delayed(func)(x) for x in input_x_mix_respy for y in x)
    ).reshape(m, n_params, n_samples, 1)

    # Equation 21b/26. Get quantiles within each bin.
    for i in range(n_params):
        for j in range(m):
            # values of conditional outputs
            y_x_mix_asc[j, i] = np.sort(y_x_mix[j, i], axis=0)
            for pp, a in enumerate(alpha_grid):
                quantile_y_x_mix[j, i, pp] = y_x_mix_asc[j, i][
                    (np.floor(a * n_samples)).astype(int)
                ]  # quantiles corresponding to alpha
    return quantile_y_x_mix


def quantitiy_of_interest(params_idx_respy):
    """Simulate model and evaluate quantity of interest.

    Parameters
    ----------
    input_params : ndarray
        Unindexed input parameters.

    Returns
    -------
    change_mean_edu : float
        Quantity of Interest. Mean changes in education years of population.

    """

    _, base_options = rp.get_example_model("kw_94_one", with_data=False)

    with _temporary_working_directory():
        simulate = rp.get_simulate_func(params_idx_respy, base_options)
        base_params = params_idx_respy.copy()
        base_df = simulate(base_params)
        base_edu = base_df.groupby("Identifier")["Experience_Edu"].max().mean()

        policy_params = params_idx_respy.copy()
        policy_params.loc[("nonpec_edu", "at_least_twelve_exp_edu"), "value"] += 500
        policy_df = simulate(policy_params)
        policy_edu = policy_df.groupby("Identifier")["Experience_Edu"].max().mean()
    change_mean_edu = policy_edu - base_edu

    print("I'm process", getpid(), ":", change_mean_edu)

    return change_mean_edu


@contextlib.contextmanager
def _temporary_working_directory():
    """Changes working directory and returns to previous on exit.
    The name of the temporary directory is 'temp_process-id_timestamp'
    The directory is deleted upon exit."""
    folder_name = f"temp_{os.getpid()}_{str(time()).replace('.', '')}"
    path = Path(".").resolve() / folder_name
    path.mkdir()
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(path)
