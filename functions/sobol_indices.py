from functions.sampling import (
    load_mean_and_cov,
    unconditional_samples,
    subset_params,
    fix_true_params,
    params_to_respy,
)
from functions.qoi import quantitiy_of_interest

from SALib.analyze import hdmr
import numpy as np
from joblib import Parallel, delayed


def sobol_indices(
    n_samples,
    seed=123,
    func=quantitiy_of_interest,
    len_alp=31,
    sampling_method="random",
):
    """Compute sobol indices for Keane and Wolpin (1994) model.

    n_samples : int
        Number of samples to draw.
    seed : int
        Seed for the random number generators.
    len_alp :int
        The lenth of alpha grid.
    sampling_method : string
        Specifies which sampling method should be employed. Possible arguments
        are in {"random", "grid", "chebyshev", "korobov","sobol", "halton",
        "hammersley", "latin_hypercube"}

    Returns
    -------
    S_total_array : np.ndarray
        Sobol indices of interested parameters which are broadcast to quantile points.

    """

    x_3, input_x_respy = _sobol_inputs(n_samples, seed, sampling_method=sampling_method)

    y_array = np.array(_unconditional_y(input_x_respy, func))

    # This is so SALib understands your model inputs
    problem = {
        "num_vars": 3,  # number of parameters
        # Names of your parameters
        "names": ["alpha_{11}", "beta_{1}", "gamma_{0}"],
    }

    Si = hdmr.analyze(problem, x_3, y_array)
    S_total = Si["ST"][0:3]

    S_total_array = np.tile(S_total, (len_alp, 1))

    return S_total_array


def _sobol_inputs(n_samples, seed, sampling_method):
    """Generate inputs for computing sobol indices"""
    # load mean and cov
    mean, cov = load_mean_and_cov()
    # get unconditioal samples
    sample_x, _ = unconditional_samples(
        mean,
        cov,
        n_samples,
        seed,
        sampling_method,
    )
    # fix parameters of interest
    x_3 = subset_params(sample_x)
    x = fix_true_params(x_3, mean)
    input_x_respy = [(params_to_respy)(i) for i in x]

    return x_3, input_x_respy


def _unconditional_y(x, func):
    """Compute qoi for sobol indices"""
    # Equation 21a
    y_x = Parallel(n_jobs=8)(delayed(func)(i) for i in x)

    return y_x
