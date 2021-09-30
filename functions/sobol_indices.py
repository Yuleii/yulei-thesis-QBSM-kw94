from functions.sampling import (
    load_mean_and_cov,
    unconditional_samples,
    substract_params,
    fix_true_params,
    params_to_respy,
)
from functions.qoi import quantitiy_of_interest

from SALib.analyze import hdmr
import numpy as np
from joblib import Parallel, delayed


def sobol_indices(
    n_samples,
    seed,
    len_alp,
    sampling_method="random",
):

    x_3, input_x_respy = _sobol_inputs(n_samples, seed, sampling_method=sampling_method)

    y_array = np.array(_unconditional_y(input_x_respy, quantitiy_of_interest))

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
    x_3 = substract_params(sample_x)
    x = fix_true_params(x_3, mean)
    input_x_respy = [(params_to_respy)(i) for i in x]

    return x_3, input_x_respy


def _unconditional_y(x, func):
    # Equation 21a
    y_x = Parallel(n_jobs=8)(delayed(func)(i) for i in x)

    return y_x
