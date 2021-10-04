"""Functions that create samples for QBSM."""
import chaospy as cp
import numpy as np
import respy as rp
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"

CHAOSPY_SAMPLING_METHODS = {
    "random",
    "grid",
    "chebyshev",
    "korobov",
    "sobol",
    "halton",
    "hammersley",
    "latin_hypercube",
}


def create_sample(
    n_samples,
    seed,
    M,
    sampling_method="random",
    MC_method="Brute force",
):
    # load mean and cov
    mean, cov = load_mean_and_cov()

    # get unconditioal samples
    sample_x, sample_x_prime = unconditional_samples(
        mean,
        cov,
        n_samples,
        seed,
        sampling_method,
    )

    # fix parameters of interest
    x_3 = substract_params(sample_x)
    x_prime_3 = substract_params(sample_x_prime)
    x = fix_true_params(x_3, mean)

    # get conditional samples
    x_mix_3 = conditional_samples(x_3, x_prime_3, MC_method, M)

    # fix parameters of interest
    x_mix = fix_true_params_mix(x_mix_3, mean, MC_method)

    input_x_respy = [(params_to_respy)(i) for i in x]
    input_x_mix_respy = [(params_to_respy)(z) for x in x_mix for y in x for z in y]

    return input_x_respy, input_x_mix_respy


def load_mean_and_cov():
    # load model specifications
    base_params = pd.read_pickle(DATA_PATH / "params_kw_94_one_se.pkl")

    # mean and cov for sampling
    mean = base_params["value"].to_numpy()[:27]
    cov = pd.read_pickle(DATA_PATH / "covariance_kw_94_one.pkl").to_numpy()

    return mean, cov


def unconditional_samples(
    mean,
    cov,
    n_samples,
    seed,
    sampling_method="random",
):

    distribution = cp.MvNormal(loc=mean, scale=cov)

    if sampling_method in CHAOSPY_SAMPLING_METHODS:
        np.random.seed(seed)
        sample_x = np.array(distribution.sample(size=n_samples, rule=sampling_method).T)

        np.random.seed(seed + 1)
        sample_x_prime = np.array(
            distribution.sample(size=n_samples, rule=sampling_method).T
        )
    else:
        raise ValueError(f"Argument 'method' is not in {CHAOSPY_SAMPLING_METHODS}.")

    return sample_x, sample_x_prime


def substract_params(x):

    n_draws = x.shape[0]

    indices = [2, 14, 16]
    params_interests = np.zeros((n_draws, 3))

    for i in range(n_draws):
        params_interests[i] = np.take(x[i], indices)

    return params_interests


def conditional_samples(x_3, x_prime_3, MC_method, M):

    n_draws, n_params = x_3.shape

    if MC_method == "Brute force":
        x_3_mix = np.zeros((n_draws, n_params, n_draws, n_params))

        for i in range(n_params):
            for j in range(n_draws):
                x_3_mix[j, i] = x_3
                x_3_mix[j, i, :, i] = x_prime_3[j, i]

    if MC_method == "DLR":
        conditional_bin = x_3[:M]
        x_3_mix = np.zeros((M, n_params, n_draws, n_params))

        # subdivide unconditional samples into M eaually bins,
        # within each bin x_i being fixed.
        for i in range(n_params):
            for j in range(M):
                x_3_mix[j, i] = x_3
                x_3_mix[j, i, :, i] = conditional_bin[j, i]

    return x_3_mix


def fix_true_params(x, true_values):

    n_draws = x.shape[0]

    true_params_fix = np.tile(true_values, (n_draws, 1))

    for i in range(n_draws):
        np.put(true_params_fix[i], [2, 14, 16], x[i])

    return true_params_fix


def fix_true_params_mix(x, true_values, MC_method):

    if MC_method == "Brute force":

        n_draws, n_3_parmas = x.shape[:2]

        true_params_fix = np.tile(true_values, (n_draws, n_3_parmas, n_draws, 1))

        for i in range(n_draws):
            for j in range(n_3_parmas):
                for k in range(n_draws):
                    np.put(true_params_fix[i, j, k], [2, 14, 16], x[i, j, k])

    if MC_method == "DLR":
        M, n_3_parmas, n_draws = x.shape[:3]

        true_params_fix = np.tile(true_values, (M, n_3_parmas, n_draws, 1))

        for i in range(M):
            for j in range(n_3_parmas):
                for k in range(n_draws):
                    np.put(true_params_fix[i, j, k], [2, 14, 16], x[i, j, k])
    return true_params_fix


def params_to_respy(input_params, *args):
    """transfer sampled paramters to respy format."""

    # baseline options and params for the indices.
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)
    base_params = pd.read_pickle(DATA_PATH / "params_kw_94_one_se.pkl")

    params_idx = pd.Series(data=input_params, index=base_params.index[0:27])

    assert len(params_idx) == 27, "Length of KW94 vector must be 27."
    part_1 = params_idx

    rp_params, _ = rp.get_example_model("kw_94_one", with_data=False)
    part_2 = rp_params.iloc[27:31, 0]

    parts = [part_1, part_2]
    rp_params_series = pd.concat(parts)
    input_params_respy = pd.DataFrame(rp_params_series, columns=["value"])

    return input_params_respy
