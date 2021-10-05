"""Functions that create samples."""
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
    n_samples=30,
    seed=123,
    M="None",
    sampling_method="random",
    MC_method="Brute force",
):
    """Simulate samples of qoi.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    seed : int
        Seed for the random number generators.
    M : int
        The number of conditional bins to genetate if `MC_method` is "DLR".
    sampling_method : string
        Specifies which sampling method should be employed. Possible arguments
        are in {"random", "grid", "chebyshev", "korobov","sobol", "halton",
        "hammersley", "latin_hypercube"}
    MC_method : string
        Specify the Monte Carlo estimator. One of ["brute force", "DLR"],
        where "DLR" denotes to the double loop reordering approach.

    Returns
    -------
    input_x_respy: list
        A list of input parameters that are ready to be passed into the
        `respy` function.
    input_x_mix_respy: list
        A list of conditional input parameters that are ready to be passed
        into the `respy` function.
    """

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
    x_3 = subset_params(sample_x)
    x_prime_3 = subset_params(sample_x_prime)
    x = fix_true_params(x_3, mean)

    # get conditional samples
    x_mix_3 = conditional_samples(x_3, x_prime_3, MC_method, M)

    # fix parameters of interest
    x_mix = fix_true_params_mix(x_mix_3, mean, MC_method)

    input_x_respy = [(params_to_respy)(i) for i in x]
    input_x_mix_respy = [(params_to_respy)(z) for x in x_mix for y in x for z in y]

    return input_x_respy, input_x_mix_respy


def load_mean_and_cov():
    """Return mean and covariance for Keane and Wolpin (1994) model."""
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
    sampling_method,
):
    """Generate two independent groups of sample points.

    Parameters
    ----------
    mean : pd.DataFrame or np.ndarray
        The mean, of shape (k, ).
    cov : pd.DataFrame or np.ndarrary
        The covariance, has to be of shape (k, k).
    n_samples : int
        Number of samples to draw.
    seed : int, optional
        Random number generator seed.
    sampling_method : string
        Specifies which sampling method should be employed. Possible arguments
        are in {"random", "grid", "chebyshev", "korobov","sobol", "halton",
        "hammersley", "latin_hypercube"}

    Returns
    -------
    sample_x, sample_x_prime : np.ndarray
        Two arrays of shape (n_draws, n_params) with i.i.d draws from a
        given joint distribution.

    """
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


def subset_params(x):
    """Pick a subset of samples from the sampled parameters.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_draws, n_params).

    Returns
    -------
    params_interests : np.ndarray
        Array of shape (n_draws, 3) contains only 3 seleted parameters.

    """

    n_draws = x.shape[0]

    indices = [2, 14, 16]
    params_interests = np.zeros((n_draws, 3))

    for i in range(n_draws):
        params_interests[i] = np.take(x[i], indices)

    return params_interests


def conditional_samples(x_3, x_prime_3, MC_method, M):
    """Generate mixed sample sets of interest distributed accroding to a conditional PDF.

    Parameters
    ----------
    x_3 : np.ndarray
        Array with shape (n_draws, 3).
    x_prime : np.ndarray
        Array with shape (n_draws, 3).
    MC_method : string
        Specify the Monte Carlo estimator. One of ["brute force", "DLR"],
        where "DLR" denotes to the double loop reordering approach.
    M : int
        The number of conditional bins to genetate if `MC_method` is "DLR".

    Returns
    -------
    x_mix :  np.ndarray
        Mixed sample sets. Shape has the form (n_draws, 3, n_draws, 3).

    """

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


def fix_true_params(x_3, true_values):
    """Replace the 3 selected point estimates with the sampled parameters.

    Parameters
    ----------
    x_3 : np.ndarray
        Array with shape (n_draws, 3).
    true_values : np.ndarray
        The point estimated, of shape (k, ).

    Returns
    -------
    true_params_fix :  np.ndarray
        Shape has the form (n_draws, n_params, n_draws, n_params).

    """

    n_draws = x_3.shape[0]

    true_params_fix = np.tile(true_values, (n_draws, 1))

    for i in range(n_draws):
        np.put(true_params_fix[i], [2, 14, 16], x_3[i])

    return true_params_fix


def fix_true_params_mix(x_3, true_values, MC_method):
    """Replace the 3 selected point estimates with the conditional sampled parameters.

    Parameters
    ----------
    x_3 : np.ndarray
        Array with shape (n_draws, 3).
    true_values : np.ndarray
        The point estimated, of shape (k, ).

    Returns
    -------
    true_params_fix :  np.ndarray
        Shape has the form (n_draws, n_params, n_draws, n_params).

    """

    if MC_method == "Brute force":

        n_draws, n_3_parmas = x_3.shape[:2]

        true_params_fix = np.tile(true_values, (n_draws, n_3_parmas, n_draws, 1))

        for i in range(n_draws):
            for j in range(n_3_parmas):
                for k in range(n_draws):
                    np.put(true_params_fix[i, j, k], [2, 14, 16], x_3[i, j, k])

    if MC_method == "DLR":
        M, n_3_parmas, n_draws = x_3.shape[:3]

        true_params_fix = np.tile(true_values, (M, n_3_parmas, n_draws, 1))

        for i in range(M):
            for j in range(n_3_parmas):
                for k in range(n_draws):
                    np.put(true_params_fix[i, j, k], [2, 14, 16], x_3[i, j, k])
    return true_params_fix


def params_to_respy(input_params, *args):
    """transfer sampled paramters to respy format."""

    # baseline options and params for the indices.
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
