import numpy as np
from sampling import create_sample
from qoi import unconditional_quantile_y, conditional_quantile_y


def compute_norm_QBSM(n_samples, seed, M, sampling_method, MC_method, alpha_grid, func):
    input_x_respy, input_x_mix_respy = create_sample(
        n_samples,
        seed,
        M,
        sampling_method,
        MC_method,
    )
    quantile_y_x = unconditional_quantile_y(input_x_respy, alpha_grid, func)
    quantile_y_x_mix = conditional_quantile_y(30, input_x_mix_respy, func, alpha_grid)
    q_1, q_2 = quantile_measures(quantile_y_x, quantile_y_x_mix)
    norm_q_1, norm_q_2 = normalized_quantile_measures(q_1, q_2)

    return norm_q_1, norm_q_2


def quantile_measures(quantile_y_x, quantile_y_x_mix):
    """Estimate the values of quantile based measures."""
    m, n_params, len_alp = quantile_y_x_mix.shape[:3]

    # initialization
    q_1 = np.zeros((len_alp, n_params))
    q_2 = np.zeros((len_alp, n_params))
    delt = np.zeros((m, n_params, len_alp, 1))

    # Equation 24&25&27&28
    for j in range(m):
        for i in range(n_params):
            for pp in range(len_alp):
                delt[j, i, pp] = quantile_y_x_mix[j, i, pp] - quantile_y_x[pp]
                q_1[pp, i] = np.mean(np.absolute(delt[:, i, pp]))
                q_2[pp, i] = np.mean(delt[:, i, pp] ** 2)

    return q_1, q_2


def normalized_quantile_measures(q_1, q_2):
    """Estimate the values of normalized quantile based measures."""
    len_alp, n_params = q_1.shape

    # initialization
    sum_q_1 = np.zeros(len_alp)
    sum_q_2 = np.zeros(len_alp)
    norm_q_1 = np.zeros((len_alp, n_params))
    norm_q_2 = np.zeros((len_alp, n_params))

    # Equation 13 & 14
    for pp in range(len_alp):
        sum_q_1[pp] = np.sum(q_1[pp, :])
        sum_q_2[pp] = np.sum(q_2[pp, :])
        for i in range(n_params):
            norm_q_1[pp, i] = q_1[pp, i] / sum_q_1[pp]
            norm_q_2[pp, i] = q_2[pp, i] / sum_q_2[pp]

    return norm_q_1, norm_q_2
