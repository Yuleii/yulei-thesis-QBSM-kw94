"""Functions that compute quantile-based sensitivity measures."""
import numpy as np


def quantile_measures(quantile_y_x, quantile_y_x_mix):
    """Estimate the values of quantile based measures."""
    m, n_params, len_alp = quantile_y_x_mix.shape[:3]

    # initialization
    q_1 = np.zeros((len_alp, n_params))
    q_2 = np.zeros((len_alp, n_params))
    delt = np.zeros((m, n_params, len_alp, 1))

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
