import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append("..")

from QBSM import quantile_measures, normalized_quantile_measures
from qoi import conditional_quantile_y, unconditional_quantile_y, quantitiy_of_interest
from sampling import create_sample


def QBSM_plot(measure_1, measure_2):
    # # range of alpha
    dalp = (0.98 - 0.02) / 30
    alpha_grid = np.arange(0.02, 0.98 + dalp, dalp)  # len(alpha_grid) = 31

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.set_title(r"$Q_i^{(1)}$ and $Q_i^{(2)}$ versus α.")
    ax.set_xlabel(r"$\alpha$", fontsize=15)
    ax.set_ylabel("Measures", fontsize=15)

    colors = ["r", "b", "g"]
    params_name = ["alpha_{11}", "beta_{1}", "gamma_{0}"]

    for i, param in enumerate(params_name):
        ax.plot(
            alpha_grid,
            measure_1[:, i],
            colors[i] + "s--",
            label=r"$Q_{(\%s)}^1$" % param,
            markerfacecolor="none",
        )
        ax.plot(
            alpha_grid,
            measure_2[:, i],
            colors[i] + "o--",
            label=r"$Q_{(\%s)}^2$" % param,
            markerfacecolor="none",
        )

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    fig.legend(handles, labels, fontsize=13, loc="center right")

    abs_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(abs_dir, "../../figures/QBSM.png"), bbox_inches="tight")

    return fig, ax


def compute_QBSM(
    func,
    n_samples,
    seed,
    M,
    sampling_method,
    MC_method,
):
    # # range of alpha
    dalp = (0.98 - 0.02) / 30
    alpha_grid = np.arange(0.02, 0.98 + dalp, dalp)  # len(alpha_grid) = 31

    input_x_respy, input_x_mix_respy = create_sample(
        n_samples,
        seed,
        M,
        sampling_method,
        MC_method,
    )

    quantile_y_x = unconditional_quantile_y(input_x_respy, alpha_grid, func)
    quantile_y_x_mix = conditional_quantile_y(M, input_x_mix_respy, func, alpha_grid)

    q_1, q_2 = quantile_measures(quantile_y_x, quantile_y_x_mix)
    norm_q_1, norm_q_2 = normalized_quantile_measures(q_1, q_2)

    return norm_q_1, norm_q_2


if __name__ == "__main__":

    measure_1, measure_2 = compute_QBSM(
        func=quantitiy_of_interest,
        n_samples=30,
        seed=123,
        M=30,
        sampling_method="random",
        MC_method="Brute force",
    )

    QBSM_plot(measure_1, measure_2)
