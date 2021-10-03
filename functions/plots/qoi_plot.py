import numpy as np
import pandas as pd
import respy as rp
import matplotlib.pyplot as plt
import os


def qoi_kde(n_bs_samples, bs_sample_size, subsidy):
    diff_edu = _bs_sampling(n_bs_samples, bs_sample_size, subsidy)
    mean_diff_edu = diff_edu.agg("mean")["Experience_Edu"]

    # exact_effect_edu = 1.44 # The exact solution reported in kw97,table 9

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = diff_edu["Experience_Edu"].plot(kind="density", label="KDE", color="#1D2F6F")
    ax.axvline(
        mean_diff_edu,
        color="#8390FA",
        linestyle="--",
        label="Point estimate of policy effect",
    )
    ax.set_xlabel("Change in mean years of schooling", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", framealpha=1, frameon=True)

    abs_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(abs_dir, "../../figures/qoi.png"), bbox_inches="tight")

    return fig, ax


def _bs_sampling(n_bs_samples, bs_sample_size, subsidy):
    # load parameters and options corresponding to model
    params, options = rp.get_example_model("kw_94_one", with_data=False)

    # Here we need 40 samples of 100 people. To process this,
    # we genetate 4000 individuals and then split them to 40
    # equally-sized samples with 100 individuals in each sample.
    options["simulation_agents"] = n_bs_samples * bs_sample_size

    # get simulate function
    simulate = rp.get_simulate_func(params, options)

    # get simulated results without tuition subsidy
    df_no_ts = simulate(params)

    # get simulated results with $500 tuition subsidy
    ts_params = params.copy()
    ts_params.loc[("nonpec_edu", "at_least_twelve_exp_edu"), "value"] += subsidy
    df_ts = simulate(ts_params)

    # assign bootstrap sample number
    for df in [df_no_ts, df_ts]:
        df["sample_num"] = pd.cut(
            df.index.get_level_values(0),
            bins=n_bs_samples,
            labels=np.arange(1, n_bs_samples + 1),
        )

    # Calculate mean end-of-life schooling
    columns = ["sample_num", "Experience_Edu"]
    mean_edu_no_ts = (
        df_no_ts[(df_no_ts.index.get_level_values(1) == 39)][columns]
        .groupby("sample_num")
        .mean()
    )
    mean_edu_ts = (
        df_ts[(df_ts.index.get_level_values(1) == 39)][columns]
        .groupby("sample_num")
        .mean()
    )
    # Calculate the difference of mean final schooling
    # with and without tuition subsidy.
    diff_edu = (
        mean_edu_ts.subtract(mean_edu_no_ts).reset_index().set_index("sample_num")
    )

    return diff_edu
