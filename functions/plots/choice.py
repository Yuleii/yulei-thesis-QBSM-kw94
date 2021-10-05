import matplotlib.pyplot as plt
import respy as rp
import numpy as np
import os


def choiceovertime():
    # Build simulate function as only parameters change, it can be reused.
    params, options, _ = rp.get_example_model("kw_94_one")
    options["simulation_agents"] = 4_000
    simulate = rp.get_simulate_func(params, options)
    models = np.repeat(["one"], 2)
    tuition_subsidies = [0, 500]

    data_frames = []

    for model, tuition_subsidy in zip(models, tuition_subsidies):
        params, _, _ = rp.get_example_model(f"kw_94_{model}")
        params.loc[
            ("nonpec_edu", "at_least_twelve_exp_edu"), "value"
        ] += tuition_subsidy
        data_frames.append(simulate(params))

    colors = [
        "#FAC748",
        "#6EAF46",
        "#1D2F6F",
        "#8390FA",
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

    axs = axs.flatten()

    for df, ax, model, tuition_subsidy in zip(
        data_frames, axs, models, tuition_subsidies
    ):
        shares = (
            df.groupby("Period")
            .Choice.value_counts(normalize=True)
            .unstack()[["home", "edu", "a", "b"]]
        )

        shares.plot.bar(stacked=True, ax=ax, width=1, legend=True, color=colors)

        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, 40, 5))
        ax.set_xticklabels(range(0, 40, 5), rotation="horizontal")
        ax.set_ylabel("Share of population")
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        if tuition_subsidy:
            label = f"with a tuition subsidy of {tuition_subsidy:,} USD"
        else:
            label = "without a tuition subsidy"
        ax.set_title(f"Parameterization {model.title()} \n {label}")

    fig.legend(
        handles,
        ["Home", "Schooling", "Blue collar", "White collar"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
    )

    abs_dir = os.path.dirname(__file__)
    plt.savefig(
        os.path.join(abs_dir, "../../figures/choiceovertime.png"), bbox_inches="tight"
    )

    return fig, ax


if __name__ == "__main__":
    choiceovertime()
