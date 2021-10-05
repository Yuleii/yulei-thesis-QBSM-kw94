import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import os
import sys


sys.path.append("..")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"


def heatmeap_corr():

    corr_df = pd.read_pickle(DATA_PATH / "est_corr_chol.pkl")

    select = corr_df.iloc[[0, 1, 7, 13, 16, 19, 20], [0, 1, 7, 13, 16, 19, 20]]
    mask = np.zeros_like(select)
    mask[np.triu_indices_from(select, 1)] = True

    labels = np.array(
        [
            r"$\hat{\delta}$",
            r"$\hat{\alpha_{10}}$",
            r"$\hat{\alpha_{11}}}$",
            r"$\hat{\alpha_{12}}$",
            r"$\hat{\alpha_{13}}$",
            r"$\hat{\alpha_{14}}$",
            r"$\hat{\alpha_{15}}$",
            r"$\hat{\alpha_{20}}$",
            r"$\hat{\alpha_{21}}$",
            r"$\hat{\alpha_{22}}$",
            r"$\hat{\alpha_{23}}$",
            r"$\hat{\alpha_{24}}$",
            r"$\hat{\alpha_{25}}$",
            r"$\hat{\beta_0}$",
            r"$\hat{\beta_1}$",
            r"$\hat{\beta_2}$",
            r"$\hat{\gamma_0}}$",
            r"$\hat{c_{1}}$",
            r"$\hat{c_{2}}$",
            r"$\hat{c_{3}}$",
            r"$\hat{c_{4}}$",
            r"$\hat{c_{1,2}}$",
            r"$\hat{c_{1,3}}$",
            r"$\hat{c_{2,3}}$",
            r"$\hat{c_{1,4}}$",
            r"$\hat{c_{2,4}}$",
            r"$\hat{c_{3,4}$}",
        ]
    )
    # custom boundaries
    boundaries = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    # here I generated twice as many colors,
    # so that I could prune the boundaries more clearly
    hex_colors = sns.light_palette(
        "navy", n_colors=len(boundaries) * 2 + 2, as_cmap=False
    ).as_hex()
    hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]

    colors = list(zip(boundaries, hex_colors))

    custom_color_map = LinearSegmentedColormap.from_list(
        name="custom_navy",
        colors=colors,
    )
    fig = plt.figure(figsize=(15, 10))

    # Aspects for heigth, pad for whitespace
    ax = sns.heatmap(
        data=select,
        mask=mask,
        cmap=custom_color_map,
        linewidths=0.0,
        square=False,
        vmin=-1,
        vmax=1,
        annot=True,
    )

    ax.tick_params(axis="both", direction="out", length=6, width=2)
    ax.set_yticklabels(labels[[0, 1, 7, 13, 16, 19, 20]], ha="left", rotation=0)
    ax.set_xticklabels(labels[[0, 1, 7, 13, 16, 19, 20]], rotation=0)
    ax.set_ylabel(r"$\hat{\theta}$", labelpad=+35, rotation=0)
    ax.set_xlabel(r"$\hat{\theta}$", labelpad=+25)

    cbar = ax.collections[0].colorbar

    # Positioning at -1 needs vmin.
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["-1.0", " -0.5", "0.0", "0.5", "1.0"])
    cbar.ax.tick_params(direction="out", length=6, width=2)

    # A bit more space for xlabels.
    ax.tick_params(axis="x", which="major", pad=8)

    # Left-align y labels.
    yax = ax.get_yaxis()
    pad = max(tick.label.get_window_extent().width for tick in yax.majorTicks) + 5
    yax.set_tick_params(pad=pad)

    abs_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(abs_dir, "../../figures/heatmap.png"), bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    heatmeap_corr()
