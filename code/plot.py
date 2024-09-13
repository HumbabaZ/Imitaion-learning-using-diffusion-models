import os
import pickle
import numpy as np
from matplotlib import pyplot

SAVE_DATA_DIR = "output_insertion"
SAVE_FIGURE_DIR = "figures"
DPI = 300
IS_USE_LATEX = False

TITLE_KWARGS = {"fontsize": 7, "multialignment": "center", "verticalalignment": "center", "horizontalalignment": "center"}
YLABEL_TITLE_KWARGS = {"fontsize": "small", "verticalalignment": "center", "horizontalalignment": "right", "rotation": 0}
SAVEFIG_KWARGS = {"dpi": DPI, "bbox_inches": "tight", "pad_inches": 0.0}
SCATTER_KWARGS = {"s": 1, "alpha": 0.25, "color": "yellow"}

N_SAMPLES_PER_METHOD = 2
INTERESTING_SAMPLE_IDX = 6
EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]

if IS_USE_LATEX:
    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

def plot_diffusion_comparison():
    MY_DIFFUSION_STEPS = [4, 16, 32]
    fig, axs = pyplot.subplots(
        nrows=len(MY_DIFFUSION_STEPS) + 1,
        ncols=N_SAMPLES_PER_METHOD,
        dpi=DPI
    )
    ylabel_title_kwargs = YLABEL_TITLE_KWARGS.copy()
    ylabel_title_kwargs["fontsize"] = 7

    # Plot the base diffusion model
    file_path = os.path.join(SAVE_DATA_DIR, "diffusion.pkl")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    for sample_i in range(N_SAMPLES_PER_METHOD):
        axs[0, sample_i].imshow(data[sample_i]["x_eval_large"])
        axs[0, sample_i].scatter(
            data[sample_i]["y_pred"][:, 0] * 16,
            data[sample_i]["y_pred"][:, 1] * 16,
            **SCATTER_KWARGS
        )
        axs[0, sample_i].set_xticks([])
        axs[0, sample_i].set_yticks([])
        axs[0, sample_i].set_ylim(15, -1)
        axs[0, sample_i].set_xlim(15, -1)
        if sample_i == 0:
            axs[0, sample_i].set_ylabel("Diffusion", **ylabel_title_kwargs)

    # Plot the extra-diffusion steps
    for diffusion_step_i, diffusion_step in enumerate(MY_DIFFUSION_STEPS):
        file_path = os.path.join(SAVE_DATA_DIR, "diffusion_extra-diffusion_{}.pkl".format(diffusion_step))
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for sample_i in range(N_SAMPLES_PER_METHOD):
            axs[1 + diffusion_step_i, sample_i].imshow(data[sample_i]["x_eval_large"])
            axs[1 + diffusion_step_i, sample_i].scatter(
                data[sample_i]["y_pred"][:, 0] * 16,
                data[sample_i]["y_pred"][:, 1] * 16,
                **SCATTER_KWARGS
            )
            axs[1 + diffusion_step_i, sample_i].set_xticks([])
            axs[1 + diffusion_step_i, sample_i].set_yticks([])
            axs[1 + diffusion_step_i, sample_i].set_ylim(15, -1)
            axs[1 + diffusion_step_i, sample_i].set_xlim(15, -1)
            if sample_i == 0:
                axs[1 + diffusion_step_i, sample_i].set_ylabel(
                    "Diff.X. ($M={}$)".format(diffusion_step),
                    **ylabel_title_kwargs
                )

    # Save fig
    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=-0.85, hspace=0.1)
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, "diffusion_comparison.pdf"), **SAVEFIG_KWARGS)


if __name__ == "__main__":
    os.makedirs(SAVE_FIGURE_DIR, exist_ok=True)
    plot_diffusion_comparison()
