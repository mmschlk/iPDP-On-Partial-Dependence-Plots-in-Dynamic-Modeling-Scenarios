import pandas as pd

from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

#  COLORS = {'SII': '#058ED9', 'STI': '#2D3142', 'SFI': '#CC2D35'}  # online inclusive colors
COLORS = {'SII': '#44cfcb', 'STI': '#7d53de', 'SFI': '#ef27a6'}
BACKGROUND_COLOR = '#f8f8f8'
MARKERS = {'SII': 'X', 'STI': 'o', 'SFI': "s"}
LABELS = {'SII': 'Shapley Interaction', 'STI': 'Shapley Taylor', 'SFI': "Faith-SHAP", 'U-KSH': "Unbiased Kernel Shap", "U-KSH-R": "Unbiased Kernel Shap (replacement)"}

STD_ALPHA = 0.10


def draw_approx_curve(df: pd.DataFrame, figsize: tuple = (10, 10), error_type: str = "approx_value",
                      mean_aggregation: bool = False, shading: bool = True,
                      x_min: int = None, y_max: float = None, y_min: float = None, plot_title: str = None,
                      x_label: str = None, y_label: str = None, save_name: str = None, max_computation_cost_n: int = None,
                      horizontal_line_y: float = None):

    grouping = ['n_absolute']

    data = df.copy()
    data = data[data['sampling'].isin(['const and sampling', np.NAN])].drop(columns=['sampling'])
    data = data[data['n_absolute'] > 0]

    data = data.rename(columns={"interaction_index": "Interaction Index", "approx_type": "Method"})

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    for interaction_index in data["Interaction Index"].unique():
        subset = data[data["Interaction Index"] == interaction_index]
        if x_min is not None:
            subset = subset[subset["n_absolute"] >= x_min]

        if 'inner_iteration' in subset.columns:
            subset = subset.groupby(by=['iteration', 'n_absolute', "Method"]).aggregate({error_type: "mean"}).reset_index()

        baseline = subset[subset["Method"] == "baseline"]
        approximation = subset[subset["Method"] == "approximation"]

        if not mean_aggregation == "median":
            baseline_mean = baseline.groupby(by=grouping)[error_type].quantile(q=0.5).reset_index()
            baseline_quantile_l = baseline.groupby(by=grouping)[error_type].quantile(q=0.25).reset_index()[error_type]
            baseline_quantile_h = baseline.groupby(by=grouping)[error_type].quantile(q=0.75).reset_index()[error_type]
            approximation_mean = approximation.groupby(by=grouping)[error_type].quantile(q=0.5).reset_index()
            approximation_quantile_l = approximation.groupby(by=grouping)[error_type].quantile(q=0.25).reset_index()[error_type]
            approximation_quantile_h = approximation.groupby(by=grouping)[error_type].quantile(q=0.75).reset_index()[error_type]
        else:
            baseline_mean = baseline.groupby(by=grouping)[error_type].mean().reset_index()
            approximation_mean = approximation.groupby(by=grouping)[error_type].mean().reset_index()
            approximation_std = approximation.groupby(by=grouping)[error_type].std().reset_index()
            approximation_quantile_l = approximation_mean[error_type] - approximation_std[error_type]
            approximation_quantile_h = approximation_mean[error_type] + approximation_std[error_type]
            baseline_std = baseline.groupby(by=grouping)[error_type].std().reset_index()
            baseline_quantile_l = baseline_mean[error_type] - baseline_std[error_type]
            baseline_quantile_h = baseline_mean[error_type] + baseline_std[error_type]

        axis.plot(baseline_mean["n_absolute"], baseline_mean[error_type],
                  ls="dashed", color=COLORS[interaction_index], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")
        axis.plot(approximation_mean["n_absolute"], approximation_mean[error_type],
                  ls="solid", color=COLORS[interaction_index], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")

        if shading:
            try:
                axis.fill_between(approximation_mean["n_absolute"],
                                  approximation_quantile_l,
                                  approximation_quantile_h,
                                  color=COLORS[interaction_index], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in baseline std:", e)
            try:
                axis.fill_between(approximation_mean["n_absolute"],
                                  baseline_quantile_l,
                                  baseline_quantile_h,
                                  color=COLORS[interaction_index], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in approximation std:", e)

    title_1 = "$\\bf{Indices}$"
    axis.plot([], [], label=title_1, color="none")
    for interaction_index in data["Interaction Index"].unique():
        axis.plot([], [], color=COLORS[interaction_index],
                  marker=MARKERS[interaction_index], label=LABELS[interaction_index], mec="white")

    title_2 = "$\\bf{Methods}$"
    axis.plot([], [], label=title_2, color="none")
    axis.plot([], [], color="black", ls="solid", label="SHAPX")
    axis.plot([], [], color="black", ls="dashed", label="baseline")

    leg = axis.legend()
    for item, label in zip(leg.legendHandles, leg.texts):
        if label._text in [title_1, title_2]:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    plt.title(plot_title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim((y_min, y_max))
    axis.set_facecolor(BACKGROUND_COLOR)

    if horizontal_line_y is not None:
        axis.axhline(y=horizontal_line_y, ls="dotted", c="gray")

    if max_computation_cost_n is not None:
        max_computation_cost = 2 ** max_computation_cost_n
        labels = [item.get_text() for item in axis.get_xticklabels()]
        for i in range(len(labels)):
            value = int(labels[i])
            labels[i] = labels[i] + "\n" + str(round(value / max_computation_cost, 2))
        axis.set_xticklabels(labels)

    if error_type == "approx_value":
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    else:
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
