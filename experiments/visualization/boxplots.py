import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#from iXAI.visualization.color import DEFAULT_COLOR_LIST
from experiments.visualization.approximation_curve import BACKGROUND_COLOR

#DEFAULT_COLOR_LIST = ['#1d4289', "orange"]
DEFAULT_COLOR_LIST = ["#4ea5d9", "#7d53de"]

def grouped_boxplot(data, feature_names, explainer_names, save_path, legend, width=0.5, min_space=0.1, feature_spacing=0.5, y_min=None, y_max=None, legend_pos=0, title=None):
    n_explainer = len(explainer_names)
    n_features = len(feature_names)

    feature_names_short = [name[0:3]+'.' if len(name) > 3 else name[0:3] for name in feature_names]

    group_length = width * n_explainer + min_space * (n_explainer-1)
    x_range = np.array([x * group_length for x in range(n_features)])
    x_range = np.array([x_range[i] + feature_spacing * i for i in range(n_features)])
    x_locations = sorted(
        [feature_loc + explainer * (width + min_space) for explainer in range(n_explainer) for feature_loc in x_range])
    feature_x_locations = x_range + group_length / 2 - width / 2

    fig = plt.figure(0, figsize=[4.8, 4.8])
    ax = fig.add_subplot(111)
    x_position_index = 0
    for feature in feature_names:
        color_index = 0
        for explainer in explainer_names:
            if explainer == 'batch_total' or explainer == 'batch_interval':
                color = 'black'
            else:
                color = DEFAULT_COLOR_LIST[color_index]
                color_index += 1
            #ax.boxplot(
            #    data[explainer][feature], positions=[x_locations[x_position_index]],
            #    widths=width, patch_artist=True,
            #    medianprops={'color': 'red', 'linestyle': '-'},boxprops=dict(facecolor=color, color=None)
            #)
            violin_parts = ax.violinplot(
                data[explainer][feature], positions=[x_locations[x_position_index]],
                widths=width,
                showmeans=True, showmedians=True
            )
            for pc in violin_parts['bodies']:
                #pc.set_facecolor(color)
                #pc.set_edgecolor(color)
                pc.set_color(color)
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                vp = violin_parts[partname]
                vp.set_edgecolor(color)
                vp.set_linewidth(1)
            vp = violin_parts['cmedians']
            vp.set_edgecolor('red')
            vp.set_linewidth(1)

            x_position_index += 1
    x_position_index = 0
    color_index = 0
    for explainer in explainer_names:
        if explainer == 'batch_total' or explainer == 'batch_interval':
            color = 'black'
        else:
            color = DEFAULT_COLOR_LIST[color_index]
            color_index += 1
        legend_name = legend[x_position_index]
        ax.plot([], c=color, label=legend_name)
        x_position_index += 1

    ax.legend(edgecolor="0.8", fancybox=False, loc=legend_pos)
    ax.set_xticks(feature_x_locations, feature_names_short)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color=(0.5, 0.5, 0.5, 0.3), ls='--')
    ax.set_ylabel('SAGE Values')
    ax.set_xlabel('Features')
    ax.set_facecolor(BACKGROUND_COLOR)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    pass
