"""This script is used to generate the rotating hyperplane example."""

import copy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from river.tree import HoeffdingAdaptiveTreeClassifier
from river import metrics
from river.utils import Rolling
from ixai.storage import GeometricReservoirStorage

from ixai.explainer.pdp import IncrementalPDP
from ixai.explainer.pfi import IncrementalPFI


if __name__ == "__main__":

    params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (6, 6),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    plt.rcParams.update(params)

    STREAM_LEN = 30_000
    RANDOM_SEED = 1
    N_SAMPLES = 20_000

    ### Creating streams

    ERROR_MEAN = 5
    ERROR_STD = ERROR_MEAN * 0.2

    # STREAM1 variables
    VAR1_MEAN = 100
    VAR1_STD = VAR1_MEAN * 0.2
    VAR2_MEAN = 200
    VAR2_STD = VAR2_MEAN * 0.2

    #STREAM2 Variables
    VAR4_MEAN = 200
    VAR4_STD = VAR4_MEAN * 0.2
    VAR5_MEAN = 100
    VAR5_STD = VAR5_MEAN * 0.2

    var1 = np.random.normal(VAR1_MEAN, VAR1_STD, size=(STREAM_LEN,))
    var2 = np.random.normal(VAR2_MEAN, VAR2_STD, size=(STREAM_LEN,))
    error = np.random.normal(ERROR_MEAN, ERROR_STD, size=(STREAM_LEN,))

    coeff_var1 = 1
    coeff_var2 = -0.5
    thresh = 0.1
    var3 = coeff_var1 * var1 + coeff_var2 * var2 + error
    y = 1 / (1 + np.exp(-var3))
    y[y>thresh] = 1
    y[y<=thresh] = 0

    var4 = np.random.normal(VAR4_MEAN, VAR4_STD, size=(STREAM_LEN,))
    var5 = np.random.normal(VAR5_MEAN, VAR5_STD, size=(STREAM_LEN,))
    coeff_var4 = -0.5
    coeff_var5 = 1
    thresh = 0.1
    var6 = coeff_var4 * var4 + coeff_var5 * var5 + error
    y1 = 1 / (1 + np.exp(-var6))
    y1[y1>thresh] = 1
    y1[y1<=thresh] = 0

    # Setup Data -----------------------------------------------------------------------------------

    stream1_df = pd.DataFrame(pd.Series(var1, name=r"$X^1$"))
    stream1_df[r"$X^2$"] = var2
    stream_1 = stream1_df.to_dict('records')
    stream_1 = zip(stream_1, y)

    stream2_df = pd.DataFrame(pd.Series(var4, name=r"$X^1$"))
    stream2_df[r"$X^2$"] = var5
    stream_2 = stream2_df.to_dict('records')
    stream_2 = zip(stream_2, y1)

    feature_names = [r"$X^1$", r"$X^2$"]

    # Setup explainers -----------------------------------------------------------------------------

    # Model and training setup
    #model = LogisticRegression()
    model = HoeffdingAdaptiveTreeClassifier(seed=42)
    loss_metric = metrics.Accuracy()
    training_metric = Rolling(metrics.Accuracy(), window_size=500)

    # Instantiating objects
    storage_pdp = GeometricReservoirStorage(
        store_targets=False,
        size=1000,
        constant_probability=1
    )

    incremental_pdp = IncrementalPDP(
        model_function=model.predict_proba_one,
        feature_names=feature_names,
        gridsize=20,
        dynamic_setting=True,
        smoothing_alpha=0.01,
        pdp_feature=r"$X^1$",
        storage=storage_pdp,
        storage_size=100,
        pdp_history_size=50,
        pdp_history_interval=4000
    )

    storage_pfi = GeometricReservoirStorage(
        store_targets=False,
        size=1000,
        constant_probability=1
    )

    inc_pfi = IncrementalPFI(
        model_function=model.predict_one,
        feature_names=feature_names,
        smoothing_alpha=0.001,
        loss_function=metrics.Accuracy(),
        n_inner_samples=3
    )

    # warm-up because of errors --------------------------------------------------------------------
    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        model.learn_one(x_i, y_i)
        x_explain = copy.deepcopy(x_i)
        if n > 10:
            break

    # Concept 1 ------------------------------------------------------------------------------------

    model_performance = []
    pfi_values = []

    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({"loss": training_metric.get()})

        # explaining
        incremental_pdp.explain_one(x_i)
        inc_pfi.explain_one(x_i, y_i)
        pfi_values.append(inc_pfi.importance_values)

        # training
        model.learn_one(x_i, y_i)

        if n % 1000 == 0:
            print(n, training_metric.get())

        if n % N_SAMPLES == 0:
            fig, axes = incremental_pdp.plot_pdp(
                title=f"iPDP at {n} samples",
                y_label="Probability for Class 1",
                y_min=0, y_max=1,
                x_min=50, x_max=250,
                show_ice_curves=False,
                return_plot=True,
                figsize=(7, 5)
            )
            plt.savefig(os.path.join("hyperplane", "hyperplane_1.pdf"))
            plt.show()
            break

    # Concept 2 ------------------------------------------------------------------------------------

    for (n, (x_i, y_i)) in enumerate(stream_2, start=N_SAMPLES):
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({"loss": training_metric.get()})

        # explaining
        incremental_pdp.explain_one(x_i)
        inc_pfi.explain_one(x_i, y_i)
        pfi_values.append(inc_pfi.importance_values)

        # training
        model.learn_one(x_i, y_i)

        if n % 1000 == 0:
            print(n, training_metric.get())

        # plot interim iPDP
        if n % int(N_SAMPLES + 500) == 0:
            fig, axes = incremental_pdp.plot_pdp(
                title=f"iPDP at {n} samples",
                y_label="Probability for Class 1",
                y_min=0, y_max=1,
                x_min=50, x_max=250,
                show_ice_curves=False,
                return_plot=True,
                figsize=(7, 5)
            )
            plt.savefig(os.path.join("hyperplane", "hyperplane_2.pdf"))
            plt.show()

        if n % int(N_SAMPLES * 2) == 0:
            fig, axes = incremental_pdp.plot_pdp(
                title=f"iPDP at {n} samples",
                y_label="Probability for Class 1",
                y_min=0, y_max=1,
                x_min=50, x_max=250,
                show_ice_curves=False,
                return_plot=True,
                figsize=(7, 5)
            )
            plt.savefig(os.path.join("hyperplane", "hyperplane_3.pdf"))
            plt.show()
            break

    color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#44001A']

    pfi_values = pd.DataFrame(pfi_values)
    performances = pd.DataFrame(model_performance)

    fig, (axis_2, axis_1) = plt.subplots(
        2, 1, figsize=(10, 4), sharex="all", gridspec_kw={'height_ratios': [2, 4]})
    for i, col_name in enumerate(list(pfi_values.columns)):
        axis_1.plot(pfi_values[col_name], label=" ".join(("iPFI for feature", col_name)), c=color_list[i], linewidth=2)
    axis_1.set_ylabel("iPFI")
    axis_1.set_ylim((0, 0.4))
    axis_1.set_xlabel("Samples")
    axis_1.legend()
    axis_1.axvline(x=N_SAMPLES, linewidth=2, c="gray", ls="dotted")

    axis_2.plot(performances["loss"], c="red", linewidth=2)
    axis_2.set_ylabel("rolling\naccuracy")
    axis_2.set_ylim((0, 1))
    axis_2.axvline(x=N_SAMPLES, linewidth=2, c="gray", ls="dotted")
    axis_2.plot([], [], label="concept drift time", linewidth=2, c="gray", ls="dotted")
    axis_2.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("hyperplane", "pfi_and_performance.pdf"))
    plt.show()

    # store data
    pfi_values.to_csv(os.path.join("hyperplane", "pfi.csv"), index=False)
    performances.to_csv(os.path.join("hyperplane", "loss.csv"), index=False)
