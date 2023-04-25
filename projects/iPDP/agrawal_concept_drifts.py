import os
import random
import sys

import pandas as pd
import river.metrics
from matplotlib import pyplot as plt
from river.utils import Rolling
from river.ensemble import BaggingClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier

from river.datasets.synth import Agrawal

from ixai import IncrementalPFI
from ixai.explainer.pdp import BatchPDP, IncrementalPDP
from ixai.storage.ordered_reservoir_storage import OrderedReservoirStorage

if __name__ == "__main__":

    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (7, 7),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    plt.rcParams.update(params)

    # Get Data -------------------------------------------------------------------------------------
    N_SAMPLES = 10_000

    RANDOM_SEED: int = 42

    stream_1 = Agrawal(classification_function=2, seed=RANDOM_SEED)
    stream_2 = Agrawal(classification_function=1, seed=RANDOM_SEED)
    feature_names = list([x_0 for x_0, _ in stream_1.take(1)][0].keys())

    loss_metric = river.metrics.Accuracy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)

    model = BaggingClassifier(
        model=HoeffdingAdaptiveTreeClassifier(RANDOM_SEED),
        n_models=5,
        seed=RANDOM_SEED
    )
    #model = HoeffdingAdaptiveTreeClassifier(RANDOM_SEED)
    # Get imputer and explainers -------------------------------------------------------------------
    model_function = model.predict_proba_one

    grid_size = 20

    storage = OrderedReservoirStorage(
        store_targets=False,
        size=100,
        constant_probability=1
    )

    incremental_explainer = IncrementalPDP(
        model_function=model_function,
        feature_names=feature_names,
        grid_size=grid_size,
        gridsize=grid_size,
        pdp_feature='salary',
        output_key=1,
        smoothing_alpha=0.001,
        storage=storage,
        dynamic_setting=True,
        storage_size=100,
        min_max_grid=True,
        pdp_history_size=30,
        pdp_history_interval=1000
    )

    incremental_pfi = IncrementalPFI(
        model_function=model.predict_one,
        feature_names=feature_names,
        loss_function=river.metrics.Accuracy(),
        smoothing_alpha=0.001,
        storage=storage,
        dynamic_setting=True,
        n_inner_samples=1
    )

    # data storages --------------------------------------------------------------------------------

    model_performance = []
    feature_importance = []

    # 1st stream -----------------------------------------------------------------------------------

    batch_explainer = BatchPDP(
        pdp_feature='salary',
        gridsize=grid_size,
        model_function=model_function,
        output_key=1
    )

    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({'loss': training_metric.get()})
        # explain
        incremental_explainer.explain_one(x_i=x_i)
        incremental_pfi.explain_one(x_i=x_i, y_i=y_i)
        feature_importance.append(incremental_pfi.importance_values)
        _ = batch_explainer.update_storage(x_i)
        # train
        model.learn_one(x_i, y_i)
        if n >= N_SAMPLES:
            batch_explainer.explain_one(x_i=x_i)
            break
        if n % 100 == 0:
            print(f"n: {n}, training accuracy: {training_metric.get():.3f}")

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_legend=True
    )
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_ipdp_stream_1.pdf"))
    plt.show()

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_ice_curves=False,
        show_legend=True
    )
    plt.savefig("sw_ipdp_stream_1_no_ice.pdf")
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_1$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_1.pdf"))
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_1$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        show_ice_curves=False,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_1_no_ice.pdf"))
    plt.show()

    # 2nd stream -----------------------------------------------------------------------------------

    batch_explainer = BatchPDP(
        pdp_feature='salary',
        gridsize=grid_size,
        model_function=model_function,
        output_key=1
    )

    for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({'loss': training_metric.get()})
        # explain
        incremental_explainer.explain_one(x_i=x_i)
        incremental_pfi.explain_one(x_i=x_i, y_i=y_i)
        feature_importance.append(incremental_pfi.importance_values)
        _ = batch_explainer.update_storage(x_i)
        # train
        model.learn_one(x_i, y_i)
        if n >= N_SAMPLES:
            batch_explainer.explain_one(x_i=x_i)
            break
        if n % 100 == 0:
            print(f"n: {n}, training accuracy: {training_metric.get():.3f}")

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n * 2} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_legend=False
    )
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_ipdp_stream_2.pdf"))
    plt.show()

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n * 2} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_ice_curves=False,
        show_legend=False
    )
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_ipdp_stream_2_no_ice.pdf"))
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_2$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_2.pdf"))
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_2$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        show_ice_curves=False,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_2_no_ice.pdf"))
    plt.show()

    # 3rd stream -----------------------------------------------------------------------------------

    batch_explainer = BatchPDP(
        pdp_feature='salary',
        gridsize=grid_size,
        model_function=model_function,
        output_key=1
    )

    for (n, (x_i, y_i)) in enumerate(stream_2, start=1):
        x_i["salary"] = int(random.uniform(1.0, 1.3334) * x_i["salary"])
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model_performance.append({'loss': training_metric.get()})
        # explain
        incremental_explainer.explain_one(x_i=x_i)
        incremental_pfi.explain_one(x_i=x_i, y_i=y_i)
        feature_importance.append(incremental_pfi.importance_values)
        _ = batch_explainer.update_storage(x_i)
        # train
        model.learn_one(x_i, y_i)
        if n >= N_SAMPLES:
            batch_explainer.explain_one(x_i=x_i)
            break
        if n % 100 == 0:
            print(f"n: {n}, training accuracy: {training_metric.get():.3f}")

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n * 3} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_legend=False
    )
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_ipdp_stream_3.pdf"))
    plt.show()

    fig, axis = incremental_explainer.plot_pdp(
        title=f"iPDP at {n * 3} samples",
        y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6),
        show_ice_curves=False,
        show_legend=False
    )
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_ipdp_stream_3_no_ice.pdf"))
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_3$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_3.pdf"))
    plt.show()

    fig, axis = batch_explainer.plot_pdp(
        title=r"PDP for $t_3$",
        n_ice_curves_prop=0.01, y_min=0, y_max=1, x_min=0, x_max=200_000,
        xticks=[0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000, 160_000, 180_000,
                200_000],
        xticklabels=["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"],
        y_label="Probability for Class 1",
        return_plot=True,
        show_ice_curves=False,
        figsize=(6, 6)
    )
    fig.patch.set_facecolor('#EBEBEB')
    plt.savefig(os.path.join("agrawal_concept_drift", "sw_pdp_stream_3_no_ice.pdf"))
    plt.show()

    feature_of_interest = ["age", "salary", "elevel", "commission"]
    color_list = ['#44cfcb', '#44001A', '#4ea5d9', '#ef27a6', '#7d53de']

    pfi_values = pd.DataFrame(feature_importance)
    performances = pd.DataFrame(model_performance)

    fig, (axis_2, axis_1) = plt.subplots(
        2, 1, figsize=(10, 4), sharex="all", gridspec_kw={'height_ratios': [2, 4]})
    c_counter = 0
    for i, col_name in enumerate(list(pfi_values.columns)):
        if col_name in feature_of_interest:
            label = col_name
            #if col_name == "commission":
            #   label = "com."
            axis_1.plot(pfi_values[col_name], label=label, c=color_list[c_counter], linewidth=2)
            c_counter += 1
        else:
            axis_1.plot(pfi_values[col_name], c='#a6a7a9', linewidth=2)
    axis_1.plot([], [], label="others", c='#a6a7a9', linewidth=2)
    axis_1.set_ylabel("iPFI")
    axis_1.set_ylim((0, 0.8))
    axis_1.set_xlabel("Samples")
    axis_1.legend(ncols=5)
    for i in range(1, 2 + 1):
        axis_1.axvline(x=N_SAMPLES * i, linewidth=2, c="gray", ls="dotted")

    axis_2.plot(performances["loss"], c="red", linewidth=2)
    axis_2.set_ylabel("rolling\naccuracy")
    axis_2.set_ylim((0, 1.0))
    for i in range(1, 2 + 1):
        axis_2.axvline(x=N_SAMPLES * i, linewidth=2, c="gray", ls="dotted")
    axis_2.plot([], [], label="concept drift time", linewidth=2, c="gray", ls="dotted")
    axis_2.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("agrawal_concept_drift", "agrawal_pfi_and_performance.pdf"))
    plt.show()

    # store data
    pfi_values.to_csv(os.path.join("agrawal_concept_drift", "pfi.csv"), index=False)
    performances.to_csv(os.path.join("agrawal_concept_drift", "loss.csv"), index=False)