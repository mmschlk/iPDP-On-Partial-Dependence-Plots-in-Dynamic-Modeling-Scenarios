"""This script contains the code to run the experiments for the Elec2 dataset."""

import os

import pandas as pd
import river.metrics
from matplotlib import pyplot as plt
from river.ensemble import BaggingClassifier
from river.utils import Rolling
from river.tree import HoeffdingAdaptiveTreeClassifier

from river.datasets import Elec2
from river.drift import ADWIN

from ixai.explainer.pdp import IncrementalPDP
from ixai.storage.ordered_reservoir_storage import OrderedReservoirStorage


from projects.iPDP.change_detector import PDPChangeDetector

if __name__ == "__main__":

    params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (7, 7),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    plt.rcParams.update(params)

    # Get Data -------------------------------------------------------------------------------------
    RANDOM_SEED: int = 42

    stream_1 = Elec2()
    feature_names = list([x_0 for x_0, _ in stream_1.take(1)][0].keys())
    feature_of_interest = "vicprice"

    loss_metric = river.metrics.Accuracy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)
    accuracy = river.metrics.Accuracy()

    model = BaggingClassifier(
        model=HoeffdingAdaptiveTreeClassifier(RANDOM_SEED),
        n_models=10,
        seed=RANDOM_SEED
    )
    #model = HoeffdingAdaptiveTreeClassifier(RANDOM_SEED)

    # Get imputer and explainers -------------------------------------------------------------------
    model_function = model.predict_proba_one
    grid_size = 10

    storage = OrderedReservoirStorage(
        store_targets=False,
        size=500,
        constant_probability=0.05
    )
    inc_explainer = IncrementalPDP(
        model_function=model_function,
        feature_names=feature_names,
        gridsize=grid_size,
        dynamic_setting=True,
        smoothing_alpha=0.001,
        pdp_feature=feature_of_interest,
        storage=storage,
        storage_size=10,
        output_key=1,
        pdp_history_interval=2000,
        pdp_history_size=25
    )

    change_detector = PDPChangeDetector(
        grid_size=grid_size
    )

    perf_detector = ADWIN(delta=0.00001, grace_period=2_000)

    # 1st stream -----------------------------------------------------------------------------------
    print("elec2")
    performance, changes_performance, changes_pdp = [], [], []
    for (n, (x_i, y_i)) in enumerate(stream_1, start=1):
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        performance.append({"loss": training_metric.get()})
        inc_explainer.explain_one(x_i)
        pdp_x = inc_explainer.pdp_x_tracker.get()
        pdp_y = inc_explainer.pdp_y_tracker.get()

        drift = change_detector.detect_one(pdp_x=pdp_x, pdp_y=pdp_y)
        if drift:
            fig, axes = inc_explainer.plot_pdp(
                title=f"elec2 after {n} samples", show_pdp_transition=True,
                show_ice_curves=False,
                y_min=0, y_max=1.0,
                return_plot=True,
                n_decimals=4
            )
            plt.savefig(os.path.join("change_detection_plots", f"{n}.pdf"))
            plt.show()
            changes_pdp.append(n)

        accuracy.update(y_true=y_i, y_pred=y_i_pred)
        accuracy_value = accuracy.get()
        accuracy.revert(y_true=y_i, y_pred=y_i_pred)
        perf_detector.update(accuracy_value)
        if perf_detector.drift_detected:
            changes_performance.append(n)
            print(f"Performance drift detected at point: {n}")

        if n % 100 == 0:
            print(n)

        model.learn_one(x_i, y_i)

    # save performances
    performance_df = pd.DataFrame(performance)
    performance_df.to_csv(os.path.join("change_detection_plots", "loss.csv"), index=False)

    # drift in performance
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    axis.plot(
        performance_df["loss"],
        ls="solid", c="red", alpha=1., linewidth=2
    )
    for change_point in changes_performance:
        axis.axvline(x=change_point, ls="dotted", c="grey", alpha=0.75, linewidth=2)
    if len(changes_performance) > 0:
        axis.plot([], label='Drift Points', ls="dotted", c="grey", alpha=0.75, linewidth=2)
        axis.legend(edgecolor="0.8", fancybox=False)
    axis.set_ylim((0, 1))
    axis.set_ylabel("Rolling Accuracy")
    axis.set_xlabel("Samples")
    plt.savefig(os.path.join("change_detection_plots", "loss.pdf"))
    plt.show()

    # drift in iPDP
    fig, axis = plt.subplots(1, 1, figsize=(10, 2))
    axis.plot(
        performance_df["loss"],
        ls="solid", c="red", alpha=1., linewidth=2
    )
    for change_point in changes_pdp:
        axis.axvline(x=change_point, ls="dashed", c="blue", alpha=0.75, linewidth=2)
    if len(changes_pdp) > 0:
        axis.plot([], label='iPDP drift point', ls="dashed", c="blue", alpha=0.75, linewidth=2)
        axis.legend(edgecolor="0.8", fancybox=False)
    axis.set_ylim((0, 1))
    axis.set_ylabel("rolling\naccuracy")
    axis.set_xlabel("Samples")
    #plt.title("model performance")
    plt.tight_layout()
    plt.savefig(os.path.join("changes_pdp", "iPDP_loss.pdf"))
    plt.show()
