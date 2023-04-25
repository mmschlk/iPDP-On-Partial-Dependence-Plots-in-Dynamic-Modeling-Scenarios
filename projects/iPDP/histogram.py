"""This module explores how an online Histogram can be used in a window manner."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from river.sketch import Histogram
from river.stream import iter_pandas
from river.cluster import KMeans


if __name__ == "__main__":

    STREAM_LEN = 10000
    RANDOM_SEED = 1
    N_SAMPLES = 4000

    # STREAM1 variables
    VAR1_MEAN = 100
    VAR1_STD = 15

    #STREAM2 Variables
    VAR2_MEAN = 300
    VAR2_STD = 15

    var1 = np.random.normal(VAR1_MEAN, VAR1_STD, size=(STREAM_LEN,))
    var2 = np.random.normal(VAR2_MEAN, VAR2_STD, size=(STREAM_LEN,))

    stream1_df = pd.DataFrame(pd.Series(var1, name='col1'))
    stream2_df = pd.DataFrame(pd.Series(var2, name='col1'))

    hist_1 = Histogram(max_bins=50)
    hist_2 = Histogram(max_bins=50)
    hist_both = Histogram(max_bins=50)
    hist_both_del = Histogram(max_bins=50)

    clusterer = KMeans(n_clusters=50)

    # Concept 1 ------------------------------------------------------------------------------------

    for (n, (x_i, y_i)) in enumerate(iter_pandas(stream1_df), start=1):
        feature_value = x_i['col1']
        hist_both = hist_both.update(feature_value)
        hist_both_del = hist_both_del.update(feature_value)
        hist_1 = hist_1.update(feature_value)
        clusterer.learn_one(x_i)

    # Concept 2 ------------------------------------------------------------------------------------

    plt.bar(
        x=[(b.left + b.right) / 2 for b in hist_1],
        height=[b.count for b in hist_1],
        width=[(b.right - b.left) / 2 for b in hist_1]
    )
    plt.show()

    for (n, (x_i, y_i)) in enumerate(iter_pandas(stream2_df), start=1):
        feature_value = x_i['col1']
        hist_both = hist_both.update(feature_value)
        hist_both_del = hist_both_del.update(feature_value)
        hist_2 = hist_2.update(feature_value)
        clusterer.learn_one(x_i)

        if n == 2000:
            new_data = []
            for hist_bin in hist_both_del.data:
                if hist_bin.left >= 230:
                    new_data.append(hist_bin)
            hist_both_del.data = new_data


    plt.bar(
        x=[(b.left + b.right) / 2 for b in hist_2],
        height=[b.count for b in hist_2],
        width=[(b.right - b.left) / 2 for b in hist_2]
    )
    plt.show()

    plt.bar(
        x=[(b.left + b.right) / 2 for b in hist_both],
        height=[b.count for b in hist_both],
        width=[(b.right - b.left) / 2 for b in hist_both]
    )
    plt.show()

    plt.bar(
        x=[(b.left + b.right) / 2 for b in hist_both_del],
        height=[b.count for b in hist_both_del],
        width=[(b.right - b.left) / 2 for b in hist_both_del]
    )
    plt.show()
