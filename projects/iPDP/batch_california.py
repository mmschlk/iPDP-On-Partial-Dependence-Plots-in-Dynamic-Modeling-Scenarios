import copy

from matplotlib import pyplot as plt
from river.stream import iter_pandas
from sklearn.model_selection import train_test_split

from ixai.explainer.pdp import BatchPDP, IncrementalPDP

import os
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torchmetrics.functional import r2_score, mean_absolute_error

from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from ixai.storage.ordered_reservoir_storage import OrderedReservoirStorage
from ixai.utils.wrappers import TorchWrapper


if __name__ == "__main__":

    params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (7, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'
    }
    plt.rcParams.update(params)

    TRAIN = False

    MODEL_SAVE_DIR = os.path.join("models")

    RANDOM_SEED = 42
    EXPLANATION_RANDOM_SEEDS = 20
    EXPLANATION_RANDOM_SEED = 1
    SMOOTHING_ALPHA = 0.001
    N_INNER_SAMPLES = 4
    """
    # Data Loading ---------------------------------------------------------------------------------
    data = pd.read_csv("housing.csv")
    target_label = "median_house_value"
    data_y = data[target_label].values
    data_x = data.drop(columns=[target_label])
    feature_names = list(data_x.columns)
    n_features = len(feature_names)
    n_samples = len(data_x)
    print("Feature Names:", feature_names)
    print("Target label:", target_label)
    
    # Data Preprocessing ---------------------------------------------------------------------------
    #data_y = FunctionTransformer(np.log10).fit_transform(data_y.reshape(-1, 1))
    data_y = MinMaxScaler().fit_transform(data_y.reshape(-1, 1))

    data_x["total_bedrooms"] = SimpleImputer(strategy="median").fit_transform(
        data_x["total_bedrooms"].values.reshape(-1, 1))

    cols_to_log = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]
    data_x[cols_to_log] = FunctionTransformer(np.log).fit_transform(
        data_x[cols_to_log].values)

    data_x["ocean_proximity"] = OrdinalEncoder().fit_transform(
        data_x["ocean_proximity"].values.reshape(-1, 1))

    data_x[data_x.columns] = MinMaxScaler().fit_transform(data_x.values)

    data_y = pd.Series(data_y.reshape(-1), name=target_label, dtype=float)
    """

    # Neural Network Data --------------------------------------------------------------------------
    data = fetch_california_housing()
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)

    feature_names = data["feature_names"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.log10(y)
    #y = FunctionTransformer(np.log10).fit_transform(y.reshape(-1, 1))

    def label_inverse(values):
        return 10 ** values

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Convert to 2D PyTorch tensors
    x_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Model Definition -----------------------------------------------------------------------------
    class NeuralNetwork(nn.Module):

        def __init__(self, n_input, n_classes):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(n_input, 50),
                nn.ReLU(),
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 5),
                nn.Linear(5, n_classes)
            )

        def forward(self, x):
            x = self.model(x)
            return x

    model = NeuralNetwork(n_input=n_features, n_classes=1)
    network_loss_function = nn.MSELoss()
    network_validation_loss = mean_absolute_error
    network_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training Batch Model -------------------------------------------------------------------------
    if TRAIN:
        EPOCHS = 1000
        BATCH_SIZE = 100
        #TRAIN_SIZE = int(0.7 * n_samples)
        #data_x, data_y = shuffle(data_x, data_y, random_state=RANDOM_SEED)

        #x_data = np.asarray(data_x, dtype=float)
        #y_data = np.asarray(data_y.values, dtype=float)

        #x_train = torch.Tensor(x_data[:TRAIN_SIZE])
        #y_train = torch.Tensor(y_data[:TRAIN_SIZE])
        dataset_train = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE)

        #x_test = torch.Tensor(x_data[TRAIN_SIZE:])
        #y_test = torch.Tensor(y_data[TRAIN_SIZE:])
        dataset_test = TensorDataset(x_test, y_test)
        test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE)

        best_r2 = -1
        mae_of_best = np.inf
        best_weights = None
        history = []

        for epoch in range(EPOCHS):
            train_loss = 0.
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                network_optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = network_loss_function(outputs, labels)
                loss.backward()
                network_optimizer.step()
                train_loss += loss.item()

            model.eval()
            predictions = model(x_test)
            mae_test = mean_absolute_error(predictions, y_test)
            r2_test = r2_score(predictions, y_test)

            print(f'[{epoch + 1}] '
                  f'train-loss: {train_loss:.6f} '
                  f'test-mae: {mae_test:.6f}, '
                  f'test-r2: {r2_test:.6f}')

            if r2_test > best_r2:
                best_r2 = r2_test
                mae_of_best = mae_test
                best_weights = copy.deepcopy(model.state_dict())

        save_name = '_'.join(
            ("california", str(round(float(best_r2), 6)), str(round(float(mae_of_best), 6))))
        save_path = os.path.join(MODEL_SAVE_DIR, save_name)

        if not os.path.exists(save_path):
            print(f"Saving best Model with a R2 of {best_r2:.6f} MAE of {mae_of_best:.6f}")
            torch.save(best_weights, save_path)

    # Get Model ------------------------------------------------------------------------------------

    LOAD_PATH = os.path.join("models", "california_0.805888_0.077409")
    model.load_state_dict(torch.load(LOAD_PATH))

    # Create Explainer Objects ---------------------------------------------------------------------

    model_function = TorchWrapper(model)
    grid_size = 20

    X, y = shuffle(X, y, random_state=RANDOM_SEED)
    data_x = pd.DataFrame(X, columns=feature_names)
    data_y = pd.Series(y.reshape(-1), name="target")

    storage = OrderedReservoirStorage(
        store_targets=False,
        size=1000,
        constant_probability=1.
    )
    incremental_explainer = IncrementalPDP(
        model_function=model_function,
        feature_names=feature_names,
        gridsize=grid_size,
        dynamic_setting=True,
        smoothing_alpha=0.01,
        pdp_feature='Longitude',
        storage=storage,
        storage_size=100,
        output_key='output',
        min_max_grid=True
    )
    batch_explainer = BatchPDP(
        pdp_feature='Longitude',
        gridsize=grid_size,
        model_function=model_function,
        output_key='output'
    )

    scaling_mean = scaler.mean_[7]
    scaling_std = scaler.scale_[7]

    def inverse_x(values):
        return values * scaling_std + scaling_mean

    x_transform = inverse_x
    y_transform = label_inverse

    # Explanation-Phase ----------------------------------------------------------------------------
    print(f"Starting Explanation for {n_samples}")
    start_time = time.time()
    for (n, (x_i, y_i)) in enumerate(iter_pandas(data_x, data_y), start=1):
        incremental_explainer.explain_one(x_i)
        batch_explainer.update_storage(x_i)
        if n % 200 == 0:
            print(n)

    batch_explainer.explain_one(x_i)
    batch_pdp_x, batch_pdp_y = batch_explainer.pdp

    fig, axes = incremental_explainer.plot_pdp(
        title=f"iPDP California Data",
        return_plot=True,
        show_pdp_transition=True,
        x_transform=x_transform,
        y_transform=y_transform,
        batch_pdp=(batch_pdp_x, batch_pdp_y),
        x_min=-124.5, x_max=-114,
        y_label="Median House Price"
    )
    axes[0].legend()
    plt.savefig(os.path.join("batch", "batch/california.pdf"))
    plt.show()
