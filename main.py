import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.parser import args
from experiment import VariationalInferenceExperiment
from src.utils.config import dataset_configs, train_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_set = args.dataset
config = dataset_configs[data_set]
x_dim = config["x_dim"]

for batch_idx, (x, _) in enumerate(train_loader):
    if batch_idx == 0:
        x = x.view(-1, x_dim)
        x = x.to(device)
        break

for batch_idx, (x_test, _) in enumerate(test_loader):
    if batch_idx == 0:
        x_test = x_test.view(-1, x_dim)
        x_test = x_test.to(device)
        break


# Begin experiments
experiment = VariationalInferenceExperiment(x, x_test, args.z_dim, args.like_dim, args.epochs, args.nn_widths, args.lr, args.weight_decay, args.batch_size, args.n_obs, args.dataset, device)
experiment.run_experiments(args.seed)

# Run experiments across different seeds for robustness
init_seed = 415
for i in range(5):
    seed = init_seed + i
    print("seed: ", seed)
    experiment = VariationalInferenceExperiment(x, x_test, args.z_dim, args.like_dim, args.epochs, args.nn_widths, args.lr, args.weight_decay, args.batch_size, args.n_obs, args.dataset, device)
    experiment.run_experiments(seed)


# Plotting

# Plot for just the main seed
seed = args.seed

loss_all = np.load(f"results_VAE/vae_{data_set}_loss_{seed}.npy")
mse_all = np.load(f"results_VAE/vae_{data_set}_mse_{seed}.npy")
mse_test_all = np.load(f"results_VAE/vae_{data_set}_mse_test_{seed}.npy")

# Negative ELBO
n_width = len(args.nn_widths)
plotted_widths = np.array([1, 4, 16])
epochs = list(range(loss_all.shape[0]))

sns.set_theme(style="white")

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_all[:, n_width + 1], label='Constant VI', color='red', linewidth=2)
plt.plot(epochs, loss_all[:, n_width], label='F-VI', color='black', linewidth=2)

for i, width in enumerate(args.nn_widths):
    plt.plot(epochs, loss_all[:, i], label=f'k = {width}', linewidth=2)

plt.title("Bayesian Neural Network Loss")
plt.xlabel("Iterations")
plt.ylabel("Negative ELBO")
plt.yscale("log")
plt.xscale("log")

plt.legend(title="Model")

plt.savefig(f"{data_set}_elbo_BNN_{seed}.pdf")
plt.show()

# Reconstruction MSE Training Set
n_width = len(args.nn_widths)
plotted_widths = np.array([1, 4, 16])
epochs = list(range(mse_all.shape[0]))

sns.set_theme(style="white")

plt.figure(figsize=(10, 6))
plt.plot(epochs, mse_all[:, n_width + 1], label='Constant VI', color='red', linewidth=2)
plt.plot(epochs, mse_all[:, n_width], label='F-VI', color='black', linewidth=2)

for i, width in enumerate(args.nn_widths):
    plt.plot(epochs, mse_all[:, i], label=f'k = {width}', linewidth=2)

plt.title("Bayesian Neural Network Reconstruction MSE on Training Set")
plt.xlabel("Iterations")
plt.ylabel("Reconstruction MSE")
plt.yscale("log")
plt.xscale("log")

plt.legend(title="Model")

plt.savefig(f"{data_set}_elbo_BNN_{seed}.pdf")
plt.show()

# Reconstruction MSE Test Set
n_width = len(args.nn_widths)
plotted_widths = np.array([1, 4, 16])

sns.set_theme(style="white")

plt.figure(figsize=(10, 6))

plt.plot(mse_test_all[:, n_width + 1], label='Const VI', color='red', linewidth=2)

for i, width in enumerate(args.nn_widths):
  plt.plot(mse_test_all[:, i], label=f'k = {width}', linewidth=2)

plt.title("Bayesian Neural Network Reconstruction MSE on Test Set")
plt.xlabel("Iterations")
plt.ylabel("Reconstruction MSE (Test)")
plt.xscale("log")
plt.yscale("log")

plt.legend(title="Model Configuration")

plt.savefig(f"{data_set}_mse_test_BNN_{seed}.pdf")
plt.show()

# Plot for all seeds
def plot_metrics_for_all_seeds(init_seed, num_seeds, args, data_set):
    sns.set_theme(style="white")

    total_plots = num_seeds * 3
    n_width = len(args.nn_widths)

    plt.figure(figsize=(20, 30))

    for seed_index in range(num_seeds):
        seed = init_seed + seed_index

        loss_all = np.load(f"results_VAE/vae_{data_set}_loss_{seed}.npy")
        mse_all = np.load(f"results_VAE/vae_{data_set}_mse_{seed}.npy")
        mse_test_all = np.load(f"results_VAE/vae_{data_set}_mse_test_{seed}.npy")

        epochs = list(range(loss_all.shape[0]))

        # Plotting Negative ELBO
        plt.subplot(num_seeds, 3, seed_index * 3 + 1)
        plt.plot(epochs, loss_all[:, n_width + 1], label='Constant VI', color='red', linewidth=2)
        plt.plot(epochs, loss_all[:, n_width], label='F-VI', color='black', linewidth=2)
        for i, width in enumerate(args.nn_widths):
            plt.plot(epochs, loss_all[:, i], label=f'k = {width}', linewidth=2)
        plt.title(f"Seed {seed}: Bayesian Neural Network Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Negative ELBO")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(title="Model")

        # Plotting Reconstruction MSE on Training Set
        plt.subplot(num_seeds, 3, seed_index * 3 + 2)
        plt.plot(epochs, mse_all[:, n_width + 1], label='Constant VI', color='red', linewidth=2)
        plt.plot(epochs, mse_all[:, n_width], label='F-VI', color='black', linewidth=2)
        for i, width in enumerate(args.nn_widths):
            plt.plot(epochs, mse_all[:, i], label=f'k = {width}', linewidth=2)
        plt.title(f"Seed {seed}: Reconstruction MSE on Training Set")
        plt.xlabel("Iterations")
        plt.ylabel("Reconstruction MSE")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(title="Model")

        # Plotting Reconstruction MSE on Test Set
        plt.subplot(num_seeds, 3, seed_index * 3 + 3)
        plt.plot(epochs, mse_test_all[:, n_width + 1], label='Const VI', color='red', linewidth=2)
        for i, width in enumerate(args.nn_widths):
            plt.plot(epochs, mse_test_all[:, i], label=f'k = {width}', linewidth=2)
        plt.title(f"Seed {seed}: Reconstruction MSE on Test Set")
        plt.xlabel("Iterations")
        plt.ylabel("Reconstruction MSE (Test)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(title="Model Configuration")

    plt.tight_layout()
    plt.show()

plot_metrics_for_all_seeds(init_seed, 5, args, args.dataset)

# Plot Computation Time
def plot_iterations_to_tolerance(dataset, tol=0.07, init_seed=415, num_seed=5, n_algorithms=6):
    def Iter_to_tol(mse, tol, grid=1, min_epoch=0, max_epoch=5000):
        iteration = min_epoch
        error = tol + 1
        while (error > tol and iteration < (max_epoch - 1)):
            iteration += grid
            error = mse[iteration]

        return iteration

    exp_seed = [init_seed + i for i in range(num_seed)]

    iter_to_tol = np.empty((num_seed, n_algorithms))
    mse_final = np.empty((num_seed, n_algorithms))

    for i in range(num_seed):
        mse_all = np.load(f"results_VAE/vae_{data_set}_mse_{exp_seed[i]}.npy")

        for j in range(n_algorithms):
            iter_to_tol[i, j] = Iter_to_tol(mse_all[:, j], tol=tol)
            mse_final[i, j] = mse_all[-1, j]

    iter_to_tol = iter_to_tol[:, [4, 5, 0, 1, 2, 3]]
    algo_names = ["F-VI", "const"]
    algo_names += ["k=" + str(args.nn_widths[i]) for i in range(n_algorithms - 2)]

    plt.figure(figsize=(12,8))
    plt.boxplot(iter_to_tol, labels=algo_names, patch_artist=True,
                boxprops=dict(facecolor="red"), vert=False)
    plt.xscale("log")
    plt.grid(which='minor', visible='true', c='grey', alpha=0.25)
    plt.gca().invert_yaxis()

    plt.title("Bayesian Neural Network")
    plt.xlabel("Number of iterations to MSE<0.07")
    plt.show()

plot_iterations_to_tolerance(args.dataset)