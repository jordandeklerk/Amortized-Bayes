import numpy as np
import os
from src.model.model import Model
from train import train


class VariationalInferenceExperiment:
    """
    A class to conduct variational inference experiments. Supports A-VI with nerual networks, mean field VI, and constant-VI.

    Parameters:
        x (torch.Tensor): Training dataset.
        x_test (torch.Tensor): Testing dataset.
        z_dim (int): Dimensionality of the latent space.
        like_dim (int): Dimensionality of the likelihood parameter space.
        n_epochs (int): Number of epochs for training.
        nn_widths (list of int): List of neural network widths to experiment with.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        batch_size (int): Batch size for training.
        n_obs (int): Number of observations in the training dataset.
        data_set (str): Name of the dataset being used.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        x_dim (int): Dimensionality of the input space (inferred from `x`).
        output_dir (str): Directory to save the experiment results.

    Methods:
        initialize_model: Initializes a model with specific configurations.
        train_model: Trains a model and returns loss metrics.
        run_experiments: Runs experiments with different variational inference methods.
        save_results: Saves the results of the experiments to disk.
    """
    def __init__(self, x, x_test, z_dim, like_dim, n_epochs, nn_widths, lr, weight_decay, batch_size, n_obs, data_set, device):
        self.x = x
        self.x_test = x_test
        self.z_dim = z_dim
        self.like_dim = like_dim
        self.n_epochs = n_epochs
        self.nn_widths = nn_widths
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_obs = n_obs
        self.data_set = data_set
        self.device = device
        self.x_dim = x.shape[1]

        self.output_dir = "results_VAE"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_model(self, nn_width, use_avi, const_z=False):
        """
        Initializes a variational autoencoder model with specified configuration.

        Parameters:
            nn_width (int): Width of the neural network (number of neurons in hidden layers).
            use_avi (bool): Whether to use amortized variational inference.
            const_z (bool, optional): Whether to use a constant value for the latent variable `z`. Defaults to False.

        Returns:
            torch.nn.Module: Initialized model ready for training.
        """
        return Model(self.x_dim, z_dim=self.z_dim, like_dim=self.like_dim, n_obs=self.n_obs, use_avi=use_avi, const_z=const_z, hidden_dim=nn_width, mc_samples=1).to(self.device)

    def train_model(self, model, seed, save_mse_test=False):
        """
        Trains the model using the specified seed and training parameters, optionally evaluating on a test set.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            seed (int): Seed for random number generation to ensure reproducibility.
            save_mse_test (bool, optional): If True, evaluates the model on the test set and saves the MSE.

        Returns:
            Tuple containing trained model, loss, training time, training MSE, and testing MSE (if applicable).
        """
        return train(seed, model, self.x, n_epochs=self.n_epochs, n_obs=self.n_obs, batch_size=self.batch_size,
                     print_output=True, lr=self.lr, weight_decay=self.weight_decay, save_mse=True, save_mse_test=save_mse_test, x_test=self.x_test)

    def run_experiments(self, seed):
        """
        Runs experiments with different configurations specified by `nn_widths` and other attributes.

        Parameters:
            seed (int): Seed for random number generation to ensure reproducibility.

        Effects:
            Trains models with different configurations and saves results to disk.
        """
        n_iter = self.x.shape[0] // self.batch_size * self.n_epochs
        loss_all = np.empty((n_iter, 2 + len(self.nn_widths)))
        mse_train_all = np.empty((self.n_epochs, 2 + len(self.nn_widths)))
        mse_test_all = np.empty((self.n_epochs, 2 + len(self.nn_widths)))

        configs = [(width, True, False) for width in self.nn_widths] + [(0, False, False), (0, False, True)] # Last two for F-VI and constant VI

        for i, (width, use_avi, const_z) in enumerate(configs):
            print(f"\tRunning {'A-VI' if use_avi else 'F-VI'} with width = {width}, const_z = {const_z}")
            model = self.initialize_model(width, use_avi, const_z)
            _model, loss, _time, mse, mse_test = self.train_model(model, seed, 'const_z' in locals())
            loss_all[:, i] = loss
            mse_train_all[:, i] = mse
            mse_test_all[:, i] = mse_test

        self.save_results(seed, loss_all, mse_train_all, mse_test_all)

    def save_results(self, seed, loss_all, mse_train_all, mse_test_all):
        """
        Saves the aggregated results of the experiments to disk.

        Parameters:
            seed (int): Seed used for the experiments, used in naming the output files.
            loss_all (numpy.ndarray): Array of loss values from all experiments.
            mse_train_all (numpy.ndarray): Array of training MSE values from all experiments.
            mse_test_all (numpy.ndarray): Array of testing MSE values from all experiments.
        """
        np.save(f"{self.output_dir}/vae_{self.data_set}_loss_{seed}.npy", loss_all)
        np.save(f"{self.output_dir}/vae_{self.data_set}_mse_{seed}.npy", mse_train_all)
        np.save(f"{self.output_dir}/vae_{self.data_set}_mse_test_{seed}.npy", mse_test_all)