import numpy as np
import torch
import torch.nn as nn
import wandb
import random
import time
import logging
from src.utils.parser import args
from src.utils.optimizer import AdamOptimizer

def train(seed, model, x, n_epochs, n_obs, batch_size, print_output=False, lr=args.lr, weight_decay=args.weight_decay,
           save_mse=False, save_mse_test=False, x_test=None, use_wandb=False):
    """
    Trains a given model using the specified parameters and data.

    Parameters:
    - seed (int): Seed for random number generators to ensure reproducibility.
    - model (torch.nn.Module): The model to be trained.
    - x (torch.Tensor): The input data for training.
    - n_epochs (int): Number of epochs to train the model.
    - n_obs (int): Number of observations in the training dataset.
    - batch_size (int): Size of batches for training.
    - print_output (bool, optional): If True, prints training progress and information (default: False).
    - lr (float, optional): Learning rate for the optimizer (default: 1e-3).
    - save_mse (bool, optional): If True, saves the Mean Squared Error (MSE) on the training dataset after each epoch (default: False).
    - save_mse_test (bool, optional): If True, and if `x_test` is provided, saves the MSE on the test dataset after each epoch (default: False).
    - x_test (torch.Tensor, optional): The input data for testing to evaluate the model's performance (default: None).
    - use_wandb (bool, optional): If True, uses wandb for logging (default: False).

    Returns:
    - model (torch.nn.Module): The trained model.
    - loss_saved (numpy.ndarray): Array containing the loss values for each iteration.
    - run_time (float): Total training time.
    - mse_saved (numpy.ndarray): MSE values for the training dataset for each epoch if `save_mse` is True; otherwise, an empty array.
    - mse_saved_test (numpy.ndarray): MSE values for the test dataset for each epoch if `save_mse_test` is True and `x_test` is provided; otherwise, an empty array.
    """
    if use_wandb:
        wandb.init(project="project_name", entity="entity_name")
        wandb.config = {
            "learning_rate": lr,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "weight_decay": weight_decay
        }

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    optimizer = AdamOptimizer(model.parameters(), lr=lr, wd=weight_decay)

    n_batches = max(n_obs // batch_size, 1)

    if print_output:
        logging.info("Starting training VAE...")

    model.train()
    loss_saved = np.empty(n_epochs * n_batches)
    mse_saved = np.empty(n_epochs)
    mse_saved_test = np.empty(n_epochs)
    index_saved = 0

    start_time = time.time()
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            device = next(model.parameters()).device
            x_batch = x[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            loss = -model.compute_elbo(x_batch, batch_idx, batch_size)
            loss_saved[index_saved] = loss.item()
            index_saved += 1

            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"batch_loss": loss.item()})

        if save_mse:
            mse = model.reconstruction_mse(x).item()
            mse_saved[epoch] = mse

            if use_wandb:
                wandb.log({"epoch": epoch, "mse_train": mse})

        if save_mse_test and x_test is not None:
            mse_test = model.reconstruction_mse(x_test).item()
            mse_saved_test[epoch] = mse_test

            if use_wandb:
                wandb.log({"epoch": epoch, "mse_test": mse_test})

        if epoch % 500 == 0 and print_output:
            logging.info(f"\tEpoch: {epoch} \tLoss: {loss.item()}")
            if save_mse:
                logging.info(f"\tMSE: {mse_saved[epoch]}")

    end_time = time.time()
    run_time = end_time - start_time

    if use_wandb:
        wandb.log({"total_runtime": run_time})

    return model, loss_saved, run_time, mse_saved, mse_saved_test