---
title: Closing the Amortization Gap in Bayesian Deep Generative Models
date: "2024-06-07T22:12:03.284Z"
description: "Explore the integration of amortized variational inference (A-VI) with deep Bayesian VAEs. Delve into the theoretical foundations and practical applications, including a detailed PyTorch implementation, to close the amortization gap in Bayesian modeling. Discover through comprehensive numerical experiments how A-VI enhances computational efficiency and accuracy in modeling on benchmark imaging datasets."
---
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kVCeCMQ1h9dbbb_jrjrbXK4IEIWj2xL9?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)

## Overview

Amortized variational inference (A-VI) has emerged as a promising approach to enhance the efficiency of Bayesian deep generative models. In this project, we aim to investigate the effectiveness of A-VI in closing the amortization gap between A-VI and traditional variational inference methods, such as factorized variational inference (F-VI), or mean-field variational inference. We conduct numerical experiments on benchmark imaging datasets to compare the performance of A-VI with varying neural network architectures against F-VI and constant-VI. 

Our findings demonstrate that A-VI, when implemented with sufficiently deep neural networks, can achieve the same evidence lower bound (ELBO) and reconstruction mean squared error (MSE) as F-VI while being 2 to 3 times computationally faster. These results highlight the potential of A-VI in addressing the amortization interpolation problem and suggest that a deep encoder-decoder linear neural network with full Bayesian inference over the latent variables can effectively approximate an ideal inference function. This work paves the way for more efficient and scalable Bayesian deep generative models.

## Introduction

In the Bayesian paradigm, statistical inference regarding unknown variables is predicated on computations involving posterior probability densities. Due to the often intractable nature of these densities, which typically lack an analytic form, estimation becomes crucial. Classical methods for estimating the posterior distribution in Bayesian inference such as MCMC are known to be computationally expensive at test time as they rely on repeated evaluations of the likelihood function and, therefore, require a new set of likelihood evaluations for each observation. In contrast, Variational Inference (VI) offers a compelling solution by recasting the difficult task of estimating complex posterior densities into a more manageable optimization problem. The essence of VI lies in selecting a parameterized distribution family, $\mathcal{Q}$, and identifying the member that minimizes the Kullback-Leibler (KL) divergence from the posterior

$$
\begin{equation}
q^* = \arg \min _{q \in \mathcal{Q}} \mathrm{KL}(q(\theta, \mathbf{z}) \| p(\theta, \mathbf{z} \mid \mathbf{x})).
\end{equation}
$$

This process enables the approximation of the posterior with $q^*$, thereby delineating the VI objective to entail the selection of an appropriate variational family $\mathcal{Q}$ for optimization. Common practice in VI applications involves the adoption of the factorized, or mean-field, family. This family is characterized by the independence of the variables

$$
\begin{equation}
\mathcal{Q}_{\mathrm{F}}=\left\{q: q(\theta, \mathbf{z})=q_0(\theta) \prod_{n=1}^N q_n\left(z_n\right)\right\},
\end{equation}
$$

wherein each latent variable is represented by a distinct factor $q_n$.

Contrary to the VI framework, the amortized family $\mathcal{Q}_{\mathrm{A}}$ leverages a stochastic inference function $f_{\phi}(x_n)$ to dictate the variational distribution of each latent variable $z_n$, typically instantiated through a neural network, facilitating the parameter mapping for each latent variable's approximating factor $q_n(z_n)$ (Ankush):

$$
\begin{equation}
\mathcal{Q}_{\mathrm{A}}=\left\{q: q(\theta, \mathbf{z})=q_0(\theta) \prod_{n=1}^N q\left(z_n ; f_\phi\left(x_n\right)\right)\right\}.
\end{equation}
$$

This paradigm, known as *amortized variational inference* (A-VI), optimizes the approximation of the posterior and the inference function simultaneously. Therefore, inference on a single observation can be performed efficiently through a single forward pass through the neural network, framing Bayesian inference as a prediction problem: for *any* observation, the neural network is trained to predict the posterior distribution, or a quantity that allows the network to infer the posterior without any further simulations.

## Background

Variational Autoencoders (VAEs) belong to a category of machine learning models known as latent variable models. These models operate on the principle that the observable characteristics of data can be derived from a set of underlying, unobservable variables, termed as latent variables. The essence of VAEs is to circumvent the direct construction of a latent space. Instead, these models employ an Encoder-Decoder architecture to access this space indirectly.

<p align="center">
  <img src="vae.png" alt="Variational Autoencoder Architecture">
  Figure 1:<em> The architecture of a Variational Autoencoder</em>
</p>

From Figure 1, we can see that this architecture is split into two major components:
- The **encoder** component of a VAE constructs a distribution based on the input data. This distribution is utilized to infer latent variables that are likely responsible for producing the input data. In technical terms, the encoder learns a set of parameters $\theta_1$, which define a distribution $Q(x, \theta_1)$. From this distribution, latent variables are sampled that maximize the probability of the observed data, represented as $\mathbb{P}(\mathbf{X}|\mathbf{z})$.
- The **decoder** component takes these inferred latent variables and generates data that aligns with the actual observed distribution of the dataset. It uses the latent variables sampled by the encoder as inputs. Specifically, the decoder learns another set of parameters, $\theta_2$, that map these latent variables back to the original data distribution through a function $f(\mathbf{z}, \theta_2)$.

In essence, VAEs model the process of data generation through encoding to and decoding from a latent space, aiming to produce outputs that closely approximate the true distribution of the data.

### Inference in Variational Autoencoders

Consider $x$ as the observed variable, \(z\) as the hidden variable, and $p(x, z)$ representing their joint distribution. For a dataset $X = \{x_1, x_2, \ldots, x_N\}$, our goal is to maximize the marginal $\log$-likelihood with respect to the model parameters $\theta$, expressed as

$$
\log p_\theta(X) = \sum_{i=1}^N \log p_\theta(x_i) = \sum_{i=1}^N \log \int p_\theta(x_i, z_i) \, dz_i.
$$

However, calculating the marginal log-likelihood directly is not feasible due to the integral over the hidden variable $z$. To address this, Variational Autoencoders (VAEs) employ an inference network $q_\phi(z | x)$ as an approximation to the actual posterior $p(z | x)$ and optimize the Evidence Lower Bound (ELBO) relative to both the model parameters $\theta$ and the parameters $\phi$ of the inference network:

$$
\log p(x) = \mathbb{E}_{q(z | x)}\left[\log \left(\frac{p(x, z)}{q(z | x)}\right)\right] + \text{KL}(q(z | x) \| p(z | x)) \geq \mathbb{E}_{q(z | x)}\left[\log \left(\frac{p(x, z)}{q(z | x)}\right)\right] = \mathcal{L}_{\text{VAE}}[q].
$$

The ELBO is equivalent to the true log likelihood when $q(z | x) = p(z | x)$. Typically, $q(z | x)$ is chosen to be a factorized Gaussian for its simplicity and computational efficiency. VAEs leverage the inference network (also known as the encoder or recognition network) to generalize inference across the entire dataset efficiently. The model is trained by stochastically optimizing the ELBO through the reparametrization trick, introduced by Kingma.

### The Reparameterization Trick

The reparametrization trick was initially introduced in [Kingma](https://arxiv.org/abs/1312.6114). It is important to understand this trick because we need a way to generate samples from $q_{\theta}(z | x)$. Our explanation here follows [this great post](https://gregorygundersen.com/blog/2018/04/29/reparameterization/) very closely.

<p align="center">
  <img src="reparam.png" alt="Reparameterization Trick">
  Figure 2:<em> The reparameterization trick illustrated</em>
</p>

#### Non-differentiable Expectations

Consider the scenario where we need to compute the gradient concerning $\theta$ for the expectation,

$$
\mathbb{E}_{p(z)}[f_\theta(z)]
$$

Here, $p$ represents a probability density. Assuming that $f_\theta(z)$ is differentiable, the computation of the gradient becomes straightforward:

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p(z)}[f_\theta(z)] &= \nabla_\theta\left[\int_z p(z) f_\theta(z) \, dz\right] \\
&= \int_z p(z)\left[\nabla_\theta f_\theta(z)\right] \, dz \\
&= \mathbb{E}_{p(z)}[\nabla_\theta f_\theta(z)].
\end{aligned}
$$

In essence, the gradient of the expectation equals the expectation of the gradient. However, complications arise if the density $p$ also depends on $\theta$:

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p_\theta(z)}[f_\theta(z)] &= \nabla_\theta\left[\int_z p_\theta(z) f_\theta(z) \, dz\right] \\
&= \int_z \nabla_\theta\left[p_\theta(z) f_\theta(z)\right] \, dz \\
&= \int_z f_\theta(z) \nabla_\theta p_\theta(z) \, dz + \int_z p_\theta(z) \nabla_\theta f_\theta(z) \, dz \\
&= \underbrace{\int_z f_\theta(z) \nabla_\theta p_\theta(z) \, dz}_{\text{How about this part?}} + \mathbb{E}_{p_\theta(z)}[\nabla_\theta f_\theta(z)].
\end{aligned}
$$

The initial part of the final equation doesn't necessarily equate to an expectation. While Monte Carlo techniques allow sampling from $p_\theta(z)$, they do not ensure the gradient of this sampling process is obtainable. This poses no issue if there exists an analytic solution for $\nabla_\theta p_\theta(z)$, but such a solution is not always available.

With a clearer understanding of the challenge, let's explore the application of the reparameterization trick to our example. Following Kingma's notation, vectors are denoted in bold, $\mathbf{v}^{(i)}$ represents the $i$-th sample of vector $\mathbf{v}$, and $l \in L$ signifies the $l$-th Monte Carlo sample. The reparameterization trick is utilized to shift from a gradient of an expectation to an expectation of a gradient, facilitating the use of Monte Carlo estimations for gradients involving expectations. This transition is pivotal, especially when the function $g_{\boldsymbol{\theta}}$ is differentiable, a point that Kingma underscores. The mathematical flow is as follows:

1. Noise variables $\boldsymbol{\epsilon}$ are drawn from a predefined distribution $p(\boldsymbol{\epsilon})$.
2. These variables, along with input $\mathbf{x}$, are transformed by $g_{\boldsymbol{\theta}}$ to obtain $\mathbf{z}$.
3. The expectation with respect to $p_\theta(\mathbf{z})$ is equated to an expectation with respect to $p(\boldsymbol{\epsilon})$, leveraging the function $g_\theta$.
4. Subsequently, the gradient of the expectation with respect to $\boldsymbol{\theta}$ is expressed and approximated via Monte Carlo sampling.

Formally, this can be represented as:

$$
\begin{aligned}
\boldsymbol{\epsilon} & \sim p(\boldsymbol{\epsilon}) \\
\mathbf{z} & =g_{\boldsymbol{\theta}}(\boldsymbol{\epsilon}, \mathbf{x}) \\
\mathbb{E}_{p_\theta(\mathbf{z})}\left[f\left(\mathbf{z}^{(i)}\right)\right] & =\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[f\left(g_\theta\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
\nabla_\theta \mathbb{E}_{p_{\boldsymbol{\theta}}(\mathbf{z})}\left[f\left(\mathbf{z}^{(i)}\right)\right] & =\nabla_\theta \mathbb{E}_{p(\boldsymbol{\epsilon})}\left[f\left(g_{\boldsymbol{\theta}}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
& =\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\theta}} f\left(g_{\boldsymbol{\theta}}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
& \approx \frac{1}{L} \sum_{l=1}^L \nabla_{\boldsymbol{\theta}} f\left(g_{\boldsymbol{\theta}}\left(\epsilon^{(l)}, \mathbf{x}^{(i)}\right)\right).
\end{aligned}
$$

It's important to emphasize that this explanation aligns with Kingma's reasoning. The challenge isn't technically about the inability to backpropagate through a "random node". The issue is more about backpropagation not providing an estimate of the derivative. Without employing the reparameterization trick, there's no assurance that sampling a large number of $\mathbf{z}$ values will lead to a correct estimate of $\nabla_{\theta}$.

Moreover, this issue directly relates to the problem we encounter with estimating the ELBO. We see that

$$
\begin{aligned}
\text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}) & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right], \\
& \downarrow \\
\nabla_{\theta, \phi} \text{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}) & = \nabla_{\theta, \phi}\left[\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]\right].
\end{aligned}
$$

Notice that the equation above doesn't quite resemble what is computed in a standard VAE. In his paper, Kingma introduces two estimators, denoted as $\mathcal{L}^A$ and $\mathcal{L}^B$. The equation for $\mathcal{L}^A$ is shown above, while $\mathcal{L}^B$ applies when there's an analytic solution for the KL-divergence term in the ELBO, such as when assuming Gaussian distributions for both the prior $p_{\boldsymbol{\theta}}(\mathbf{z})$ and the posterior approximation $q_{\phi}(\mathbf{z} | \mathbf{x})$:

$$
\nabla_{\theta, \phi} \mathcal{L}^B = -\nabla_{\theta, \phi} \left[\text{KL}\left[q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(i)}) \| p_{\theta}(\mathbf{z})\right]\right] + \nabla_{\theta, \phi} \left[\frac{1}{L} \sum_{l=1}^L \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)} \mid \mathbf{z}^{(l)})\right].
$$

Now, with the capability to compute the full loss through a series of differentiable operations, gradient-based optimization techniques can be applied to maximize the ELBO.

#### Implementation

Closing the loop with implementation details, the standard VAE follows the equation derived by Kingma for the KL term in Appendix 2. This equation presents the model with the likelihood as a "decoder" and the approximate posterior as an "encoder"

$$
\mathcal{L}^B = -\text{KL}\left[q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)}) \| p_\theta(\mathbf{z})\right] + \frac{1}{L} \sum_{l=1}^L \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)} \mid \mathbf{z}^{(l)}).
$$

This can be understood as follows: the KL-divergence term encourages the encoder's output to resemble the fixed prior distribution $p_{\boldsymbol{\theta}}(\mathbf{z})$. Ideally, if the encoder perfectly captures both the real posterior and the prior, Bayes' rule would imply $p(\mathbf{x})=p(\mathbf{x} \mid \mathbf{z})$, aligning with the objectives of a generative model. Through the reparameterization trick, we can sample $\mathbf{z}$ and conditionally generate realistic samples of $\mathbf{x}$.

A single computational graph pass for Gaussian priors and posteriors might look like the following:

$$
\begin{aligned}
\boldsymbol{\mu}_x, \boldsymbol{\sigma}_x & = \text{Encoder}(\mathbf{x}) & & \quad \text{Push $\mathbf{x}$ through encoder} \\
\boldsymbol{\epsilon} & \sim \mathcal{N}(0,1) & & \quad \text{Sample noise} \\
\mathbf{z} & = \boldsymbol{\epsilon} \boldsymbol{\sigma}_x + \boldsymbol{\mu}_x & & \quad \text{Reparameterize} \\
\mathbf{x}_r & = \text{Decoder}(\mathbf{z}) & & \quad \text{Push $\mathbf{z}$ through decoder} \\
\text{Recon. Loss} & = \text{MSE}(\mathbf{x}, \mathbf{x}_r) & & \quad \text{Compute reconstruction loss} \\
\text{Var. Loss} & = -\text{KL}\left[\mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\sigma}_x^2) \| \mathcal{N}(0, I)\right] & & \quad \text{Compute variational loss} \\
\text{Total Loss} & = \text{Recon. Loss} + \text{Var. Loss} & & \quad \text{Combine losses}
\end{aligned}
$$

By computing each variable in this graph through differentiable operations, backpropagation can be applied to compute gradients.

## Theoretical Preliminaries

To comprehend when A-VI can narrow the difference with F-VI, we initially consider that the variational families in Eqs. 2 and 3 employ the identical initial distribution $q_0(\theta)$. This allows us to concentrate on the characteristics of the variational distributions for $z_n$.

Recall, in F-VI, each local latent variable $z_n$ is described by a marginal distribution $q_n(z_n ; \nu_n)$ from a parametric family $\mathcal{O}_{l}$, where $\nu_n$ belongs to the parameter space $\,\mathcal{U}$. The complete family, or mean-field family, $\mathcal{Q}_F$ is defined as the product of these marginals

$$
q_0(\theta) \prod_{n=1}^N q_n(z_n ; \nu_n).
$$

We determine the best variational parameters $\nu^* = (\nu_0^*, \nu_1^*, \ldots, \nu_N^*)$ by minimizing the KL-divergence shown in Eq. 1.

Assuming $\mathcal{X}$ represents the space of $x_n$, A-VI approximates a function $f_{\phi}: \mathcal{X} \to \mathcal{U}$ across a set of inference functions $\mathcal{F}$, each parameterized by $\phi$. We also minimize the KL-divergence in Eq. 1 with respect to $\phi$ to derive the optimal variational family $\mathcal{Q}_A(\mathcal{F})$. Clearly from Eq. 2 and Eq. 3, $\mathcal{Q}_A(\mathcal{F})$ is a subset of $\mathcal{Q}_F$. This subset relationship is exemplified by considering $x_n = x_m$, which implies $f_{\phi}(x_n) = f_{\phi}(x_m)$, whereas a potential distribution $\tilde{q}(\theta, z) \in \mathcal{Q}_F$ could have $\nu_n \neq \nu_m$, confirming a strict subset.

This subset relation means A-VI cannot theoretically surpass F-VI in achieving a lower KL-divergence, leading to what is known as the *amortization gap*. To bridge this gap, the inference function $f_{\phi}$ must interpolate between $x_n$ and the optimal variational parameter $\nu_n^*$, represented by

$$
f_{\phi}(x_n) = \nu_n^*, \quad \forall n.
$$

Addressing this requirement is termed the *amortization interpolation problem*. Any function $f$ that resolves this is called an *ideal inference function*. Conditions can be set on the model $p(\theta, z, x)$ ensuring the feasibility of the ideal inference function, thereby making the amortization interpolation problem well-posed. Once the existence of an ideal inference function is established, we can explore the necessary complexity of the family of inference functions $\mathcal{F}$ to include such an ideal $f$, although this exploration is beyond the scope of this post.

In the following section, we will practically investigate the amortization interpolation problem through numerical experiments, demonstrating that a deep encoder-decoder linear neural network with full Bayesian inference on latent variables can approximate this ideal inference function and mitigate the amortization gap.

## Numerical Experiments

### Experimental Setup

We use a family of factorized Gaussians as the variational approximation. We will compare A-VI against the benchmark methods F-VI and constant-VI, which assigns the same factor $\bar{q}$ to each latent variable $z_n$. Note that, as established earlier, F-VI is the richest variational family while it should be clear that constant-VI is comparatively the poorest variational family. We will implement A-VI with neural networks with various degrees of complexity, which in this case, corresponds to varying hidden dimensions.

We optimize the KL-divergence by maximizing the evidence lower-bound (ELBO),

$$
\mathbb{E}_{q(\mathbf{z}, \theta ; \nu)} [\log p(\theta, \mathbf{z}, \mathbf{x}) - \log q(\theta, \mathbf{z})],
$$

which is estimated via Monte Carlo with a mini-batching strategy that will be described in detail in the next section.

In all of our experiments, we employ the Adam optimizer in PyTorch using the reparameterization trick to calculate the gradients. We use a learning rate of 1e-3 with weight decay of 1e-1, and train the model for 5,000 epochs. Experiments are naturally stochastic, so we repeat each numerical experiment across 5 different seeds and find that the initial results are stable. We ran these experiments on a single NVIDIA A100 GPU in Google Colab. Training time for `MNIST` and `FashionMNIST` takes less than 2 hours. Training time for `CIFAR-10` takes much longer in comparison (over 8 hours).

**Bayesian Deep Generative Model**

We explore a deep generative model on two standard image datasets, `MNIST` and `FashionMNIST`. In our model, each image is represented by a low-dimensional latent vector $\mathbf{z}_n \in \mathbb{R}^{64}$, modeled with the probability distribution

$$
p(\mathbf{x_n} | \mathbf{z_n}, \theta) = \mathcal{N}(\Omega(\mathbf{z_n}; \theta), \mathbb{I}),
$$

where $\Omega$ is a linear neural network consisting of two hidden layers with varying sizes (64, 128, 256), and uses leaky ReLU activations. The neural network parameters $\theta$ include all the weights and biases. This model is a variant of the standard VAE but incorporates a Bayesian approach to estimating a posterior distribution over $\theta$, which helps in addressing the amortization gap often observed in such models. For practicality and due to computational limits, we trained our model on 10,000 images from each dataset using full-batch training. During training, we assessed the Evidence Lower Bound (ELBO) on a mini-batch of 1,000 images ten times per epoch.

Let's first import all of the necessary libraries we will be using in this tutorial. We will be working mainly with `torch` and `torchvision`. 

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import os
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torch.optim import AdamW, Adam

from typing import Tuple

from tqdm import tqdm
import random

from ipywidgets import Widget
Widget.close_all()
```

Let's define our global arguments using the argument parser, enabling us to conveniently adjust and experiment with various hyperparameters.

```python
def get_args_parser():
    parser = argparse.ArgumentParser('Bayesian VAE', add_help=False)
    parser.add_argument('--dataset', default='MNIST', type=str, choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='dataset to use')
    parser.add_argument('--batch_size', default=10000, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=5000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-1, type=float,
                        help='weight decay')
    parser.add_argument('--z_dim', default=64, type=int,
                        help='z dimension')
    parser.add_argument('--like_dim', default=256, type=int,
                        help='likelihood dimension')
    parser.add_argument('--nn_widths', default=[1, 64, 128, 256], type=list,
                        help='neural network widths')
    parser.add_argument('--n_obs', default=10000, type=int,
                        help='number of observations')
    parser.add_argument('--mc_samples', default=100, type=int,
                        help='number of monte carlo samples')
    parser.add_argument('--seed', default=315, type=int,
                        help='seed')
    return parser

args, unknown = get_args_parser().parse_known_args()
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

We are now ready to establish the configurations for the four datasets we plan to explore. In the following setup, we will define the input dimension (`x_dim`), the hidden dimension (`hidden_dim`), and the latent dimension (`latent_dim`). 

- The `x_dim` specifies the size of the input data, which in our context refers to the dimensions of an image. It's important to note that `MNIST`, `FashionMNIST`, and `SVHN` datasets consist of grayscale images, whereas the `CIFAR-10` dataset comprises RGB images.
- The `hidden_dim` indicates the size of the neural network's hidden layers, essentially representing the network's width at each layer.
- Lastly, the `latent_dim` denotes the size of the latent space to which the Encoder compresses the input data, providing a compact representation of the original input.

These configurations are crucial for tailoring our model to effectively handle the specific characteristics of each dataset.

```python
dataset_path = '~/datasets'

dataset_configs = {
    "MNIST": {"x_dim": 28 * 28, "hidden_dim": 400, "latent_dim": 200, "dataset": MNIST},
    "FashionMNIST": {"x_dim": 28 * 28, "hidden_dim": 400, "latent_dim": 200, "dataset": FashionMNIST},
    "CIFAR10": {"x_dim": 32 * 32 * 3, "hidden_dim": 128, "latent_dim": 100, "dataset": CIFAR10},
    "SVHN": {"x_dim": 32 * 32, "hidden_dim": 128, "latent_dim": 100, "dataset": SVHN},
}

transform = transforms.Compose([transforms.ToTensor()])
kwargs = {'num_workers': 4, 'pin_memory': True}

def get_dataset(data_set_name, dataset_path, transform, **kwargs):
    if data_set_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {data_set_name}")

    config = dataset_configs[data_set_name]
    dataset_class = config["dataset"]

    if data_set_name == "SVHN":
        train_dataset = dataset_class(root=dataset_path, split='train', transform=transform, download=True)
        test_dataset = dataset_class(root=dataset_path, split='test', transform=transform, download=True)
    else:
        train_dataset = dataset_class(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = dataset_class(root=dataset_path, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

train_dataset, test_dataset = get_dataset(args.dataset, dataset_path, transform, **kwargs)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
```
#### Model Setup

The encoder is a type of neural network that takes an input, such as a 28x28 pixel image of a handwritten digit (which makes it a 784-dimensional vector), and compresses it into a smaller, or latent, representation denoted as $z$. This process involves learning to reduce the data from its original high dimensionality to a more compact form, often called a "bottleneck," because it forces the network to find an efficient way to represent the data in fewer dimensions. The function of the encoder is captured by $q_\theta(z | x)$, where it actually outputs parameters defining a Gaussian distribution from which we can sample the latent representation $z$.

```python
class Encoder(nn.Module):
    """
    Encoder module for a Variational Autoencoder (VAE), transforming input data into a latent space representation.

    The encoder consists of sequential linear layers with LeakyReLU activations, followed by separate linear layers
    for producing the mean and log variance vectors.

    Parameters:
    - input_dim (int): Dimensionality of the input data.
    - hidden_dim (int): Size of the hidden layer(s). This implementation uses two hidden layers of the same size.
    - latent_dim (int): Dimensionality of the latent space representation (output).
    - use_bias (bool, optional): Whether to include bias terms in the linear layers (default: True).
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, use_bias=True):
        super(Encoder, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0], bias=use_bias), nn.LeakyReLU(0.2)]
        layers.extend([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=use_bias) for i in range(len(hidden_dims) - 1)
        ])
        layers.append(nn.LeakyReLU(0.2))

        self.features = nn.Sequential(*layers)
        self.FC_mean = nn.Linear(hidden_dims[-1], latent_dim, bias=use_bias)
        self.FC_sd_log = nn.Linear(hidden_dims[-1], latent_dim, bias=use_bias)

    def forward(self, x):
        h = self.features(x)
        mean = self.FC_mean(h)
        sd_log = self.FC_sd_log(h)
        return mean, sd_log

    def init_weights(self, nu_mean_z=None, nu_sd_z_log=None, init_type='zero', device='cpu'):
        """
        Initializes the encoder's weights and biases, supporting custom initial values for
        mean and log standard deviation biases, and allowing for more initialization types.

        Parameters:
        - nu_mean_z: Initial value for the variational mean of z, applicable to FC_mean bias.
        - nu_sd_z_log: Initial values for the variational log standard deviation of z, applicable to FC_sd_log bias.
        - init_type (str, optional): The type of weight initialization ('zero', 'normal', etc.).
        - device: The device to allocate tensors to ('cpu', 'cuda', etc.).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'zero':
                    nn.init.zeros_(m.weight)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if nu_mean_z is not None:
            self.FC_mean.bias.data = nu_mean_z.to(device)
        if nu_sd_z_log is not None:
            self.FC_sd_log.bias.data = nu_sd_z_log.to(device)
```

On the flip side, the decoder is a complementary neural network that takes the encoded latent representation $z$ and works to reconstruct the original data from it. The decoder is represented by $p_\phi(x | z)$ and aims to output parameters that define the probability distribution of the original data points. Continuing with the handwritten digit example, where each pixel is either black or white, the output distribution for each pixel can be modeled as a Bernoulli distribution. Thus, from a given latent representation $z$, the decoder produces 784 probabilities, corresponding to the 784 pixels, to recreate the image.

The reconstruction process isn't perfect—there's information loss because the decoder operates on a condensed version of the original data. We assess this loss through the reconstruction log-likelihood, $\log p_\phi(x | z)$. This metric evaluates how well the decoder has learned to regenerate an input image $x$ from its encoded version $z$, reflecting the efficiency of the compression and reconstruction process.

```python
def Decoder(theta, z, latent_dim, hidden_dim, x_dim):
    """
    Reconstructs the network output from latent inputs using specified network architecture.

    Args:
        theta (Tensor): A flat tensor of the network's parameters (weights and biases),
                        ordered with all weights first followed by all biases.
        z (Tensor): The latent inputs to the network, typically representing encoded data.
        latent_dim (int): The size of the latent input dimension.
        hidden_dim (int): The size of the hidden layers in the network.
        x_dim (int): The size of the output dimension, or the dimensionality of the data being reconstructed.

    Returns:
        Tensor: The reconstructed output from the network.
    """
    expected_theta_size = latent_dim * hidden_dim + hidden_dim**2 + hidden_dim * x_dim + 2 * hidden_dim + x_dim
    if theta.numel() != expected_theta_size:
        raise ValueError("Theta size does not match the expected size based on dimensions.")

    indices = [latent_dim * hidden_dim, hidden_dim**2, hidden_dim * x_dim, hidden_dim, hidden_dim, x_dim]
    splits = torch.split(theta, indices)
    W1, W2, W3, b1, b2, b3 = [splits[i].reshape(shape) for i, shape in enumerate([
        (hidden_dim, latent_dim), (hidden_dim, hidden_dim), (x_dim, hidden_dim),
        (hidden_dim,), (hidden_dim,), (x_dim,)
    ])]

    LeakyRelu = nn.LeakyReLU(0.2)
    h = LeakyRelu(z @ W1.T + b1)
    h = LeakyRelu(h @ W2.T + b2)
    out = h @ W3.T + b3
    return out
```

We will need to define some utility functions to use when calculating the ELBO.

```python
# Utility functions
def gaussian_lpdf(x, mu, sigma_2):
    """
    Computes the log probability density function for a Gaussian distribution.

    Parameters:
    - x: Tensor of observed values.
    - mu: Tensor of means for the Gaussian distribution.
    - sigma_2: Tensor of variances for the Gaussian distribution.
    """
    return -0.5 * torch.sum((x - mu)**2 / sigma_2 + torch.log(sigma_2))

def log_joint_gaussian(x, mu, sigma, z, theta):
    """
    Computes the log joint probability for a dataset with Gaussian likelihood,
    standard Gaussian priors on z and theta.
    """
    like_weight = z.size(0) / x.size(0)
    return -0.5 * torch.sum(z**2) - like_weight * torch.sum((x - mu)**2) - 0.5 * torch.sum(theta**2)

def log_q(theta, z, nu_mean_theta, nu_sd_theta_2, nu_mean_z, nu_sd_z_2):
    """
    Evaluates the log density of the Gaussian variational approximation for theta and z,
    given means and variances of the variational distributions.
    """
    log_q_theta = -0.5 * torch.sum(torch.log(nu_sd_theta_2) + (theta - nu_mean_theta)**2 / nu_sd_theta_2)
    log_q_z = -0.5 * torch.sum(torch.log(nu_sd_z_2) + (z - nu_mean_z)**2 / nu_sd_z_2)
    return log_q_theta + log_q_z
```

#### Main Model

With both the encoder and decoder components in place, alongside essential utility functions, we can now introduce our central model class. 

This class acts as the backbone of our setup, incorporating options for Amortized Variational Inference (A-VI), Mean-Field Variational Inference (F-VI), and Constant Variational Inference. It encompasses methods for initializing variational parameters, applying the reparameterization technique to $z$ and $\theta$, and calculating both the Evidence Lower Bound (ELBO) and the Mean Squared Error (MSE) for reconstruction.

```python
class Model(nn.Module):
    """
    Implements a PyTorch module for variational inference in a variational autoencoder (VAE) setup.

    Parameters:
        x_dim (int): Dimensionality of the input data.
        z_dim (int): Dimensionality of the latent space.
        like_dim (int): Dimensionality of the likelihood parameter space.
        n_obs (int): Number of observations in the dataset.
        use_avi (bool): Flag to use amortized variational inference (default: True).
        hidden_dim (int): Dimensionality of the hidden layer(s) in the encoder. If set to 0, defaults to double the z_dim.
        const_z (bool): Flag to use a constant latent variable z (default: False).
        mc_samples (int): Number of Monte Carlo samples to use for estimating the ELBO.
        nu_mean_z_init (torch.Tensor or None): Initial values for the mean of the latent variable z.
        nu_sd_z_log_init (torch.Tensor or None): Initial values for the log standard deviation of the latent variable z.
        nu_mean_theta_init (torch.Tensor or None): Initial values for the mean of the likelihood parameters theta.
        nu_sd_theta_log_init (torch.Tensor or None): Initial values for the log standard deviation of the likelihood parameters theta.
        use_init_encoder (bool): Flag to initialize encoder weights manually if True.

    Methods:
        reparam: Performs the reparameterization trick to sample from the latent space and likelihood parameters.
        variational_z: Computes the variational parameters for the latent variable z.
        compute_elbo: Computes the Evidence Lower BOund (ELBO) for a given input batch.
        variational_parameters: Returns the variational parameters for both z and theta.
        reconstruction_mse: Computes the mean squared error of the reconstruction for evaluation purposes.
    """
    def __init__(self, x_dim, z_dim, like_dim, n_obs, use_avi=True, hidden_dim=0,
                 const_z=False, mc_samples=1,
                 nu_mean_z_init=None, nu_sd_z_log_init=None,
                 nu_mean_theta_init=None, nu_sd_theta_log_init=None,
                 use_init_encoder=False):

        super(Model, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.like_dim = like_dim
        self.n_obs = n_obs
        self.use_avi = use_avi
        self.const_z = const_z
        self.mc_samples = mc_samples
        self.hidden_dim = hidden_dim if hidden_dim else z_dim * 2
        self.dim_theta = z_dim * like_dim + like_dim**2 + like_dim * x_dim + 2 * like_dim + x_dim

        self.nu_mean_theta = nn.Parameter(nu_mean_theta_init if nu_mean_theta_init is not None else torch.zeros(self.dim_theta))
        self.nu_sd_theta_log = nn.Parameter(nu_sd_theta_log_init if nu_sd_theta_log_init is not None else torch.zeros(self.dim_theta) - 2)

        if use_avi:
            self.encoder = Encoder(x_dim, [self.hidden_dim], z_dim)
            if use_init_encoder:
                if nu_mean_z_init is not None and nu_sd_z_log_init is not None:
                    self.encoder.init_weights(nu_mean_z=nu_mean_z_init, nu_sd_z_log=nu_sd_z_log_init, device=device)
        else:
            self.encoder = None

        if const_z:
            self.nu_mean_z = nn.Parameter(torch.randn(z_dim) if nu_mean_z_init is None else nu_mean_z_init)
            self.nu_sd_z_log = nn.Parameter(torch.randn(z_dim) if nu_sd_z_log_init is None else nu_sd_z_log_init)
        else:
            size = (n_obs, z_dim) if not use_avi else (z_dim,)
            self.nu_mean_z = nn.Parameter(torch.randn(size) if nu_mean_z_init is None else nu_mean_z_init)
            self.nu_sd_z_log = nn.Parameter(torch.randn(size) - 1 if nu_sd_z_log_init is None else nu_sd_z_log_init)

    def variational_z(self, x):
        """
        Computes the variational parameters (mean and log standard deviation) for the latent variable z.

        Returns:
            nu_mean_z (torch.Tensor): Mean of the latent variable z.
            nu_sd_z_log (torch.Tensor): Log standard deviation of the latent variable z.
        """
        if self.use_avi:
            nu_mean_z, nu_sd_z_log = self.encoder(x)
        elif self.const_z:
            nu_mean_z = self.nu_mean_z.repeat((self.n_obs, 1))
            nu_sd_z_log = self.nu_sd_z_log.repeat((self.n_obs, 1))
        else:
            nu_mean_z = self.nu_mean_z
            nu_sd_z_log = self.nu_sd_z_log
        return nu_mean_z, nu_sd_z_log

    def variational_parameters(self, x):
        """
        Returns the variational parameters for both z and theta.

        Returns:
            A tuple containing variational parameters: nu_mean_theta, nu_sd_theta_log, nu_mean_z, nu_sd_z_log.
        """
        nu_mean_z, nu_sd_z_log = self.variational_z(x)
        return self.nu_mean_theta, self.nu_sd_theta_log, nu_mean_z, nu_sd_z_log

    def reparam(self, nu_mean_z, nu_sd_z, nu_mean_theta, nu_sd_theta, mc_samples, n_obs=None):
        """
        Performs the reparameterization trick for both z and theta.
        The n_obs parameter allows for reparameterization of a subset of the data.
        """
        device = nu_mean_z.device
        n_obs = n_obs if n_obs is not None else self.n_obs
        epsilon = torch.randn((mc_samples, n_obs, self.z_dim), device=device)
        z = nu_mean_z + nu_sd_z * epsilon
        epsilon_theta = torch.randn((mc_samples, self.dim_theta), device=device)
        theta = nu_mean_theta + nu_sd_theta * epsilon_theta
        return z, theta

    def compute_elbo(self, x, batch_index=None, batch_size=1000):
        """
        Computes the Evidence Lower Bound (ELBO) for a given input batch.

        Returns:
            Elbo (float): The ELBO value for the input batch.
        """
        nu_mean_z, nu_sd_z_log = self.variational_z(x)
        nu_sd_z = torch.exp(nu_sd_z_log)
        nu_sd_theta = torch.exp(self.nu_sd_theta_log)
        z, theta = self.reparam(nu_mean_z, nu_sd_z, self.nu_mean_theta, nu_sd_theta, self.mc_samples)
        Elbo = 0
        for i in range(self.mc_samples):
            mu = Decoder(theta[i], z[i], self.z_dim, self.like_dim, self.x_dim)
            sigma = torch.ones((batch_size, self.x_dim))
            Elbo += log_joint_gaussian(x, mu, sigma, z[i], theta[i]) - log_q(theta[i], z[i], self.nu_mean_theta, nu_sd_theta, nu_mean_z, nu_sd_z)
        return Elbo / self.mc_samples

    def reconstruction_mse(self, x):
        """
        Computes the mean squared error of the reconstruction using the Bayes estimator which is used for evaluation.

        Returns:
            mse (float): The mean squared error of the reconstruction.
        """
        nu_mean_z, nu_sd_z_log = self.variational_z(x)
        mu = Decoder(self.nu_mean_theta, nu_mean_z, self.z_dim, self.like_dim, self.x_dim)
        return torch.mean((mu - x)**2)

    def reconstruct_and_plot_images(self, data_loader, num_images=5):
        self.eval()
        images, labels = next(iter(data_loader))

        original_images = images[:num_images]
        labels = labels[:num_images]

        if len(original_images.size()) > 2:
            original_images = original_images.view(original_images.size(0), -1)

        original_images = original_images.to(next(self.parameters()).device)

        nu_mean_z, nu_sd_z_log = self.variational_z(original_images)
        nu_sd_z = torch.exp(nu_sd_z_log)

        z, _ = self.reparam(nu_mean_z, nu_sd_z, self.nu_mean_theta, torch.exp(self.nu_sd_theta_log), 1, num_images)
        with torch.no_grad():
            reconstructed_images = Decoder(self.nu_mean_theta, z, self.z_dim, self.like_dim, self.x_dim).view_as(original_images).cpu()

        if len(images.size()) > 2:
            reconstructed_images = reconstructed_images.view(num_images, *images.size()[1:])
            original_images = original_images.view(num_images, *images.size()[1:])

        sns.set(style='white', context='talk', palette='colorblind')  # Set the seaborn style

        fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(15, 8))

        for i in range(num_images):
            # Plot originals with true class labels
            original_image = original_images[i].squeeze().cpu().numpy()
            axes[0, i].imshow(original_image, cmap='gray', aspect='auto')
            axes[0, i].set_title(f'Original: {labels[i].item()}')
            axes[0, i].axis('off')

            # Plot reconstructed with true class labels
            reconstructed_image = reconstructed_images[i].squeeze().numpy()
            axes[1, i].imshow(reconstructed_image, cmap='gray', aspect='auto')
            axes[1, i].set_title(f'Reconstructed: {labels[i].item()}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join("/content/Amortized_Bayes/Results", f"{data_set}_reconstructed.png"))
        plt.show()
```

To optimize computational efficiency, the training and testing processes will be conducted on a subset of 10,000 images.

```python
data_set = args.dataset
config = dataset_configs[data_set]
x_dim = config["x_dim"]

def prepare_first_batch(data_loader, x_dim, device):
    for batch_idx, (x, _) in enumerate(data_loader):
        if batch_idx == 0:
            x = x.view(-1, x_dim)
            x = x.to(device)
            return x
    return None

x = prepare_first_batch(train_loader, x_dim, device)
x_test = prepare_first_batch(test_loader, x_dim, device)
```

Below we introduce an optimizer helper class that that wraps around PyTorch's Adam or AdamW optimizers to provide flexibility in handling weight decay and filtering parameters by their requirement for gradient computation. 

```python
class AdamOptimizer:
    """
    Taken from https://github.com/lucidrains/pytorch-custom-utils/blob/main/pytorch_custom_utils/get_adam_optimizer.py

    A helper optimizer class that wraps around PyTorch's Adam or AdamW optimizers to provide
    flexibility in handling weight decay and filtering parameters by their requirement for gradient computation.

    Parameters:
    - params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
    - lr (float, optional): The learning rate. Default is 1e-4.
    - wd (float, optional): Weight decay. Default is 1e-2. If set to a positive number, enables weight decay.
    - betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.99).
    - eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
    - filter_by_requires_grad (bool, optional): If True, only parameters that require gradients are optimized. Default is False.
    - omit_gammas_and_betas_from_wd (bool, optional): If True, parameters named 'gamma' and 'beta' are excluded from weight decay. Default is True.
    - **kwargs: Additional keyword arguments to be passed to the optimizer.

    The class automatically decides whether to use Adam or AdamW based on the weight decay configuration and
    the setting for omitting 'gamma' and 'beta' parameters from weight decay.
    """
    def __init__(self, params, lr: float = 1e-4, wd: float = 1e-2, betas: Tuple[float, float] = (0.9, 0.99),
                 eps: float = 1e-8, filter_by_requires_grad: bool = False, omit_gammas_and_betas_from_wd: bool = True, **kwargs):
        self.params = params
        self.lr = lr
        self.wd = wd
        self.betas = betas
        self.eps = eps
        self.filter_by_requires_grad = filter_by_requires_grad
        self.omit_gammas_and_betas_from_wd = omit_gammas_and_betas_from_wd
        self.kwargs = kwargs

        self.optimizer = self.get_adam_optimizer()

    def separate_weight_decayable_params(self, params):
        wd_params, no_wd_params = [], []

        for param in params:
            param_list = no_wd_params if param.ndim < 2 else wd_params
            param_list.append(param)

        return wd_params, no_wd_params

    def get_adam_optimizer(self):
        has_weight_decay = self.wd > 0.

        if self.filter_by_requires_grad:
            self.params = [t for t in self.params if t.requires_grad]

        opt_kwargs = dict(
            lr = self.lr,
            betas = self.betas,
            eps = self.eps
        )

        if not has_weight_decay:
            return Adam(self.params, **opt_kwargs)

        opt_kwargs['weight_decay'] = self.wd

        if not self.omit_gammas_and_betas_from_wd:
            return AdamW(self.params, **opt_kwargs)

        wd_params, no_wd_params = self.separate_weight_decayable_params(self.params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

        return AdamW(params, **opt_kwargs)

    def zero_grad(self):
      """Clears the gradients of all optimized parameters."""
      self.optimizer.zero_grad()

    def step(self):
      """Performs a single optimization step."""
      self.optimizer.step()
```

#### Training the Model

We are now ready to start training our model. This code below trains a given model on provided training data for 5,000 epochs on a subset of 10,000 images, with options for printing output, saving mean squared error (MSE) metrics for both training and testing datasets, and configuring learning rate and weight decay for the optimizer. It initializes the training environment by setting random seeds for reproducibility, and configuring the optimizer.

During training, it computes the loss (negative Evidence Lower Bound, ELBO) for each batch, updates the model parameters through backpropagation, and logs the batch loss. Optionally, it calculates and logs the MSE for the training and, if provided, testing datasets after each epoch. The function returns the trained model, arrays of loss and MSE metrics, and the total training time.

```python
def train(seed, model, x, n_epochs, n_obs, batch_size, print_output=False, lr=args.lr, weight_decay=args.weight_decay,
           save_mse=False, save_mse_test=False, x_test=None):
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

    Returns:
    - model (torch.nn.Module): The trained model.
    - loss_saved (numpy.ndarray): Array containing the loss values for each iteration.
    - run_time (float): Total training time.
    - mse_saved (numpy.ndarray): MSE values for the training dataset for each epoch if `save_mse` is True; otherwise, an empty array.
    - mse_saved_test (numpy.ndarray): MSE values for the test dataset for each epoch if `save_mse_test` is True and `x_test` is provided; otherwise, an empty array.
    """

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
        print("Starting training VAE...")

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

        if save_mse:
            mse = model.reconstruction_mse(x).item()
            mse_saved[epoch] = mse

        if save_mse_test and x_test is not None:
            mse_test = model.reconstruction_mse(x_test).item()
            mse_saved_test[epoch] = mse_test

        if epoch % 500 == 0 and print_output:
            print(f"\tEpoch: {epoch} \tLoss: {loss.item()}")
            if save_mse:
                print(f"\tMSE: {mse_saved[epoch]}")

    end_time = time.time()
    run_time = end_time - start_time

    return model, loss_saved, run_time, mse_saved, mse_saved_test
```

To make the training process easier, we will introduce an experiment wrapper class. This class is designed to be a simple wrapper to conduct our variational inference experiments. It supports experiments with different configurations, including Amortized Variational Inference (A-VI) with neural networks, Mean-Field Variational Inference (F-VI), and Constant Variational Inference (Constant-VI). 

The class is structured to handle the entire experimental workflow, which includes initializing models with specific configurations, training these models, running a series of experiments with different settings, and saving the results.

```python
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

        self.output_dir = "/content/Amortized_Bayes/Results"
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

    def display_reconstructed_images(self, data_loader, num_images=5, seed=None):
        """
        Displays reconstructed images using a trained model.

        Parameters:
        - data_loader (DataLoader): A PyTorch DataLoader instance for the dataset.
        - num_images (int): The number of images to reconstruct and display.
        - seed (int, optional): Seed for random number generation to ensure reproducibility. If provided, it will retrain the model with this seed before displaying images.
        """
        if seed is not None:
            model = self.initialize_model(self.nn_widths[3], True)  # Use the largest nn-width (256) with A-VI since this will be best result
            trained_model, _, _, _, _ = self.train_model(model, seed)
        else:
            trained_model = self.model

        trained_model.eval()
        trained_model.reconstruct_and_plot_images(data_loader, num_images=num_images)

    def save_results(self, seed, loss_all, mse_train_all, mse_test_all):
        """
        Saves the aggregated results of the experiments to disk.

        Parameters:
            seed (int): Seed used for the experiments, used in naming the output files.
            loss_all (numpy.ndarray): Array of loss values from all experiments.
            mse_train_all (numpy.ndarray): Array of training MSE values from all experiments.
            mse_test_all (numpy.ndarray): Array of testing MSE values from all experiments.
        """
        np.save(os.path.join(self.output_dir, f"vae_{self.data_set}_loss_{seed}.npy"), loss_all)
        np.save(os.path.join(self.output_dir, f"vae_{self.data_set}_mse_train_{seed}.npy"), mse_train_all)
        np.save(os.path.join(self.output_dir, f"vae_{self.data_set}_mse_test_{seed}.npy"), mse_test_all)
```

#### Running the Experiments

Finally, we are ready to conduct our experiments. In this post, we will see the results for the `MNIST` and `FashionMNIST` datasets. However, the results for other datasets are very similar. We can call our experiment wrapper class to start running the experiements and handle all of the heavy lifting for us.


```python
# Begin experiments
experiment = VariationalInferenceExperiment(x, 
                                            x_test, 
                                            args.z_dim, 
                                            args.like_dim, 
                                            args.epochs, 
                                            args.nn_widths, 
                                            args.lr, 
                                            args.weight_decay, 
                                            args.batch_size, 
                                            args.n_obs, 
                                            args.dataset, 
                                            device)
experiment.run_experiments(args.seed)
```

As we noted earlier, our experiments are inherently stochastic. Therefore, we will run these experiments across several different seeds to confirm our initial results.

```python
# Run experiments across different seeds for robustness
init_seed = 415
for i in range(5):
    seed = init_seed + i
    print("seed: ", seed)
    experiment = VariationalInferenceExperiment(x, 
                                            x_test, 
                                            args.z_dim, 
                                            args.like_dim, 
                                            args.epochs, 
                                            args.nn_widths, 
                                            args.lr, 
                                            args.weight_decay, 
                                            args.batch_size, 
                                            args.n_obs, 
                                            args.dataset, 
                                            device)
    experiment.run_experiments(seed)
```

### Results

**MNIST**

Our results, presented in Figure 3 for the MNIST dataset, examine the effects of different network widths and configurations. After 5,000 epochs, our amortized variational inference (A-VI) achieved comparable ELBO values to fixed variational inference (F-VI) with sufficiently deep networks (k ≥ 64). We also evaluated the mean squared error (MSE) for image reconstruction on both the training and testing sets and noted that A-VI effectively bridged the performance gap here too.

<table>
  <tr>
    <td><img src="./mnist_elbo.png" alt="Image 1" style="width: 100%;"></td>
    <td><img src="./mnist_mse.png" alt="Image 2" style="width: 100%;"></td>
    <td><img src="./mnist_mse_test.png" alt="Image 3" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 3:<em>Results for the MNIST dataset</em>
</p>

Moreover, A-VI proved to be 2 to 3 times faster computationally than F-VI, as seen in Figure 4, underscoring its efficiency in leveraging shared inference computations across data, thus negating the need to estimate unique latent factors $q_n$ for each $z_n$.

<p align="center">
  <img src="./mnist_comp.png" alt="Computation Time MNIST">
  Figure 4:<em> Computational efficiency of A-VI on MNIST</em>
</p>

**FashionMNIST**

Our results for the `FashionMNIST` experiments are presented in Figure 4 and show the same conclusions as the `MNIST` experiments.

<table>
  <tr>
    <td><img src="./fmnist_elbo.png" alt="Image 1" style="width: 100%;"></td>
    <td><img src="./fmnist_mse.png" alt="Image 2" style="width: 100%;"></td>
    <td><img src="./fmnist_mse_test.png" alt="Image 3" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 4:<em>Results for the FashionMNIST dataset</em>
</p>

We also see a similar increase in computational speed on the `FashionMNIST` dataset as shown in Figure 5.

<p align="center">
  <img src="./fmnist_comp.png" alt="Computation Time FashionMNIST">
  Figure 5:<em> Computational efficiency of A-VI on FashionMNIST</em>
</p>

In Figure 6, we present reconstructed images for a sample of five original images from the `MNIST` and `FashionMNIST` datasets. It’s important to note that these reconstructions, produced using a linear neural network, exhibit lower visual quality. This outcome, while noticeable, was not the primary focus of our project. Implementing a convolutional neural network for both the encoder and decoder could significantly enhance the aesthetic quality of these images.

<table>
  <tr>
    <td style="padding-right: 15px;"><img src="./re1.png" alt="Image 1" style="width: 100%;"></td>
    <td style="padding-left: 15px;"><img src="./re2.png" alt="Image 2" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 6:<em>Reconstructed images for MNIST and FashionMNIST</em>
</p>

## Conclusion

In this project, we explored the application of amortized variational inference (A-VI) in Bayesian deep generative models, with a focus on closing the amortization gap between A-VI and traditional variational inference approaches such as factorized variational inference (F-VI) and constant variational inference. We conducted numerical experiments on benchmark imaging datasets, including MNIST, to compare the performance of A-VI with varying neural network architectures against F-VI and constant-VI.

Our findings demonstrated that A-VI, when implemented with sufficiently deep neural networks (hidden dimensions $\geq$ 64), was able to achieve the same evidence lower bound (ELBO) as F-VI after 5,000 epochs of training. Moreover, A-VI closed the gap in terms of reconstruction mean squared error (MSE) on both the training and test datasets. Importantly, A-VI proved to be 2 to 3 times computationally faster than F-VI, highlighting its efficiency in amortizing inference across the data by avoiding the need to estimate a different latent factor for each latent variable.

These results underscore the potential of A-VI in addressing the amortization interpolation problem, demonstrating that a deep encoder-decoder linear neural network with full Bayesian inference over the latent variables can effectively approximate the ideal inference function and close the amortization gap. The successful application of A-VI in this context paves the way for more efficient and scalable Bayesian deep generative models, enabling their deployment in various domains where computational efficiency and accurate inference are crucial.

Future research directions could involve exploring more complex neural network architectures, investigating the impact of different variational approximation families, and extending the application of A-VI to other types of generative models and datasets. Additionally, theoretical work on the conditions under which an ideal inference function is guaranteed to exist and the required richness of the inference function family could further deepen our understanding of the amortization interpolation problem and guide the development of more effective A-VI implementations.

## References

- Michael I. Jordan et al. "An Introduction to Variational Methods for Graphical Models." Machine Learning, vol. 37, pp. 183-233, 1999.

- Christopher M. Bishop, David Spiegelhalter, John Winn. "Vibes: A Variational Inference Engine for Bayesian Networks." Neural Information Processing Systems, 2002.

- Andrew Gelman et al. Bayesian Data Analysis. Chapman & Hall/CRC Texts in Statistical Science, 2013.

- Samuel Gershman, Noah Goodman. "Amortized Inference in Probabilistic Reasoning." Proceedings of the Annual Meeting of the Cognitive Science Society, 2014.

- D. Rezende, S. Mohamed, D. Wierstra. "Stochastic Backpropagation and Approximate Inference in Deep Generative Models." International Conference on Machine Learning, 2014.

- Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization." International Conference on Learning Representations, 2015.

- David M. Blei, Alp Kucukelbir, Jon D. McAuliffe. "Variational Inference: A Review for Statisticians." Journal of the American Statistical Association, vol. 112, 2017.

- Chris Cremer, Xuechen Li, David Duvenaud. "Inference Suboptimality in Variational Autoencoders." International Conference of Machine Learning, 2018.

- Yoon Kim et al. "Semi-Amortized Variational Autoencoders." International Conference on Machine Learning, 2018.

- Rui Shu et al. "Amortized Inference Regularization." Neural Information Processing Systems, 2018.

- Adam Paszke et al. "Pytorch: An Imperative Style, High-Performance Deep Learning Library." Neural Information Processing Systems, 2019.

- Abhinav Agrawal, Justin Domke. "Amortized Variational Inference in Simple Hierarchical Models." Neural Information Processing Systems, 2021.

- Laurent Girin et al. "Dynamical Variational Autoencoders: A Comprehensive Review." Foundations and Trends in Machine Learning, vol. 15, pp. 1-175, 2021.

- Minyoung Kim, Vladmir Pavlovic. "Reducing the Amortization Gap in Variational Autoencoders: A Bayesian Random Function Approach." arXiv:2102.03151, 2021. [URL](https://arxiv.org/abs/2102.03151).

- Diederik Kingma, Max Welling. "Auto-Encoding Variational Bayes." 2022. [URL](https://arxiv.org/abs/1312.6114).

- Ankush Ganguly, Sanjana Jain, Ukrit Watchareeruetai. "Amortized Variational Inference: A Systematic Review." Journal of Artificial Intelligence Research, vol. 78, pp. 167-215, 2023. [URL](https://arxiv.org/abs/2209.10888).

- Ryan Giordano, Martin Ingram, Tamara Broderick. "Black Box Variational Inference with a Deterministic Objective: Faster, More Accurate, and Even More Black Box." arXiv:2304.05527, 2023. [URL](https://arxiv.org/abs/2304.05527).

- Charles C Margossian, Laurence K Saul. "The Shrinkage-Delinkage Trade-off: An Analysis of Factorized Gaussian Approximations for Variational Inference." Uncertainty in Artificial Intelligence, 2023.

- Charles C. Margossian, David M. Blei. "Amortized Variational Inference: When and Why?" 2024. [URL](https://arxiv.org/pdf/2307.11018).



