# Closing the Amortization Gap in Bayesian Deep Generative Models

Amortized variational inference (A-VI) has emerged as a promising approach to enhance the efficiency of Bayesian deep generative models. In this project, we aim to investigate the effectiveness of A-VI in closing the amortization gap between A-VI and traditional variational inference methods, such as factorized variational inference (F-VI), or mean-field variational inference. We conduct numerical experiments on benchmark imaging datasets to compare the performance of A-VI with varying neural network architectures against F-VI and constant-VI.

Our findings demonstrate that A-VI, when implemented with sufficiently deep neural networks, can achieve the same evidence lower bound (ELBO) and reconstruction mean squared error (MSE) as F-VI while being 2 to 3 times computationally faster. These results highlight the potential of A-VI in addressing the amortization interpolation problem and suggest that a deep encoder-decoder linear neural network with full Bayesian inference over the latent variables can effectively approximate an ideal inference function. This work paves the way for more efficient and scalable Bayesian deep generative models.

## Overview

In the Bayesian paradigm, statistical inference regarding unknown variables is predicated on computations involving posterior probability densities. Due to the often intractable nature of these densities, which typically lack an analytic form, estimation becomes crucial. Classical methods for estimating the posterior distribution in Bayesian inference such as MCMC are known to be computationally expensive at test time as they rely on repeated evaluations of the likelihood function and, therefore, require a new set of likelihood evaluations for each observation. In contrast, Variational Inference (VI) offers a compelling solution by recasting the difficult task of estimating complex posterior densities into a more manageable optimization problem. The essence of VI lies in selecting a parameterized distribution family, $\mathcal{Q}$, and identifying the member that minimizes the Kullback-Leibler (KL) divergence from the posterior

$$
\begin{equation}
q^* = \arg \min _{q \in \mathcal{Q}} \mathrm{KL}(q(\theta, \mathbf{z}) \| p(\theta, \mathbf{z} \mid \mathbf{x})).
\end{equation}
$$

This process enables the approximation of the posterior with $q^*$, thereby delineating the VI objective to entail the selection of an appropriate variational family $\mathcal{Q}$ for optimization. Common practice in VI applications involves the adoption of the factorized, or mean-field, family. This family is characterized by the independence of the variables

```math
\begin{equation}
\mathcal{Q}_{\mathrm{F}}=\left\{q: q(\theta, \mathbf{z})=q_0(\theta) \prod_{n=1}^N q_n\left(z_n\right)\right\},
\end{equation}
```

wherein each latent variable is represented by a distinct factor $q_n$.

Contrary to the VI framework, the amortized family leverages a stochastic inference function to dictate the variational distribution of each latent variable $z_n$, typically instantiated through a neural network, facilitating the parameter mapping for each latent variable's approximating factor $q_n(z_n)$:

```math
\begin{equation}
\mathcal{Q}_{\mathrm{A}}=\left\{q: q(\theta, \mathbf{z})=q_0(\theta) \prod_{n=1}^N q\left(z_n ; f_\phi\left(x_n\right)\right)\right\}.
\end{equation}
```

This paradigm, known as _amortized variational inference_ (A-VI), optimizes the approximation of the posterior and the inference function simultaneously. Therefore, inference on a single observation can be performed efficiently through a single forward pass through the neural network, framing Bayesian inference as a prediction problem: for _any_ observation, the neural network is trained to predict the posterior distribution, or a quantity that allows the network to infer the posterior without any further simulations.

## Key Findings

- A-VI, when implemented with sufficiently deep neural networks, can achieve the same evidence lower bound (ELBO) and reconstruction mean squared error (MSE) as F-VI while being 2 to 3 times computationally faster.
- These results highlight the potential of A-VI in addressing the amortization interpolation problem and suggest that a deep encoder-decoder linear neural network with full Bayesian inference over the latent variables can effectively approximate an ideal inference function.

## Getting Started

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/jordandeklerk/Amortized-Bayes.git
cd Amortized-Bayes
pip install -r requirements.txt
```

Then run the `main.py` script:
```bash
python main.py
```
## Project Structure

```bash
├── experiment.py
├── images
│   ├── fmnist_comp.png
│   ├── fmnist_elbo.png
│   ├── fmnist_mse.png
│   ├── fmnist_mse_test.png
│   ├── index.md
│   ├── mnist_comp.png
│   ├── mnist_elbo.png
│   ├── mnist_mse.png
│   ├── mnist_mse_test.png
│   ├── re1.png
│   ├── re2.png
│   ├── reparm.png
│   ├── reparm4.png
│   ├── vae.png
│   └── variational.png
├── main.py
├── src
│   ├── model
│   │   └── model.py
│   └── utils
│       ├── config.py
│       ├── optimizer.py
│       └── parser.py
└── train.py
```

## Main Results

**MNIST**

Our results, presented in Figure 3 for the MNIST dataset, examine the effects of different network widths and configurations. After 5,000 epochs, our amortized variational inference (A-VI) achieved comparable ELBO values to fixed variational inference (F-VI) with sufficiently deep networks (k ≥ 64). We also evaluated the mean squared error (MSE) for image reconstruction on both the training and testing sets and noted that A-VI effectively bridged the performance gap here too.

<table>
  <tr>
    <td><img src="./images/mnist_elbo.png" alt="Image 1" style="width: 100%;"></td>
    <td><img src="./images/mnist_mse.png" alt="Image 2" style="width: 100%;"></td>
    <td><img src="./images/mnist_mse_test.png" alt="Image 3" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 3:<em> Results for the MNIST dataset</em>
</p>

Moreover, A-VI proved to be 2 to 3 times faster computationally than F-VI, as seen in Figure 4, underscoring its efficiency in leveraging shared inference computations across data, thus negating the need to estimate unique latent factors $q_n$ for each $z_n$.

<p align="center">
  <img src="./images/mnist_comp.png" alt="Computation Time MNIST">
  Figure 4:<em> Computational efficiency of A-VI on MNIST</em>
</p>

**FashionMNIST**

Our results for the `FashionMNIST` experiments are presented in Figure 4 and show the same conclusions as the `MNIST` experiments.

<table>
  <tr>
    <td><img src="./images/fmnist_elbo.png" alt="Image 1" style="width: 100%;"></td>
    <td><img src="./images/fmnist_mse.png" alt="Image 2" style="width: 100%;"></td>
    <td><img src="./images/fmnist_mse_test.png" alt="Image 3" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 4:<em> Results for the FashionMNIST dataset</em>
</p>

We also see a similar increase in computational speed on the `FashionMNIST` dataset as shown in Figure 5.

<p align="center">
  <img src="./images/fmnist_comp.png" alt="Computation Time FashionMNIST">
  Figure 5:<em> Computational efficiency of A-VI on FashionMNIST</em>
</p>

In Figure 6, we present reconstructed images for a sample of five original images from the `MNIST` and `FashionMNIST` datasets. It’s important to note that these reconstructions, produced using a linear neural network, exhibit lower visual quality. This outcome, while noticeable, was not the primary focus of our project. Implementing a convolutional neural network for both the encoder and decoder could significantly enhance the aesthetic quality of these images.

<table>
  <tr>
    <td style="padding-right: 15px;"><img src="./images/re1.png" alt="Image 1" style="width: 100%;"></td>
    <td style="padding-left: 15px;"><img src="./images/re2.png" alt="Image 2" style="width: 100%;"></td>
  </tr>
</table>
<p align="center">
  Figure 6:<em> Reconstructed images for MNIST and FashionMNIST</em>
</p>

