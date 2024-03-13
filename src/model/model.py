import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def reparam(self, nu_mean_z, nu_sd_z, nu_mean_theta, nu_sd_theta, mc_samples):
        """
        Performs the reparameterization trick for both z and theta.

        Returns:
            z (torch.Tensor): Sampled latent variables.
            theta (torch.Tensor): Sampled likelihood parameters.
        """
        device = nu_mean_z.device
        epsilon = torch.randn((mc_samples, self.n_obs, self.z_dim), device=device)
        z = nu_mean_z + nu_sd_z * epsilon
        epsilon_theta = torch.randn((mc_samples, self.dim_theta), device=device)
        theta = nu_mean_theta + nu_sd_theta * epsilon_theta
        return z, theta

    def compute_elbo(self, x, batch_index, batch_size=1000):
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