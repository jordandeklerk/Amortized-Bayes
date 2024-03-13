from typing import Tuple
from torch.optim import AdamW, Adam


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