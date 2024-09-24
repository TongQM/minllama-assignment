from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']

                # Update first and second moments of the gradients
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros(size=p.shape)
                    state['exp_avg_sq'] = torch.zeros(size=p.shape)

                previous_mt = state['exp_avg']
                previous_vt = state['exp_avg_sq']

                state['step'] += 1
                t = state['step']
                mt = beta1 * previous_mt + (1 - beta1) * grad
                vt = beta2 * previous_vt + (1 - beta2) * (grad * grad)
                state['exp_avg'] = mt
                state['exp_avg_sq'] = vt

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                original_alpha = alpha # To record the alpha before bias correction, which could potentially change alpha to alpha_t
                if correct_bias:
                    numerator = pow((1 - pow(beta2, t)), 1/2)
                    denominator = (1 - pow(beta1, t))
                    alpha = alpha *  numerator / denominator
                    eps = numerator * eps # eps_hat as in in Kigma & Ba (2014)

                # Update parameters
                p.data = p.data - alpha * mt / (torch.sqrt(vt) + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data = p.data - original_alpha * weight_decay

        return loss