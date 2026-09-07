"""Explicit EMA state and disposable Adam candidates for Exp5."""
import copy

import torch


def beta1_ema(previous, gradient, beta1):
    """One first-moment EMA update, elementwise over parameter tensors."""
    return [beta1 * old + (1. - beta1) * new for old, new in zip(previous, gradient)]


def beta1_bias_correct(moment, beta1, step):
    if step < 1:
        raise ValueError("EMA step must be positive")
    scale = 1. - beta1 ** step
    return [value / scale for value in moment]


def beta2_update(previous, q, beta2, delta_t):
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    a = beta2 ** delta_t
    return a * previous + (1. - a) * q


class FirstMomentState:
    def __init__(self, params, beta1):
        self.beta1 = beta1
        self.m = [torch.zeros_like(p) for p in params]
        self.step = 0

    def peek(self, gradients):
        moment = beta1_ema(self.m, gradients, self.beta1)
        return beta1_bias_correct(moment, self.beta1, self.step + 1)

    def update(self, gradients):
        self.m = beta1_ema(self.m, gradients, self.beta1)
        self.step += 1
        return [value / (1. - self.beta1 ** self.step) for value in self.m]

    def state_dict(self):
        return {"beta1": self.beta1, "step": self.step, "m": [x.detach().clone() for x in self.m]}


class SecondMomentState:
    def __init__(self, beta2):
        self.beta2 = beta2
        self.v = None
        self.last_refresh_step = None

    def update(self, q, step):
        if self.v is None:
            self.v = {name: value.detach().clone() for name, value in q.items()}
            delta_t = None
        else:
            delta_t = step - self.last_refresh_step
            self.v = {name: beta2_update(self.v[name], value, self.beta2, delta_t)
                      for name, value in q.items()}
        self.last_refresh_step = step
        return self.v, delta_t


def adam_optimizer(params, c):
    return torch.optim.Adam(params, lr=c["learning_rate"], betas=(c["beta1"], c["beta2"]),
                            eps=c["adam_eps"], weight_decay=c["weight_decay"])


def adam_candidate(params, optimizer, gradients, lr):
    """Return a one-step Adam direction without changing official state/RNG."""
    shadow_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    shadow = torch.optim.Adam(shadow_params, lr=lr, betas=optimizer.param_groups[0]["betas"],
                              eps=optimizer.param_groups[0]["eps"],
                              weight_decay=optimizer.param_groups[0]["weight_decay"])
    shadow.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    before = [p.detach().clone() for p in shadow_params]
    for p, g in zip(shadow_params, gradients):
        p.grad = g.detach().clone()
    shadow.step()
    return [(old - new.detach()) / lr for old, new in zip(before, shadow_params)]


def apply_gradients(params, gradients, lr):
    with torch.no_grad():
        for p, g in zip(params, gradients):
            p.add_(g, alpha=-lr)

