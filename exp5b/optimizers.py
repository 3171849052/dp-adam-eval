"""Explicit persistent temporal states used by every Exp5b momentum method."""
import torch


def state_bytes(value):
    """Count tensor storage only; strings and diagnostic metadata cost zero."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(state_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(state_bytes(item) for item in value)
    return 0


def beta1_ema(previous, gradient, beta1):
    """The one and only Exp5b first-moment update: beta*m+(1-beta)*g."""
    return [beta1 * old + (1. - beta1) * new for old, new in zip(previous, gradient)]


def beta1_bias_correct(moment, beta1, step):
    if step < 1:
        raise ValueError("EMA step must be positive")
    scale = 1. - beta1 ** step
    return [value / scale for value in moment]


class FirstMomentState:
    """Post-DP first-moment EMA with exact Adam-style bias correction.

    This is deliberately independent of framework optimizer state semantics.
    """
    def __init__(self, params, beta1):
        self.beta1 = beta1
        self.m = [torch.zeros_like(p) for p in params]
        self.step = 0

    def peek(self, privatized_gradient):
        moment = beta1_ema(self.m, privatized_gradient, self.beta1)
        return beta1_bias_correct(moment, self.beta1, self.step + 1)

    def update(self, privatized_gradient):
        self.m = beta1_ema(self.m, privatized_gradient, self.beta1)
        self.step += 1
        return beta1_bias_correct(self.m, self.beta1, self.step)

    def state_dict(self):
        return {"beta1": self.beta1, "step": self.step,
                "m": [value.detach().clone() for value in self.m]}


def beta2_update(previous, q, beta2, delta_t):
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    a = beta2 ** delta_t
    return a * previous + (1. - a) * q


def beta2_diagnostics(previous, q, current, eps):
    """Layer-wise RMS log-distance for the beta2 innovation and EMA move."""
    if previous is None:
        return {name: {"beta2_D_innovation": None, "beta2_D_ema": None}
                for name in q}
    result = {}
    for name in q:
        log_old = torch.log(previous[name].double() + eps)
        log_q = torch.log(q[name].double() + eps)
        log_current = torch.log(current[name].double() + eps)
        innovation = (log_q - log_old).square().mean().sqrt()
        ema = (log_current - log_old).square().mean().sqrt()
        result[name] = {"beta2_D_innovation": float(innovation), "beta2_D_ema": float(ema)}
    return result


class SecondMomentState:
    """Synthetic second moment using elapsed private-step time."""
    def __init__(self, beta2):
        self.beta2 = beta2
        self.v = None
        self.last_refresh_step = None

    def update(self, q, step):
        if self.v is None:
            previous = None
            self.v = {name: value.detach().clone() for name, value in q.items()}
            delta_t = None
        else:
            previous = self.v
            delta_t = step - self.last_refresh_step
            self.v = {name: beta2_update(previous[name], value, self.beta2, delta_t)
                      for name, value in q.items()}
        self.last_refresh_step = step
        return self.v, previous, delta_t


def apply_gradients(params, gradients, lr):
    with torch.no_grad():
        for parameter, gradient in zip(params, gradients):
            parameter.add_(gradient, alpha=-lr)
