"""Explicit Adam state used by all three Exp6 methods."""
import torch


def beta1_ema(previous, gradient, beta1):
    return [beta1 * old + (1.0 - beta1) * new
            for old, new in zip(previous, gradient)]


def beta1_bias_correct(moment, beta1, step):
    if step < 1:
        raise ValueError("EMA step must be positive")
    return [value / (1.0 - beta1 ** step) for value in moment]


def beta2_update(previous, source, beta2):
    return {name: beta2 * previous[name] + (1.0 - beta2) * value
            for name, value in source.items()}


def beta2_bias_correct(moment, beta2, step):
    if step < 1:
        raise ValueError("EMA step must be positive")
    scale = 1.0 - beta2 ** step
    return {name: value / scale for name, value in moment.items()}


class FirstMomentState:
    def __init__(self, params, beta1):
        self.beta1 = beta1
        self.m = [torch.zeros_like(param) for param in params]
        self.step = 0

    def update(self, gradients):
        self.m = beta1_ema(self.m, gradients, self.beta1)
        self.step += 1
        return beta1_bias_correct(self.m, self.beta1, self.step)


class SecondMomentState:
    """Per-parameter beta2 EMA; callers choose the observed source."""
    def __init__(self, beta2):
        self.beta2 = beta2
        self.v = None
        self.step = 0

    def update(self, source):
        if self.v is None:
            self.v = {name: torch.zeros_like(value)
                      for name, value in source.items()}
        self.v = beta2_update(self.v, source, self.beta2)
        self.step += 1
        return beta2_bias_correct(self.v, self.beta2, self.step)


class ExplicitAdamState:
    """The standard Adam recurrence without torch.optim.Adam."""
    def __init__(self, params, beta1, beta2, eps):
        self.first = FirstMomentState(params, beta1)
        self.beta2 = beta2
        self.eps = eps
        self.v = [torch.zeros_like(param) for param in params]
        self.step = 0

    def update(self, gradients):
        m_hat = self.first.update(gradients)
        self.v = [self.beta2 * old + (1.0 - self.beta2) * g.square()
                  for old, g in zip(self.v, gradients)]
        self.step += 1
        v_hat = [value / (1.0 - self.beta2 ** self.step) for value in self.v]
        denominator = [value.sqrt().add(self.eps) for value in v_hat]
        direction = [m / d for m, d in zip(m_hat, denominator)]
        return direction, m_hat, v_hat, denominator
