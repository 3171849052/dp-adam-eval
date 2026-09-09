"""The deployable scalar, one-interval-lag beta controller.

The public observation API accepts only already-DP scalar observations.  In
particular it has no route for clean gradients, oracle beta, or a Gaussian
noise realization.
"""

from dataclasses import dataclass
import math


FALLBACK_REASONS = ("initial", "none", "negative", "nonfinite")


@dataclass
class LayerBetaState:
    beta_train: float = 1.0
    numerator_sum: float = 0.0
    denominator_sum: float = 0.0
    observation_count: int = 0
    current_interval: int = 0
    accepted_update_count: int = 0
    fallback_count: int = 0
    last_beta_raw: float = None
    last_beta_source_interval: int = None
    last_update_accepted: bool = False
    last_fallback_used: bool = False
    last_fallback_reason: str = "initial"


class AdaptiveBetaController:
    """Layer-independent scalar controllers with hold-last-positive updates."""

    def __init__(self, layers, beta_initial=1.0):
        if not math.isfinite(float(beta_initial)) or float(beta_initial) <= 0:
            raise ValueError("beta_initial must be positive and finite")
        self.layers = tuple(layers)
        self.states = {layer: LayerBetaState(beta_train=float(beta_initial)) for layer in self.layers}

    def _state(self, layer):
        if layer not in self.states:
            raise KeyError(layer)
        return self.states[layer]

    def active_beta(self, layer):
        return self._state(layer).beta_train

    def observe(self, layer, noisy_energy, expected_noise_energy, trace_F):
        """Accumulate one DP-safe scalar observation for ``layer``.

        This intentionally computes q from ``noisy_energy - expected_noise_energy``
        and never accepts the clean signal or actual noise as an argument.
        """
        state = self._state(layer)
        state.numerator_sum += float(noisy_energy) - float(expected_noise_energy)
        state.denominator_sum += float(trace_F)
        state.observation_count += 1

    def snapshot(self, layer):
        state = self._state(layer)
        return {
            "beta_train": state.beta_train,
            "numerator_sum": state.numerator_sum,
            "denominator_sum": state.denominator_sum,
            "observation_count": state.observation_count,
            "current_interval": state.current_interval,
            "accepted_update_count": state.accepted_update_count,
            "fallback_count": state.fallback_count,
        }

    def finalize_interval(self, interval_index=None, apply=True):
        """Finalize and reset the current interval, returning auditable rows."""
        rows = {}
        for layer, state in self.states.items():
            previous_beta = state.beta_train
            numerator = state.numerator_sum
            denominator = state.denominator_sum
            n_steps = state.observation_count
            raw = numerator / denominator if denominator != 0 else float("nan")
            finite_positive = math.isfinite(raw) and raw > 0
            if apply and finite_positive:
                next_beta = raw
                accepted = True
                fallback = False
                reason = "none"
                state.beta_train = raw
                state.accepted_update_count += 1
            elif apply:
                next_beta = previous_beta
                accepted = False
                fallback = True
                reason = "negative" if math.isfinite(raw) and raw <= 0 else "nonfinite"
                state.fallback_count += 1
            else:
                next_beta = previous_beta
                accepted = False
                fallback = False
                reason = "none"
            source_interval = interval_index if interval_index is not None else state.current_interval
            rows[layer] = {
                "interval_index": int(source_interval),
                "beta_raw": raw,
                "beta_train_previous": previous_beta,
                "beta_train_next": next_beta,
                "numerator_sum": numerator,
                "denominator_sum": denominator,
                "observation_count": n_steps,
                "accepted": accepted,
                "fallback": fallback,
                "fallback_reason": reason,
                "applied_to_training": bool(apply and state.observation_count > 0),
            }
            state.last_beta_raw = raw
            state.last_beta_source_interval = int(source_interval)
            state.last_update_accepted = accepted
            state.last_fallback_used = fallback
            state.last_fallback_reason = reason
            state.numerator_sum = 0.0
            state.denominator_sum = 0.0
            state.observation_count = 0
            state.current_interval = int(source_interval) + 1
        return rows

    def reset_interval(self):
        """Reset accumulators without changing the active beta."""
        for state in self.states.values():
            state.numerator_sum = 0.0
            state.denominator_sum = 0.0
            state.observation_count = 0

    def finalize_all(self, interval_index=None, apply=True):
        return self.finalize_interval(interval_index=interval_index, apply=apply)

