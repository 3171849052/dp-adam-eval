"""Small, measurement-only estimators for the ExpV4a gamma experiment."""

import math

import torch


def model_gamma_from_state(state, beta):
    """Return the deployable model gamma and its scalar certificate terms.

    The state is the already-active ExpV3 Fisher state. No tensor in it is
    modified; the calculation only reads the eigenvalues and H.
    """
    lambda_f = (state["lambda_G"].detach().double()[:, None]
                * state["lambda_A"].detach().double()[None, :])
    h = state["H"].detach().double()
    a = float(beta) * lambda_f
    sum_a = float(a.sum())
    sum_a_h2 = float((a * h.square()).sum())
    mean_h2 = float(h.square().mean())
    model_signal_retention = sum_a_h2 / sum_a
    gamma = math.sqrt(sum_a / sum_a_h2)
    return {
        "sum_a": sum_a,
        "sum_a_H2": sum_a_h2,
        "mean_H2": mean_h2,
        "model_signal_retention": model_signal_retention,
        "gamma_model_raw": gamma,
        "model_noise_retention_after_raw": gamma * gamma * mean_h2,
    }


def model_gamma_from_eigenvalues(lambda_a, lambda_g, beta, r):
    """Reconstruct the same model gamma from refresh certificate values."""
    lambda_a = torch.as_tensor(lambda_a, dtype=torch.float64)
    lambda_g = torch.as_tensor(lambda_g, dtype=torch.float64)
    lambda_f = lambda_g[:, None] * lambda_a[None, :]
    scaled = float(beta) * lambda_f
    h = torch.where(scaled + float(r) > 0,
                    scaled / (scaled + float(r)),
                    torch.zeros_like(lambda_f))
    return model_gamma_from_state({
        "lambda_A": lambda_a, "lambda_G": lambda_g, "H": h,
    }, beta)


def ratio_sqrt(numerator, denominator):
    """Return sqrt(numerator / denominator), or None for an invalid ratio."""
    numerator, denominator = float(numerator), float(denominator)
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or numerator <= 0 or denominator <= 0):
        return None
    return math.sqrt(numerator / denominator)


def multiplicative_error(left, right):
    if left is None or right is None:
        return None
    left, right = float(left), float(right)
    if not (math.isfinite(left) and math.isfinite(right) and left > 0 and right > 0):
        return None
    return max(left / right, right / left)


def certificate_holds(gamma_model, mean_h2, tolerance=1e-8):
    gamma_model, mean_h2 = float(gamma_model), float(mean_h2)
    return (math.isfinite(gamma_model) and math.isfinite(mean_h2)
            and gamma_model >= 1.0 - tolerance
            and gamma_model * gamma_model * mean_h2 <= 1.0 + tolerance)
