from __future__ import annotations

from typing import Dict

import numpy as np


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.size == 0:
        return 1.0
    y = labels.astype(np.float64).reshape(-1)
    z = logits.astype(np.float64).reshape(-1)

    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 5.0, 46):
        p = 1.0 / (1.0 + np.exp(-np.clip(z / t, -40.0, 40.0)))
        nll = -np.mean(y * np.log(np.clip(p, 1e-9, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-9, 1.0)))
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(t)
    return best_t


def fit_sigma_scale(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    sig = np.clip(sigma.astype(np.float64).reshape(-1), 1e-6, None)
    z = (y.astype(np.float64).reshape(-1) - mu.astype(np.float64).reshape(-1)) / sig
    var = float(np.mean(z ** 2))
    if var <= 1e-9:
        return 1.0
    return float(np.sqrt(var))


def build_calibration_bundle(
    *,
    direction_logit: np.ndarray | None,
    y_net: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["sigma_scale"] = fit_sigma_scale(mu=mu, sigma=sigma, y=y_net)
    if direction_logit is not None:
        labels = (y_net >= 0.0).astype(np.float64)
        out["direction_temperature"] = fit_temperature_scaling(direction_logit, labels)
    else:
        out["direction_temperature"] = 1.0
    return out
