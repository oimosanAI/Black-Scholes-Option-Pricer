"""Monte Carlo pricing of European options under geometric Brownian motion.

The simulator draws terminal asset prices from the exact GBM distribution and
discounts the average payoff. It exists to *validate* the closed-form
Black-Scholes price in :mod:`src.pricing.black_scholes`: with enough paths the
two should agree to within Monte Carlo standard error.

Reproducibility is guaranteed by seeding a dedicated ``numpy.random.Generator``;
the same seed always yields the same estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pricing.black_scholes import OptionType, _coerce_option_type

# 95% normal confidence interval half-width multiplier, used when reporting the
# Monte Carlo error band.
_CI_Z_95: float = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    """Outcome of a Monte Carlo pricing run.

    Attributes
    ----------
    price
        Discounted mean payoff (the price estimate).
    std_error
        Standard error of the estimate; shrinks as ``1/sqrt(n_paths)``.
    n_paths
        Number of simulated paths.
    ci_95
        95% confidence interval ``(low, high)`` for the price.
    """

    price: float
    std_error: float
    n_paths: int
    ci_95: tuple[float, float]


def simulate_terminal_prices(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    q: float = 0.0,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Sample terminal asset prices ``S_T`` from the exact GBM law.

    Under GBM the terminal price has a closed-form lognormal distribution, so we
    sample it directly in one step rather than discretising the path. This is
    exact (no time-stepping bias) and fast.

    Parameters
    ----------
    n_paths
        Number of samples to draw.
    seed
        Seed for the random generator; fixing it makes the run reproducible.
    antithetic
        If ``True`` use antithetic variates (pair each normal draw ``Z`` with
        ``-Z``) to reduce variance at no extra simulation cost.

    Returns
    -------
    numpy.ndarray
        Array of ``n_paths`` terminal prices.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    if S < 0.0 or T < 0.0 or sigma < 0.0:
        raise ValueError("S, T and sigma must be non-negative.")

    rng = np.random.default_rng(seed)

    if antithetic:
        # Generate half the draws and mirror them; if n_paths is odd we draw one
        # extra independent sample to make up the count exactly.
        half = n_paths // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
        if z.size < n_paths:
            z = np.concatenate([z, rng.standard_normal(n_paths - z.size)])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> MCResult:
    """Price a European option by Monte Carlo simulation under GBM.

    Parameters
    ----------
    n_paths
        Number of simulated terminal prices. More paths shrink the standard
        error as ``1/sqrt(n_paths)``.
    seed
        Random seed for reproducibility.
    antithetic
        Enable antithetic variates for variance reduction.

    Returns
    -------
    MCResult
        Price estimate together with its standard error and 95% CI.
    """
    opt = _coerce_option_type(option_type)
    discount = float(np.exp(-r * T))

    terminal = simulate_terminal_prices(
        S, T, r, sigma, n_paths, q=q, seed=seed, antithetic=antithetic
    )

    if opt is OptionType.CALL:
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)

    discounted = discount * payoff
    price = float(np.mean(discounted))
    # Sample standard error of the mean: std(payoff) / sqrt(n). ddof=1 for the
    # unbiased sample variance.
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    half_width = _CI_Z_95 * std_error
    ci_95 = (price - half_width, price + half_width)

    return MCResult(price=price, std_error=std_error, n_paths=n_paths, ci_95=ci_95)
