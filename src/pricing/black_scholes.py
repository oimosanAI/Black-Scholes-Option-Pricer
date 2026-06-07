"""Black-Scholes analytics for European options.

This module provides the closed-form Black-Scholes-Merton price for European
calls and puts, the full set of first/second-order Greeks, and an implied
volatility solver based on Newton-Raphson with a bisection fallback.

All functions are vectorised: scalar inputs return floats, while array inputs
(broadcastable NumPy arrays) return arrays. The pricing model assumes a single
constant volatility, a constant risk-free rate, and a continuous dividend yield.

Notation
--------
S : spot price of the underlying
K : strike price
T : time to expiry in years
r : continuously-compounded risk-free rate
sigma : annualised volatility of log returns
q : continuous dividend yield (default 0.0)
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from scipy.stats import norm

# A NumPy-friendly numeric type: a plain Python float or any array-like.
Numeric = float | np.ndarray

# Newton-Raphson defaults for the implied-volatility solver. These trade a few
# extra iterations for robustness across the strikes/maturities seen in practice.
_IV_MAX_ITER: int = 100
_IV_TOLERANCE: float = 1e-8
_IV_INITIAL_GUESS: float = 0.2  # 20% vol is a reasonable equity starting point.
_IV_VOL_LOWER: float = 1e-6
_IV_VOL_UPPER: float = 10.0  # 1000% vol upper bound for the bracketing fallback.

# Below this vega the Newton step is numerically unstable (deep ITM/OTM), so we
# switch to a bracketed bisection instead of dividing by a near-zero derivative.
_MIN_VEGA: float = 1e-8

# Days per calendar year, used to express Theta as a *per-day* sensitivity, which
# is how desks usually quote it.
_DAYS_PER_YEAR: float = 365.0


class OptionType(StrEnum):
    """European option flavour."""

    CALL = "call"
    PUT = "put"


def _validate_inputs(
    S: Numeric, K: Numeric, T: Numeric, sigma: Numeric, r: Numeric
) -> None:
    """Reject economically meaningless inputs early.

    We validate at the boundary so downstream math (logs, square roots) never
    receives values that would silently produce NaNs. Volatility and time of
    exactly zero are *allowed* here because they are legitimate edge cases that
    the pricing functions handle explicitly via the intrinsic-value limit.
    """
    if np.any(np.asarray(S) < 0.0):
        raise ValueError("Spot price S must be non-negative.")
    if np.any(np.asarray(K) < 0.0):
        raise ValueError("Strike price K must be non-negative.")
    if np.any(np.asarray(T) < 0.0):
        raise ValueError("Time to expiry T must be non-negative.")
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Volatility sigma must be non-negative.")
    if np.any(~np.isfinite(np.asarray(r, dtype=float))):
        raise ValueError("Risk-free rate r must be finite.")


def _coerce_option_type(option_type: OptionType | str) -> OptionType:
    """Accept either an :class:`OptionType` or a case-insensitive string."""
    if isinstance(option_type, OptionType):
        return option_type
    try:
        return OptionType(str(option_type).lower())
    except ValueError as exc:
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}."
        ) from exc


def _d1_d2(
    S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric, q: Numeric
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms.

    Where ``sigma * sqrt(T)`` is zero (zero vol or zero time) ``d1``/``d2``
    are undefined; we return +/- inf so that the calling price/Greek formulas
    collapse to the correct intrinsic-value limits via ``norm.cdf``.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    vol_sqrt_t = sigma * np.sqrt(T)
    # Guard against division by zero; we overwrite these entries below.
    safe_denom = np.where(vol_sqrt_t > 0.0, vol_sqrt_t, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / safe_denom
        d2 = d1 - vol_sqrt_t

    # When there is no diffusion, the option value is purely intrinsic. Push d1/d2
    # to +/- inf based on moneyness so N(d1), N(d2) become 1 or 0 accordingly.
    degenerate = vol_sqrt_t <= 0.0
    if np.any(degenerate):
        sign = np.sign(np.asarray(S, dtype=float) - np.asarray(K, dtype=float))
        # sign == 0 (exactly at-the-money) maps to 0 -> N(0)=0.5, a sensible limit.
        inf_fill = np.where(sign >= 0, np.inf, -np.inf)
        d1 = np.where(degenerate, inf_fill, d1)
        d2 = np.where(degenerate, inf_fill, d2)

    return d1, d2


def bs_price(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    q: Numeric = 0.0,
) -> Numeric:
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S, K, T, r, sigma, q
        Spot, strike, time to expiry (years), risk-free rate, volatility and
        continuous dividend yield. All are broadcastable.
    option_type
        ``OptionType.CALL``/``OptionType.PUT`` or the strings ``"call"``/``"put"``.

    Returns
    -------
    float or numpy.ndarray
        The option premium. Scalar inputs yield a Python ``float``.

    Notes
    -----
    At ``T == 0`` or ``sigma == 0`` the formula returns the discounted intrinsic
    value, which is the correct limit of the Black-Scholes price.
    """
    _validate_inputs(S, K, T, sigma, r)
    opt = _coerce_option_type(option_type)

    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-np.asarray(r, dtype=float) * T_arr)
    carry = np.exp(-np.asarray(q, dtype=float) * T_arr)

    if opt is OptionType.CALL:
        price = S_arr * carry * norm.cdf(d1) - K_arr * discount * norm.cdf(d2)
    else:
        price = K_arr * discount * norm.cdf(-d2) - S_arr * carry * norm.cdf(-d1)

    return _to_scalar(price)


def delta(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    q: Numeric = 0.0,
) -> Numeric:
    """Delta: sensitivity of price to a unit change in spot (``dV/dS``)."""
    _validate_inputs(S, K, T, sigma, r)
    opt = _coerce_option_type(option_type)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    carry = np.exp(-np.asarray(q, dtype=float) * np.asarray(T, dtype=float))

    if opt is OptionType.CALL:
        result = carry * norm.cdf(d1)
    else:
        result = carry * (norm.cdf(d1) - 1.0)
    return _to_scalar(result)


def gamma(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    q: Numeric = 0.0,
) -> Numeric:
    """Gamma: rate of change of Delta with spot (``d2V/dS2``).

    Gamma is identical for calls and puts, so no ``option_type`` is required.
    """
    _validate_inputs(S, K, T, sigma, r)

    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    carry = np.exp(-np.asarray(q, dtype=float) * T_arr)
    denom = S_arr * sigma_arr * np.sqrt(T_arr)

    # Gamma is zero in the degenerate (no-diffusion) limit; avoid 0/0 -> NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0.0, carry * norm.pdf(d1) / denom, 0.0)
    return _to_scalar(result)


def vega(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    q: Numeric = 0.0,
) -> Numeric:
    """Vega: sensitivity of price to a unit (1.0 = 100%) change in volatility.

    Vega is identical for calls and puts. To express it per 1% vol move, divide
    the returned value by 100.
    """
    _validate_inputs(S, K, T, sigma, r)

    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    carry = np.exp(-np.asarray(q, dtype=float) * T_arr)
    result = S_arr * carry * norm.pdf(d1) * np.sqrt(T_arr)
    return _to_scalar(result)


def theta(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    q: Numeric = 0.0,
    per_day: bool = True,
) -> Numeric:
    """Theta: sensitivity of price to the passage of time.

    Parameters
    ----------
    per_day
        If ``True`` (default) return the per-calendar-day decay (annual Theta
        divided by 365), matching desk convention. If ``False`` return the
        annualised value.
    """
    _validate_inputs(S, K, T, sigma, r)
    opt = _coerce_option_type(option_type)

    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-r_arr * T_arr)
    carry = np.exp(-q_arr * T_arr)

    # The decay term is undefined when there is no diffusion; set it to 0 there.
    sqrt_t = np.sqrt(T_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(
            sqrt_t > 0.0,
            -(S_arr * carry * norm.pdf(d1) * sigma_arr) / (2.0 * sqrt_t),
            0.0,
        )

    if opt is OptionType.CALL:
        annual = (
            decay
            + q_arr * S_arr * carry * norm.cdf(d1)
            - r_arr * K_arr * discount * norm.cdf(d2)
        )
    else:
        annual = (
            decay
            - q_arr * S_arr * carry * norm.cdf(-d1)
            + r_arr * K_arr * discount * norm.cdf(-d2)
        )

    result = annual / _DAYS_PER_YEAR if per_day else annual
    return _to_scalar(result)


def rho(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    q: Numeric = 0.0,
) -> Numeric:
    """Rho: sensitivity of price to a unit (1.0 = 100%) change in the rate.

    To express it per 1% rate move, divide the returned value by 100.
    """
    _validate_inputs(S, K, T, sigma, r)
    opt = _coerce_option_type(option_type)

    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    discount = np.exp(-np.asarray(r, dtype=float) * T_arr)

    _, d2 = _d1_d2(S, K, T, r, sigma, q)

    if opt is OptionType.CALL:
        result = K_arr * T_arr * discount * norm.cdf(d2)
    else:
        result = -K_arr * T_arr * discount * norm.cdf(-d2)
    return _to_scalar(result)


def greeks(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    q: Numeric = 0.0,
    theta_per_day: bool = True,
) -> dict[str, Numeric]:
    """Convenience wrapper returning all Greeks (and price) in one dict.

    The single call avoids recomputing ``d1``/``d2`` in the caller and keeps the
    risk report assembled in one place.
    """
    return {
        "price": bs_price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q, per_day=theta_per_day),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType | str = OptionType.CALL,
    q: float = 0.0,
    initial_guess: float = _IV_INITIAL_GUESS,
    max_iter: int = _IV_MAX_ITER,
    tolerance: float = _IV_TOLERANCE,
) -> float:
    """Recover the volatility implied by an observed option ``price``.

    Uses Newton-Raphson (fast quadratic convergence) and falls back to bisection
    whenever Vega is too small for a stable Newton step, e.g. for deep in/out-of
    the-money options. This is a scalar solver by design: implied vol is computed
    per quote.

    Parameters
    ----------
    price
        Observed market premium to invert.
    initial_guess
        Starting volatility for Newton-Raphson.
    max_iter, tolerance
        Iteration budget and absolute price-error convergence threshold.

    Returns
    -------
    float
        The implied volatility.

    Raises
    ------
    ValueError
        If ``price`` violates the no-arbitrage bounds, or the solver fails to
        converge within ``max_iter`` iterations.
    """
    opt = _coerce_option_type(option_type)
    if price < 0.0:
        raise ValueError("Observed option price must be non-negative.")
    if T <= 0.0:
        raise ValueError("Implied volatility requires positive time to expiry.")

    # No-arbitrage bounds. Outside them no real volatility reproduces the price,
    # so we fail fast rather than let the solver diverge.
    discount = np.exp(-r * T)
    carry = np.exp(-q * T)
    if opt is OptionType.CALL:
        lower_bound = max(S * carry - K * discount, 0.0)
        upper_bound = S * carry
    else:
        lower_bound = max(K * discount - S * carry, 0.0)
        upper_bound = K * discount
    # Allow a tiny tolerance for floating-point noise at the boundaries.
    if price < lower_bound - tolerance or price > upper_bound + tolerance:
        raise ValueError(
            f"Price {price} is outside the no-arbitrage bounds "
            f"[{lower_bound:.6f}, {upper_bound:.6f}]."
        )

    sigma = float(initial_guess)
    for _ in range(max_iter):
        model_price = float(bs_price(S, K, T, r, sigma, opt, q))
        diff = model_price - price
        if abs(diff) < tolerance:
            return sigma

        v = float(vega(S, K, T, r, sigma, q))
        if v < _MIN_VEGA:
            # Vega too small for a reliable Newton step: hand off to bisection.
            return _implied_vol_bisection(
                price, S, K, T, r, opt, q, max_iter, tolerance
            )

        sigma -= diff / v
        # Keep the iterate inside sane bounds so a bad step cannot escape to
        # negative or absurdly large vols.
        if sigma <= _IV_VOL_LOWER or sigma >= _IV_VOL_UPPER:
            return _implied_vol_bisection(
                price, S, K, T, r, opt, q, max_iter, tolerance
            )

    raise ValueError(
        f"Implied volatility did not converge within {max_iter} iterations."
    )


def _implied_vol_bisection(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float,
    max_iter: int,
    tolerance: float,
) -> float:
    """Robust bracketed fallback for the implied-vol solver.

    Bisection cannot diverge once a sign change is bracketed, so it guarantees a
    result when Newton-Raphson stalls (flat Vega regions).
    """
    low, high = _IV_VOL_LOWER, _IV_VOL_UPPER
    price_low = float(bs_price(S, K, T, r, low, option_type, q)) - price
    price_high = float(bs_price(S, K, T, r, high, option_type, q)) - price

    if price_low * price_high > 0.0:
        raise ValueError(
            "Implied volatility could not be bracketed within "
            f"[{_IV_VOL_LOWER}, {_IV_VOL_UPPER}]."
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        diff = float(bs_price(S, K, T, r, mid, option_type, q)) - price
        if abs(diff) < tolerance:
            return mid
        if diff * price_low < 0.0:
            high = mid
        else:
            low = mid
            price_low = diff

    return 0.5 * (low + high)


def _to_scalar(value: np.ndarray) -> Numeric:
    """Return a Python float for 0-d results, otherwise the array unchanged.

    This keeps the ergonomic ``float`` return for scalar calls while preserving
    full vectorisation for array inputs.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr
