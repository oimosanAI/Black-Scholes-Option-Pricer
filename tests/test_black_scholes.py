"""Test suite for the Black-Scholes pricer and Monte Carlo validator.

The tests cover four pillars expected of a production pricing library:

1. Put-Call parity (a model-free no-arbitrage identity).
2. Agreement between the closed form and Monte Carlo, within MC error.
3. Greeks matching finite-difference bumps of the price (the analytic
   derivatives are correct).
4. Degenerate edge cases (zero maturity, zero volatility) returning intrinsic
   value rather than NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pricing.black_scholes import (
    OptionType,
    bs_price,
    delta,
    gamma,
    greeks,
    implied_volatility,
    rho,
    theta,
    vega,
)
from src.pricing.monte_carlo import mc_price

# A representative, well-conditioned parameter set reused across tests.
S0 = 100.0
K0 = 100.0
T0 = 1.0
R0 = 0.05
SIGMA0 = 0.2
Q0 = 0.0

# Step sizes for finite-difference Greek checks. Chosen small enough for
# accuracy but large enough to avoid floating-point cancellation.
_BUMP_S = 1e-4
_BUMP_SIGMA = 1e-5
_BUMP_R = 1e-6
_BUMP_T = 1e-6


# --------------------------------------------------------------------------- #
# Reference values & basic sanity
# --------------------------------------------------------------------------- #
def test_known_reference_price() -> None:
    """ATM 1Y call at 20% vol, 5% rate matches the textbook value ~10.4506."""
    price = bs_price(S0, K0, T0, R0, SIGMA0, OptionType.CALL)
    assert price == pytest.approx(10.450583572185565, rel=1e-9)


def test_string_and_enum_option_type_agree() -> None:
    """The string API and the enum API must produce identical prices."""
    assert bs_price(S0, K0, T0, R0, SIGMA0, "call") == bs_price(
        S0, K0, T0, R0, SIGMA0, OptionType.CALL
    )
    assert bs_price(S0, K0, T0, R0, SIGMA0, "put") == bs_price(
        S0, K0, T0, R0, SIGMA0, OptionType.PUT
    )


# --------------------------------------------------------------------------- #
# 1. Put-Call parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
def test_put_call_parity(S: float, sigma: float) -> None:
    """C - P == S*e^{-qT} - K*e^{-rT} for any strike/vol (no-arbitrage)."""
    call = bs_price(S, K0, T0, R0, sigma, OptionType.CALL, q=Q0)
    put = bs_price(S, K0, T0, R0, sigma, OptionType.PUT, q=Q0)
    parity_rhs = S * math.exp(-Q0 * T0) - K0 * math.exp(-R0 * T0)
    assert (call - put) == pytest.approx(parity_rhs, abs=1e-9)


def test_put_call_parity_with_dividends() -> None:
    """Parity must also hold with a non-zero continuous dividend yield."""
    q = 0.03
    call = bs_price(S0, K0, T0, R0, SIGMA0, OptionType.CALL, q=q)
    put = bs_price(S0, K0, T0, R0, SIGMA0, OptionType.PUT, q=q)
    parity_rhs = S0 * math.exp(-q * T0) - K0 * math.exp(-R0 * T0)
    assert (call - put) == pytest.approx(parity_rhs, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Analytic vs Monte Carlo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_monte_carlo_matches_analytic(option_type: OptionType) -> None:
    """MC price agrees with the closed form within ~4 standard errors."""
    analytic = bs_price(S0, K0, T0, R0, SIGMA0, option_type)
    result = mc_price(S0, K0, T0, R0, SIGMA0, option_type, n_paths=200_000, seed=42)
    # 4 SE is a ~99.99% band: tight enough to catch bugs, loose enough to avoid
    # flakiness given the fixed seed.
    assert abs(result.price - analytic) < 4.0 * result.std_error


def test_monte_carlo_is_reproducible() -> None:
    """Identical seeds must yield identical estimates."""
    a = mc_price(S0, K0, T0, R0, SIGMA0, "call", n_paths=10_000, seed=7)
    b = mc_price(S0, K0, T0, R0, SIGMA0, "call", n_paths=10_000, seed=7)
    assert a.price == b.price
    assert a.std_error == b.std_error


def test_monte_carlo_error_shrinks_with_paths() -> None:
    """Standard error should fall roughly as 1/sqrt(n_paths)."""
    small = mc_price(S0, K0, T0, R0, SIGMA0, "call", n_paths=10_000, seed=1)
    large = mc_price(S0, K0, T0, R0, SIGMA0, "call", n_paths=160_000, seed=1)
    # 16x paths -> ~4x smaller SE. Allow generous slack for sampling noise.
    assert large.std_error < small.std_error / 2.5


# --------------------------------------------------------------------------- #
# 3. Greeks vs finite differences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_delta_matches_finite_difference(option_type: OptionType) -> None:
    """Analytic Delta == central-difference of price w.r.t. spot."""
    up = bs_price(S0 + _BUMP_S, K0, T0, R0, SIGMA0, option_type)
    down = bs_price(S0 - _BUMP_S, K0, T0, R0, SIGMA0, option_type)
    fd = (up - down) / (2.0 * _BUMP_S)
    assert delta(S0, K0, T0, R0, SIGMA0, option_type) == pytest.approx(fd, abs=1e-6)


def test_gamma_matches_finite_difference() -> None:
    """Analytic Gamma == second central-difference of price w.r.t. spot."""
    up = bs_price(S0 + _BUMP_S, K0, T0, R0, SIGMA0, "call")
    mid = bs_price(S0, K0, T0, R0, SIGMA0, "call")
    down = bs_price(S0 - _BUMP_S, K0, T0, R0, SIGMA0, "call")
    fd = (up - 2.0 * mid + down) / (_BUMP_S**2)
    assert gamma(S0, K0, T0, R0, SIGMA0) == pytest.approx(fd, abs=1e-3)


def test_vega_matches_finite_difference() -> None:
    """Analytic Vega == central-difference of price w.r.t. volatility."""
    up = bs_price(S0, K0, T0, R0, SIGMA0 + _BUMP_SIGMA, "call")
    down = bs_price(S0, K0, T0, R0, SIGMA0 - _BUMP_SIGMA, "call")
    fd = (up - down) / (2.0 * _BUMP_SIGMA)
    assert vega(S0, K0, T0, R0, SIGMA0) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_rho_matches_finite_difference(option_type: OptionType) -> None:
    """Analytic Rho == central-difference of price w.r.t. the rate."""
    up = bs_price(S0, K0, T0, R0 + _BUMP_R, SIGMA0, option_type)
    down = bs_price(S0, K0, T0, R0 - _BUMP_R, SIGMA0, option_type)
    fd = (up - down) / (2.0 * _BUMP_R)
    assert rho(S0, K0, T0, R0, SIGMA0, option_type) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_theta_matches_finite_difference(option_type: OptionType) -> None:
    """Annual Theta == -d(price)/dt via a central difference in maturity.

    Theta is the negative derivative w.r.t. calendar time, i.e. the derivative
    w.r.t. *decreasing* T. We compare against the annualised value.
    """
    up = bs_price(S0, K0, T0 + _BUMP_T, R0, SIGMA0, option_type)
    down = bs_price(S0, K0, T0 - _BUMP_T, R0, SIGMA0, option_type)
    fd = -(up - down) / (2.0 * _BUMP_T)
    analytic = theta(S0, K0, T0, R0, SIGMA0, option_type, per_day=False)
    assert analytic == pytest.approx(fd, abs=1e-3)


def test_greeks_dict_consistency() -> None:
    """The aggregate helper must equal the individual Greek functions."""
    bundle = greeks(S0, K0, T0, R0, SIGMA0, "call")
    assert bundle["price"] == bs_price(S0, K0, T0, R0, SIGMA0, "call")
    assert bundle["delta"] == delta(S0, K0, T0, R0, SIGMA0, "call")
    assert bundle["gamma"] == gamma(S0, K0, T0, R0, SIGMA0)
    assert bundle["vega"] == vega(S0, K0, T0, R0, SIGMA0)
    assert bundle["rho"] == rho(S0, K0, T0, R0, SIGMA0, "call")


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #
# Minimum Vega for implied vol to be well-determined. Below this the price is
# insensitive to vol (deep ITM/OTM at low vol), so inversion is ill-posed and a
# wide band of vols all reproduce the quote within tolerance.
_IV_VEGA_FLOOR = 1e-2


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("true_sigma", [0.05, 0.2, 0.6, 1.2])
@pytest.mark.parametrize("S", [70.0, 100.0, 130.0])
def test_implied_vol_round_trip(
    option_type: OptionType, true_sigma: float, S: float
) -> None:
    """Pricing then inverting must recover the input volatility.

    Deep ITM/OTM combined with very low vol leaves almost no time value, so the
    price barely depends on vol and implied vol is mathematically undetermined;
    we skip those corners by checking Vega.
    """
    if vega(S, K0, T0, R0, true_sigma) < _IV_VEGA_FLOOR:
        pytest.skip("Vega too small for implied vol to be well-determined.")
    price = bs_price(S, K0, T0, R0, true_sigma, option_type)
    recovered = implied_volatility(price, S, K0, T0, R0, option_type)
    assert recovered == pytest.approx(true_sigma, abs=1e-5)


def test_implied_vol_rejects_arbitrage_price() -> None:
    """A price above the no-arbitrage upper bound must raise."""
    with pytest.raises(ValueError):
        # A call can never be worth more than the (carry-adjusted) spot.
        implied_volatility(S0 + 1.0, S0, K0, T0, R0, "call")


# --------------------------------------------------------------------------- #
# 4. Edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_zero_maturity_returns_intrinsic(option_type: OptionType) -> None:
    """At T=0 the price collapses to intrinsic value."""
    itm_spot = 120.0 if option_type is OptionType.CALL else 80.0
    price = bs_price(itm_spot, K0, 0.0, R0, SIGMA0, option_type)
    intrinsic = max(
        (itm_spot - K0) if option_type is OptionType.CALL else (K0 - itm_spot),
        0.0,
    )
    assert price == pytest.approx(intrinsic, abs=1e-12)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_zero_volatility_returns_discounted_intrinsic(
    option_type: OptionType,
) -> None:
    """At sigma=0 the price is the discounted forward intrinsic value."""
    price = bs_price(120.0, K0, T0, R0, 0.0, option_type)
    forward = 120.0 * math.exp(R0 * T0)  # deterministic terminal value
    discounted_intrinsic = math.exp(-R0 * T0) * max(
        (forward - K0) if option_type is OptionType.CALL else (K0 - forward),
        0.0,
    )
    assert price == pytest.approx(discounted_intrinsic, abs=1e-9)


def test_zero_maturity_greeks_are_finite() -> None:
    """Degenerate inputs must not produce NaN/inf Greeks."""
    g = greeks(100.0, K0, 0.0, R0, 0.0, "call")
    for name, value in g.items():
        assert np.isfinite(value), f"{name} is not finite"


def test_negative_inputs_raise() -> None:
    """Negative spot, strike, maturity or vol must be rejected."""
    with pytest.raises(ValueError):
        bs_price(-1.0, K0, T0, R0, SIGMA0, "call")
    with pytest.raises(ValueError):
        bs_price(S0, -1.0, T0, R0, SIGMA0, "call")
    with pytest.raises(ValueError):
        bs_price(S0, K0, -1.0, R0, SIGMA0, "call")
    with pytest.raises(ValueError):
        bs_price(S0, K0, T0, R0, -0.2, "call")


def test_invalid_option_type_raises() -> None:
    """An unknown option flavour must raise a clear error."""
    with pytest.raises(ValueError):
        bs_price(S0, K0, T0, R0, SIGMA0, "straddle")


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #
def test_vectorised_pricing_matches_scalar() -> None:
    """Array inputs must equal element-wise scalar calls."""
    spots = np.array([80.0, 100.0, 120.0])
    vec = bs_price(spots, K0, T0, R0, SIGMA0, "call")
    scalar = np.array([bs_price(float(s), K0, T0, R0, SIGMA0, "call") for s in spots])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)
