# Black-Scholes Option Pricer

`Python 3.11+` · `numpy / scipy` · `pytest` · `black + ruff` · `MIT`

A production-quality European option pricing library: closed-form
Black-Scholes-Merton prices, the full set of Greeks, a Newton-Raphson implied
volatility solver, and a Monte Carlo engine that independently validates the
analytics.

> 日本語版は [README.ja.md](README.ja.md) を参照してください。

---

## Why this project (relevance to quant hiring)

Option pricing sits at the intersection of stochastic calculus, numerical
methods, and disciplined software engineering — exactly the skill set a
quant/risk desk screens for. This repository is deliberately small but complete:
it derives the analytics, proves them correct two independent ways (finite-
difference Greeks **and** Monte Carlo convergence), handles the degenerate edge
cases that break naïve implementations (zero maturity, zero volatility, deep
ITM/OTM implied vol), and ships with tests, type hints, linting, and
documentation. It is meant to demonstrate not just that I can write the
Black-Scholes formula, but that I can ship a numerical component someone else
would trust in a risk system.

---

## Theory in one minute

Under the Black-Scholes-Merton model the underlying follows a geometric
Brownian motion under the risk-neutral measure:

```
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

The European call/put prices have the closed form

```
Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
```

where `N` is the standard normal CDF. The **Greeks** are the analytic partial
derivatives of this price (Delta `∂V/∂S`, Gamma `∂²V/∂S²`, Vega `∂V/∂σ`, Theta
`-∂V/∂t`, Rho `∂V/∂r`). **Implied volatility** inverts the price map: given a
market premium, solve for the `sigma` that reproduces it. Because the model
assumes a single flat volatility, the Monte Carlo estimator — which simulates
terminal prices directly from the exact lognormal law and discounts the average
payoff — must converge to the same number, providing an independent check.

---

## Project layout

```
quant-portfolio/
├── src/
│   └── pricing/
│       ├── black_scholes.py   # prices, Greeks, implied vol (vectorised)
│       └── monte_carlo.py     # GBM Monte Carlo + standard error / CI
├── tests/
│   └── test_black_scholes.py  # parity, MC vs analytic, FD Greeks, edge cases
├── notebooks/
│   └── black_scholes_demo.ipynb
├── requirements.txt
└── README.md / README.ja.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/quant-portfolio.git
cd quant-portfolio

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

```python
from src.pricing.black_scholes import bs_price, greeks, implied_volatility, OptionType
from src.pricing.monte_carlo import mc_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# Closed-form price
price = bs_price(S, K, T, r, sigma, OptionType.CALL)
print(f"Call price: {price:.4f}")        # Call price: 10.4506

# Full risk report (price + all Greeks)
report = greeks(S, K, T, r, sigma, OptionType.CALL)
print(report)
# {'price': 10.4506, 'delta': 0.6368, 'gamma': 0.0188,
#  'vega': 37.524, 'theta': -0.0176, 'rho': 53.232}

# Invert a market price back to implied volatility
iv = implied_volatility(price, S, K, T, r, OptionType.CALL)
print(f"Implied vol: {iv:.4f}")          # Implied vol: 0.2000

# Independent Monte Carlo cross-check (reproducible via seed)
mc = mc_price(S, K, T, r, sigma, OptionType.CALL, n_paths=200_000, seed=42)
print(f"MC price: {mc.price:.4f} ± {mc.std_error:.4f} (95% CI {mc.ci_95})")
```

The pricing and Greek functions are **vectorised** — pass NumPy arrays for any
argument to price a whole grid at once:

```python
import numpy as np
spots = np.linspace(80, 120, 5)
bs_price(spots, K, T, r, sigma, OptionType.CALL)   # -> array of 5 prices
```

---

## Running the tests

```bash
pytest -q
```

The suite covers four pillars of correctness:

- **Put-Call parity** — the model-free no-arbitrage identity `C - P = S e^{-qT} - K e^{-rT}`.
- **Monte Carlo vs analytic** — agreement within a few standard errors.
- **Greeks vs finite differences** — analytic derivatives match bumped prices.
- **Edge cases** — zero maturity / zero volatility return intrinsic value (no NaNs),
  negative inputs are rejected.

### Code quality

```bash
black src tests      # format
ruff check src tests # lint
```

The codebase is formatted with **black** and passes **ruff** with no warnings.

---

## Sample results

Run `notebooks/black_scholes_demo.ipynb` to reproduce the figures:

1. **Price vs spot** — call/put premiums against intrinsic-value payoffs.
2. **Greeks vs spot** — Delta, Gamma, Vega, Theta, Rho across moneyness.
3. **Monte Carlo convergence** — the MC estimate and its shrinking 95% band
   converging to the analytic price as paths increase.
4. **Volatility smile** — a stylised skew priced in, then recovered strike-by-
   strike by the implied-vol solver.

---

## License

MIT.
