# Monte Carlo Portfolio Simulation

Multi-asset, regime-switching Monte Carlo simulation with fat-tailed returns, risk parity allocation, macro factor decomposition, and stress testing.

## Features

- **8-asset universe** across Equities, Bonds, Commodities, and FX
- **Correlated fat-tailed returns** via Cholesky decomposition with Student-t innovations
- **3-state Markov regime switching** (Bull / Bear / Crisis) with calibrated transition probabilities
- **Risk parity allocation** — equal-risk-contribution optimization across all assets and asset classes
- **Macro factor decomposition** into Growth, Inflation, and Rate drivers using OLS projection
- **Stress testing** against GFC 2008, COVID 2020, Rate Shock, and Stagflation scenarios
- **Comprehensive risk metrics**: VaR, CVaR, Maximum Drawdown, Sharpe, Sortino, Calmar, Skewness, Kurtosis

## Asset Universe

| Asset | Class |
|-------|-------|
| US Equity | Equity |
| International Equity | Equity |
| US Treasury 10Y | Bond |
| TIPS | Bond |
| Gold | Commodity |
| Commodities | Commodity |
| USD/EUR | FX |
| USD/JPY | FX |

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Or run the Jupyter notebook:

```bash
jupyter notebook monte_carlo_simulation.ipynb
```

## Output

Running `python main.py` produces:

- **Console report** — portfolio weights, risk metrics with confidence intervals, stress test results
- **9 charts** saved to `output/`:
  - `simulation_paths.png` — 10,000 wealth paths with percentile bands
  - `return_distribution.png` — histogram with VaR lines + Q-Q plot
  - `regime_timeline.png` — regime fractions over time
  - `risk_parity_allocation.png` — capital weights vs. risk contribution
  - `efficient_frontier.png` — Markowitz frontier with risk parity portfolio
  - `factor_decomposition.png` — variance decomposition by macro factor
  - `stress_test_waterfall.png` — portfolio impact under each scenario
  - `drawdown_distribution.png` — max drawdown histogram
  - `risk_metrics_summary.png` — full metrics table
- **summary.csv** — metrics in tabular format

## Methodology

| Component | Approach |
|-----------|----------|
| Return generation | Cholesky decomposition + Student-t innovations (fat tails) |
| Regime dynamics | 3-state Markov chain with calibrated transition matrix |
| Allocation | Risk parity via SLSQP optimization (equal risk contribution) |
| Factor model | OLS projection onto Growth / Inflation / Rates |
| Risk metrics | Historical VaR/CVaR, drawdown, Sharpe/Sortino/Calmar |
| Stress testing | Deterministic scenario shocks calibrated to historical events |

## Simulation Parameters

- **Simulations**: 10,000
- **Horizon**: 10 years (2,520 trading days)
- **Student-t degrees of freedom**: Bull=8, Bear=5, Crisis=3
- **Risk-free rate**: 4.0%
- **Seed**: 42 (reproducible)

## Project Structure

```
montecarlo/
├── main.py                    # Entry point
├── config.py                  # All parameters and scenario definitions
├── requirements.txt
├── monte_carlo_simulation.ipynb  # Interactive notebook version
├── engine/
│   ├── simulation.py          # Core MC engine (Cholesky + Student-t + regimes)
│   ├── regimes.py             # Markov regime-switching model
│   └── factors.py             # Macro factor decomposition
├── allocation/
│   ├── risk_parity.py         # Risk parity optimizer
│   └── mean_variance.py       # Efficient frontier computation
├── risk/
│   ├── metrics.py             # VaR, CVaR, drawdown, Sharpe, Sortino, Calmar
│   └── stress.py              # Stress testing scenarios
├── visualization/
│   └── charts.py              # Chart generation
└── output/                    # Generated charts and CSV
```

## Requirements

- Python 3.10+
- numpy, scipy, pandas, matplotlib, seaborn
