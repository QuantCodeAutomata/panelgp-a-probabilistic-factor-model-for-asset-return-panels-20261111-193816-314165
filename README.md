# PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Replication of selected experiments from "PanelGP: A Probabilistic Factor
Model for Asset-Return Panels" (Ruglioni and Filipovic, 2026).

PanelGP models each stock's excess return as a bilinear product of a
characteristic-driven loading map and a context-driven factor map, both
Gaussian processes, with stock- and date-specific innovations and
observation noise. Inference uses a sparse variational approximation with a
dense, whitened posterior over inducing loadings and realized factors
(Sections 3-4). This repository reproduces:

1. **Experiment 1** -- simulation recovery of the smooth loading/factor maps
   and the three innovation/noise variances (Section 5.1).
2. **Experiment 2** -- fixed-training comparison of four PanelGP kernel pairs
   against two IPCA benchmarks on a stock-return panel (Section 6).
3. **Experiment 3** -- effect of the number of factors H on predictive
   performance (Section 6.3).
4. **Experiment 4** -- annual re-estimation versus fixed training, and
   trading-cost analysis (Sections 6.2, 6.7).
5. **Experiment 5** -- log-score gain of the non-Gaussian bilinear predictive
   density over a Gaussian with the same first two moments (Section 6.8 /
   Appendix B.3).
6. **Experiment 6** -- decomposition and calibration of predicted variances
   (Section 6.8, Equation 27).

See `results/RESULTS.md` for the numbers produced by the latest run next to
the paper's reported values, with a verdict for each metric, and for the full
list of deviations from the paper and of results not reproduced.

## Repository layout

```
src/    PanelGP model, kernels, IPCA benchmark, metrics, training loop,
        prediction/variance decomposition, feature construction, config
exp/    One script per experiment, run as `python -m exp.<name>`
data/   data_loader.py (QuantX Dataloader OHLCV) and generate_synthetic.py
        (Experiment 1's data-generating process)
tests/  pytest unit tests of src/ and experiment-level checks
results/ All generated tables, plots and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Experiments 2-6 use real daily bars loaded through the QuantX Dataloader REST
API (`POST /fetch-ohlcvs`); see `data/data_loader.py` and
`results/RESULTS.md`, "Deviations from the paper", for why the paper's own
WRDS/CRSP characteristics panel and point-in-time S&P 500 membership could
not be used and what was substituted.

## Running

```bash
python -m pytest -m "not market_data"      # unit tests (no network)
python -m pytest                            # include the market-data tests
                                             # (needs DATALOADER_API_KEY/MASSIVE_API_KEY)

python -m exp.exp1_simulation_recovery
python -m exp.exp2_fixed_kernel_comparison
python -m exp.exp3_factor_dimension_sweep
python -m exp.exp4_rolling_and_trading_costs
python -m exp.exp5_bilinear_higher_moments
python -m exp.exp6_variance_components
```

Each experiment script can also be run with `--full` to select the paper's
full configuration (sample period, universe size, epoch count) in place of
the reduced configuration used for the committed results; see
`results/RESULTS.md` for the runtime estimate behind that reduction.

`reproduce.sh` runs the unit tests and every experiment script with these
commands.
