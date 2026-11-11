# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this repository implements

A replication of selected experiments from "PanelGP: A Probabilistic Factor
Model for Asset-Return Panels" (Ruglioni and Filipovic, 2026): a sparse
variational Gaussian-process factor model for panels of stock excess
returns, in which both the characteristic-driven loading map and the
context-driven factor map are learned jointly with stock- and
date-specific innovations and observation noise (Sections 3-4). See
`README.md` for the list of reproduced experiments and `results/RESULTS.md`
for the numbers produced by the latest run next to the paper's reported
values, with a verdict for each metric and the full list of deviations.

## Layout

```
src/        PanelGP model (src/panelgp.py), kernels (src/kernels.py),
            variational training loop (src/training.py), bilinear predictive
            density (src/bilinear.py), variance decomposition and metrics
            (src/prediction.py, src/metrics.py), IPCA benchmark
            (src/ipca_model.py), feature construction (src/features.py),
            S&P-500-style universe selection (src/universe.py), panel
            assembly (src/panel_data.py), the shared experiment pipeline
            (src/pipeline.py), and every experiment parameter
            (src/config.py, module-level UPPER_CASE variables only).
exp/        One script per experiment, run as `python -m exp.<name>`. Each
            script only calls the src/ functions by their documented
            signatures and writes its outputs to results/.
data/       data_loader.py (QuantX Dataloader OHLCV bars, the only market
            data path) and generate_synthetic.py (Experiment 1's
            data-generating process). Downloaded bars are cached under
            data/cache/, which is not committed.
tests/      pytest tests of src/ components and of the data path.
results/    Every generated table, plot and metric, plus RESULTS.md.
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Tests:

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # also runs the market-data tests
                                         # (needs DATALOADER_API_KEY/MASSIVE_API_KEY)
```

Experiments:

```bash
python -m exp.exp1_simulation_recovery
python -m exp.exp2_fixed_kernel_comparison
python -m exp.exp3_factor_dimension_sweep
python -m exp.exp4_rolling_and_trading_costs
python -m exp.exp5_bilinear_higher_moments
python -m exp.exp6_variance_components
```

Each experiment script accepts `--full` to run the paper's full
configuration (sample period, universe size, inducing points, epochs,
seeds) instead of the reduced configuration used for the results committed
in `results/`; see `results/RESULTS.md`, "Deviations from the paper", for
the runtime estimate behind the reduction. `reproduce.sh` runs the test
suite and every experiment with the commands above.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a one-line comment giving its meaning,
  unit and source (a paper section/equation, or "not specified in the
  paper"); no classes or parameter dicts. Reduced-versus-paper settings are
  kept as adjacent `_RUN`/`_PAPER` pairs.
- Market data is loaded only through `data/data_loader.py`, which wraps the
  QuantX Dataloader REST API (`POST /fetch-ohlcvs`); Experiment 1's
  simulated panel is generated only through `data/generate_synthetic.py`.
  No other module performs I/O against an external data source.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` file via `python-dotenv`; `.env.example` holds only
  empty placeholders. Keys must never appear in code, logs or results.
- Deviations from the paper (data source, universe, sample period,
  optimization budget) are recorded only in `results/RESULTS.md`, not in
  code comments or config defaults.
- After any rerun of an experiment script, the files it writes under
  `results/` are overwritten; the narrative numbers and verdicts in
  `results/RESULTS.md` must then be checked against the new files and
  updated by hand, they are not regenerated automatically.
