# Results: PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Ruglioni and Filipovic (2026), "PanelGP: A Probabilistic Factor Model for
Asset-Return Panels".

Status: experiment runs in progress. This file is updated after each
experiment's final run with the metrics saved under `results/`, compared
against the paper's reported values.

## Deviations from the paper

- Data source: the paper's cross-section (Section 6.1) uses stock
  characteristics from the global factor data of Jensen, Kelly and Pedersen
  (2023) distributed through WRDS, historical S&P 500 membership from CRSP,
  and the macro-finance predictors of Goyal, Welch and Zafirov (2024). None
  of WRDS, CRSP or that predictor file is reachable through the QuantX
  Dataloader or Massive REST API used in this project. Experiments 2-6
  instead use daily OHLCV bars loaded through the QuantX Dataloader
  (`POST /fetch-ohlcvs`, dividend-adjusted) for a fixed list of large,
  long-history U.S. common stocks, with a reduced set of price/volume-derived
  characteristics and context variables computed in `src/features.py` in
  place of the WRDS panel and the Goyal et al. predictors. This is recorded
  in full detail once the application experiments have been run (see the
  per-experiment sections below).
- Universe and sample period: see `src/config.py` (`APP_*_RUN` versus
  `APP_*_PAPER` variables) and the per-experiment sections below for the
  realized dates, universe size and the cut rule applied.
- Experiment 1 (Section 5.1) uses the paper's own simulated data-generating
  process with no external data dependency, so it has no data-source
  deviation; its optimization epoch count is reduced from the paper's 5,000
  to a smaller number (`SIM_N_EPOCHS_RUN` in `src/config.py`), a scalable
  count rather than a method parameter, with the paper's full count runnable
  via `--full`.

## Not reproduced

(updated as experiments complete)
