#!/usr/bin/env bash
# Runs the unit tests and every experiment of the PanelGP replication with
# the same commands as README.md. Market-data tests and the application
# experiments (2-6) need DATALOADER_API_KEY and MASSIVE_API_KEY in .env.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1_simulation_recovery
python -m exp.exp2_fixed_kernel_comparison
python -m exp.exp3_factor_dimension_sweep
python -m exp.exp4_rolling_and_trading_costs
python -m exp.exp5_bilinear_higher_moments
python -m exp.exp6_variance_components
