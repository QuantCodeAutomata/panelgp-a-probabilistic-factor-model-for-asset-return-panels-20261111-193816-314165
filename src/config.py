"""Paper-derived parameters for PanelGP (Ruglioni and Filipovic, 2026).

Every module-level variable states the paper section or equation it comes
from, or notes that the paper does not specify it. No classes or parameter
dictionaries are used so that every value is a single, greppable name.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: simulation recovery of smooth maps and variance parameters
# (Section 5, "Simulation")
# ---------------------------------------------------------------------------
SIM_N_ASSETS = 80          # balanced panel size, Section 5.1
SIM_N_MONTHS = 50          # balanced panel size, Section 5.1
SIM_H = 2                  # number of factors, Section 5.1
SIM_SIGMA_BETA2 = 0.05     # true loading-innovation variance D_beta = sigma_beta^2 I, Sec. 5.1
SIM_SIGMA_PHI2 = 0.05      # true factor-innovation variance D_phi = sigma_phi^2 I, Sec. 5.1
SIM_SIGMA_R2 = 0.05        # true observation-noise variance, Sec. 5.1
SIM_N_REPLICATIONS = 5     # independent panels and optimization seeds, Sec. 5.1
SIM_N_INDUCING = 64        # trainable inducing inputs per component, Sec. 5 / App. C.3
SIM_N_EPOCHS_PAPER = 5000  # training epochs, App. C.4 ("the simulation 5,000")
SIM_N_EPOCHS_RUN = 1500    # reduced epoch count (optimization is a scalable count, not a
                           # method parameter); kept well past the point where the training
                           # loss has flattened in the executed runs, see results/RESULTS.md
SIM_GRID_RESOLUTION = 41   # resolution of the dense grid for function R^2; not specified
                           # in the paper, chosen fine enough that R^2 is stable to the digit

# ---------------------------------------------------------------------------
# Shared optimization settings (Appendix C.4) used by both the simulation and
# the application, except where overridden above/below.
# ---------------------------------------------------------------------------
LR_VARIATIONAL = 0.01       # Adam learning rate for variational parameters, App. C.4
LR_MODEL = 0.005            # Adam learning rate for model parameters (incl. inducing inputs)
GRAD_CLIP_NORM = 1.0        # gradients clipped to unit Euclidean norm, App. C.4
MINIBATCH_ROWS = 2048       # rows per minibatch, App. C.4
N_FACTOR_DRAWS = 32         # independent Gaussian draws per likelihood evaluation
                            # (paired with their negatives -> 64 factor samples), App. C.4
LR_HALVING_FRACTIONS = (0.30, 0.50, 0.60, 0.70, 0.80, 0.90)  # App. C.4
JITTER_INITIAL = 1e-7       # initial Cholesky diagonal stabilizer, App. C.3
JITTER_GROWTH = 10.0        # tenfold increase until factorization succeeds, App. C.3
WHITENED_MEAN_INIT_STD = 0.001  # Gaussian perturbation std for whitened mean, App. C.4
INIT_SIGMA_R2 = 0.1         # initial observation-noise variance in scaled units, App. C.4
INIT_RHO_H = 0.1            # initial factor-innovation share rho_h, App. C.4
INIT_B_H = 0.1              # initial constant-term share b_h, App. C.4
INIT_A_H = 0.8              # initial RBF-term share a_h, App. C.4

# ---------------------------------------------------------------------------
# Experiments 2-4: fixed-training kernel comparison, factor-dimension sweep,
# and annual re-estimation on S&P 500 constituents (Sections 6, App. D.2)
# ---------------------------------------------------------------------------
APP_H_PAPER = 5                 # number of factors chosen on the validation period, Sec. 6
APP_H_GRID_PAPER = list(range(1, 9))  # H = 1..8 swept in Experiment 3, Sec. 6.3/Fig. 3
APP_N_CHARACTERISTICS_PAPER = 137     # characteristics kept (>=70% coverage), Sec. 6.1
APP_N_CONTEXT_PAPER = 6               # Goyal et al. (2024) context variables, Sec. 6.1
APP_N_INDUCING_PAPER = 160            # trainable inducing inputs per component, Sec. 6/App. C.3
APP_N_EPOCHS_PAPER = 500              # training epochs in the application, App. C.4
APP_TRAIN_START_PAPER = "1960-01-01"  # fixed-training window start, Sec. 6.2
APP_TRAIN_END_PAPER = "1989-12-31"    # fixed-training window end, Sec. 6.2
APP_VALID_START_PAPER = "1990-01-01"  # validation window used to pick H, Sec. 6.2
APP_VALID_END_PAPER = "1994-12-31"
APP_EVAL_START_PAPER = "1995-01-01"   # evaluation window, Sec. 6.2
APP_EVAL_END_PAPER = "2024-12-31"
APP_WINDOW_MONTHS_PAPER = 360         # input months per training window, Sec. 6.2/6.5
APP_N_SEEDS_PAPER = 5                 # optimization seeds per specification, Sec. 6.2

# Executed configuration: the paper's data source (WRDS characteristics panel
# of Jensen et al. 2023, CRSP point-in-time S&P 500 membership, and the Goyal
# et al. 2024 macro-finance predictors) is not reachable through the QuantX
# Dataloader or Massive REST API, and the available OHLCV history for the
# substitute universe does not reach back to 1960. The executed run keeps the
# paper's method, kernels, number of factors and inference exactly, over a
# shorter, available sample window and a smaller universe. See
# results/RESULTS.md, "Deviations from the paper", for the full explanation.
APP_H_RUN = 5                          # number of factors kept at the paper's choice
APP_H_GRID_RUN = [1, 2, 3, 5, 8]       # reduced grid of factor counts for Experiment 3
APP_N_CHARACTERISTICS_RUN = 16         # price/volume-derived characteristics (see data_loader)
APP_N_CONTEXT_RUN = 4                  # price-derived context variables (see data_loader)
APP_N_INDUCING_RUN = 48                # reduced inducing inputs per component
APP_N_EPOCHS_RUN = 60                  # reduced training epochs (scalable count)
APP_TRAIN_START_RUN = "2005-01-01"
APP_TRAIN_END_RUN = "2014-12-31"       # 120 months, proportional to the paper's 360
APP_VALID_START_RUN = "2015-01-01"
APP_VALID_END_RUN = "2016-12-31"
APP_EVAL_START_RUN = "2017-01-01"
APP_EVAL_END_RUN = "2024-12-31"
APP_WINDOW_MONTHS_RUN = 120
APP_N_SEEDS_RUN = 1                    # reduced optimization seeds (scalable count)
APP_UNIVERSE_SIZE_PAPER = "465-497 (full historical S&P 500 membership)"
APP_UNIVERSE_SIZE_RUN = 60             # tickers, selected by average dollar volume

NEWEY_WEST_LAGS = 12          # lags for Newey-West standard errors, Sec. 6.6
HOLM_N_COMPARISONS = 15       # comparisons per criterion in the Holm step-down test, Sec. 6.6
LONG_SHORT_DECILE = 0.10      # top/bottom decile for the long-short portfolio, Sec. 6.5
LONG_SHORT_GROSS_PER_SIDE = 0.5  # gross exposure on each side, Sec. 6.5

# ---------------------------------------------------------------------------
# Experiment 4: trading costs (Section 6.7)
# ---------------------------------------------------------------------------
TRADING_COST_BPS = 10.0        # proportional cost per dollar traded, Sec. 6.7 (Frazzini et al. 2018)
ROLLING_FIRST_WINDOW_START_PAPER = "1965-01-01"  # first rolling-window start, Sec. 6.2
SUBPERIOD_SPLIT_YEAR = 2010     # 1995-2009 vs 2010-2024 split, Table S3

# ---------------------------------------------------------------------------
# Experiment 5: higher moments of the bilinear predictive distribution
# (Section 6.8 / Appendix B.3)
# ---------------------------------------------------------------------------
CGF_QUAD_ABSTOL = 1e-9     # SciPy quad_vec absolute tolerance, App. B.3
CGF_QUAD_RELTOL = 1e-9     # SciPy quad_vec relative tolerance, App. B.3
STANDARDIZED_ERROR_BINS = (1.0, 2.0, 3.0)  # bin edges for |error|/sd grouping, Sec. 6.8/Fig. S3

# ---------------------------------------------------------------------------
# Experiment 6: predicted variance calibration (Section 6.8, Equation 27)
# ---------------------------------------------------------------------------
N_QUINTILES = 5            # temporal and cross-sectional quintiles, Sec. 6.8

RANDOM_SEED_BASE = 20261111  # base seed; replication/seed k uses RANDOM_SEED_BASE + k.
                              # The paper does not specify the seed generation procedure.

# Input scaling: characteristics and context in [0, 1] in the simulation
# (Appendix C.1), with coordinate variance 1/12 for a uniform variable.
UNIFORM_VARIANCE = 1.0 / 12.0
