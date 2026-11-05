# GridAlpha: notes for the AI engineer

You are fixing or improving a gated ML trading-signal pipeline. Read this before editing.

## What the project does
Physics-informed spatio-temporal GNN that forecasts the distribution of real-time minus
day-ahead nodal price spreads, turns it into sized virtual-bid tickets (Kelly), backtests
walk-forward, and publishes a Blueprint.js dashboard. Entry point: `python -m gridalpha.cli`.

## Rules
- Never edit `config/gates.yaml`, the `trade:` section of `config/default.yaml`, or `.github/`.
  If a gate fails, fix the model, features or data, not the threshold.
- Never introduce look-ahead: features for operating day D may use forecasts published
  before D-1 10:00 CT and realized data through D-2 only. `data/build.py::leak_test` must pass.
- Keep the system signal-only. Nothing may submit bids or touch money.
- Every change must pass, locally, before a PR:
  `python -m pytest -q` and `GRIDALPHA_GATES=config/gates_ci.yaml python -m gridalpha.cli run --profile ci`.

## Map
- `gridalpha/sim/`: physics-based market simulator (DC-OPF, LMPs from LP duals)
- `gridalpha/data/`: canonical schema, ERCOT adapter (gridstatus), point-in-time features, leak test
- `gridalpha/models/`: ST-GNN (plain PyTorch), losses incl. DC power-flow physics, baselines
- `gridalpha/train/`: pretraining, fine-tuning, walk-forward folds, ensembling, conformal calibration
- `gridalpha/trade/`: Kelly sizing, tickets, backtest, live signals
- `gridalpha/autopilot.py`: stages, gates, budget, failure -> GitHub issue
- `gridalpha/cloud/`: Modal app + launcher, DigitalOcean launcher, artifact sync
- `dashboard/`: React + Blueprint.js + ECharts; reads `signals.json`

## Typical failures and where to look
- ERCOT column names changed -> `data/ercot.py::_col` candidates.
- Gate `gnn_beats_lgbm_pinball` -> try `model.hidden`, `model.layers`, `loss.w_*`, more epochs,
  or new features in `data/build.py` (add them to the leak test's coverage automatically).
- Gate `coverage80` -> `train/calibrate.py` (per-hour conformal shift).
- Budget refusal -> reduce `train.ensemble_seeds` or the profile; do not raise the budget.
