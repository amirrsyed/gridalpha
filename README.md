# GridAlpha

A physics-informed graph neural network that forecasts where the power grid will congest
tomorrow, turns that forecast into complete, Kelly-sized virtual-bid tickets for the ERCOT
day-ahead market, and runs itself: data, training, gating, backtests, signals and a
Blueprint.js dashboard, on a schedule, with an AI engineer that fixes failures through
pull requests.

Built on the market mechanics in Neel Somani's *Power 2026*: LMP = energy + loss +
congestion, congestion comes from binding lines, and location and timing of load (AI data
centers, the evening solar ramp, wind trapped behind export limits) decide who pays.

## Quick start (no accounts needed)

```bash
pip install -r requirements.txt
python -m pytest -q
# full pipeline on the built-in market simulator, laptop-sized
python -m gridalpha.cli run --profile local
python -m gridalpha.cli status
# dashboard
cd dashboard && npm install && npm run dev        # reads public/signals.json
```

The simulator is a 48-bus Texas-like grid whose day-ahead and real-time markets are cleared
by a DC optimal power flow, so every congestion price comes from a binding line. It exists
to develop and test the pipeline end to end; its inefficiencies are built in and are larger
than a real market's, so treat simulated P&L as a pipeline test, not as evidence of edge.

## Pipeline

`ingest -> validate* -> features -> pretrain -> baselines* -> train* -> calibrate* -> backtest* -> signals -> publish`
(* gated by `config/gates.yaml`)

| Stage | What it does |
| --- | --- |
| ingest | Simulator, or ERCOT via `gridstatus` + Open-Meteo forecasts + FRED gas |
| validate | Missing data, and a leak test that perturbs everything after the bid deadline |
| features | Point-in-time node, edge, global and history tensors |
| pretrain | Self-supervised masked-node reconstruction of congestion (no labels) |
| baselines | Naive empirical quantiles and LightGBM quantile regression |
| train | ST-GNN ensemble per walk-forward fold, multi-task + DC power-flow physics loss |
| calibrate | Quantile ensembling and per-hour conformal recalibration |
| backtest | Tickets sized by portfolio Kelly, cleared against their price limits, compounding |
| signals | Tomorrow's tickets from the latest fold's ensemble |
| publish | `signals.json` for the dashboard |

## Going live on real data

1. Create a free ERCOT Public API account; add `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`,
   `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY` as repository secrets.
2. Create a Modal account (Starter plan includes monthly compute credits); add
   `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
3. Add `ANTHROPIC_API_KEY` for the AI engineer, create the labels `autopilot` and `claude`,
   set `@OWNER` in `.github/CODEOWNERS` and protect `main` with code-owner review.
4. Enable GitHub Pages (source: GitHub Actions).
5. Run the `retrain` workflow by hand (Actions -> retrain -> Run workflow). For the first run,
   set "Stop after this stage" to `features`: it downloads ERCOT history and builds features
   without any GPU cost, and the download cache is saved even if the run stops early. Then run
   it again from `pretrain` to train. `daily-signals` publishes tickets every morning once a
   retrain has passed every gate; before that it exits early with a notice.

Cost guardrails: every cloud job has a timeout, and the autopilot refuses to launch work
that would push the month past `compute.monthly_budget_usd` (default $60).

## Safety

The system produces signals only. It never registers with an ISO, posts collateral or
submits bids; a person does that. Gate thresholds and trading limits are owner-protected.
