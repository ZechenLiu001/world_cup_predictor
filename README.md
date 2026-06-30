# World Cup Champion-Chain Signal

This repository is a clean report package for the World Cup Champion-Chain Signal (CCS).

CCS is evaluated as a pre-tournament champion shortlist filter and downgrade signal. It is not presented as a standalone calibrated probability model. The main practical claim is narrower:

> Teams outside the recent champion chain should be downgraded as title picks unless odds, Elo, squad quality, injuries, or draw path provide strong counter-evidence.

## Read The Report

- [Chinese PDF](output/pdf/world_cup_ccs_report_zh.pdf)
- [English PDF](output/pdf/world_cup_ccs_report_en.pdf)

## Repository Layout

```text
input/
  raw/          Original compact input tables used as starting material
  original/     Original corrected CCS report PDF

output/
  pdf/          Final Chinese and English report PDFs
  figures/      Report exhibits as PNG files
  tables/       Reader-facing result tables and benchmark summaries
```

The repository is organized as an input/output package. Intermediate Python scripts, caches, and temporary generation files are intentionally excluded from the public package view.

## Core Finding

The report separates two definitions:

- Pure CCS: a broad champion-chain candidate pool. A team is in the pool if it appears in either of the previous two champion-chain sets.
- Strict prior-two-participation exclusion: a narrower rule for teams that played both previous World Cups but had no champion/runner-up path contact.

Across the historical backtest, CCS is strongest as:

- a first-layer champion shortlist filter;
- a downgrade signal for famous non-CCS contenders;
- a complement to public strength measures such as FIFA ranking, Elo, and market odds.

## Key Output Tables

Selected CSVs in `output/tables/` are meant to be read directly:

- `historical_knockout_summary.csv` - historical pure-CCS champion coverage.
- `path_exclusion_summary.csv` - strict prior-two-participation exclusion test.
- `historical_random_benchmark.csv` - same-size random benchmark for pure CCS.
- `path_exclusion_random_benchmark.csv` - same-size random benchmark for strict exclusion.
- `strength_inertia_summary.csv` - previous, prior-two-union, and two-straight top-four/top-eight/top-16 baselines.
- `model_incremental_summary.csv` - FIFA, Elo, blended model, and CCS same-size comparison.
- `model_topk_curve_summary.csv` - Top-K recall comparison.
- `ccs_rank_compression_summary.csv` - CCS plus FIFA rank-gate compression test.
- `ccs_top15_stage_remaining_summary.csv` - CCS plus FIFA Top-15 candidates remaining at field, R16, QF, SF, finalist, and champion stages.
- `ccs_2026_watchlist.csv` - 2026 qualified teams with CCS status.
- `ccs_2026_downgrade_giants.csv` - 2026 high-reputation non-CCS teams highlighted in the report.

## Interpretation Boundary

The random simulations are necessary benchmarks, not forecast odds. They show that same-size random pools do not easily reproduce the observed historical pattern, but they do not prove causality and do not penalize the process of historical rule discovery.

The recommended use is therefore:

1. Start with CCS as a candidate filter.
2. Downgrade non-CCS favorites.
3. Re-admit or rank teams only with explicit strength evidence such as odds, Elo, squad health, or draw path.
