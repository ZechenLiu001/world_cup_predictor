# World Cup Champion-Chain Signal Predictor

Investment-grade backtest and pre-tournament application report for the World Cup Champion-Chain Signal (CCS).

CCS marks a team as a pre-tournament champion-chain candidate if, in either of the prior two World Cups, it:

- won the tournament; or
- was eliminated in the knockout stage by that tournament's champion or runner-up.

The report tests CCS as a champion-candidate filter, not as a standalone champion picker.

## Deliverables

- `reports/world_cup_ccs_investment_report_en.html` - English executive report.
- `reports/world_cup_ccs_investment_report_zh.html` - Chinese executive report.
- `reports/world_cup_ccs_investment_report.html` - English compatibility copy.
- `reports/pdf/world_cup_ccs_investment_report_en.pdf` - English PDF export.
- `reports/pdf/world_cup_ccs_investment_report_zh.pdf` - Chinese PDF export.
- `reports/pdf/world_cup_ccs_investment_report.pdf` - English compatibility PDF.
- `data/derived/` - reproducible output tables used in the report.
- `reports/assets/` - generated PNG exhibits.
- `scripts/build_report.py` - one-command data refresh and report builder.

## Reproduce

```bash
python3 scripts/build_report.py
```

The script downloads public data from:

- Fjelstul World Cup Database CSVs for historical World Cup teams, matches, standings, and tournament metadata.
- FIFA API ranking schedule and ranking endpoints for pre-tournament FIFA/Coca-Cola Men's World Rankings.

The 2026 view uses the June 11, 2026 FIFA ranking as the frozen pre-tournament ranking snapshot. The report does not use 2026 match results to evaluate 2026 outcomes.

## Method Notes

- `West Germany` and `Germany` are treated as one national-team continuity for CCS logic.
- Early World Cup formats are retained as context, but the primary empirical claim uses the modern knockout era from 1986 to 2022.
- The random-candidate benchmark preserves each year's actual CCS candidate-pool size and asks how often a random same-size pool would cover at least 9 of 10 modern champions.
- FIFA ranking comparisons are used as a strong-team baseline and reader-facing sanity check, not as a full predictive model.
- The favorite-downgrade exhibit is curated from the Top-20 non-CCS audit pool to focus on recognizable strong teams/title-relevant football brands: Argentina, Germany, England, Spain, Portugal, Netherlands, Belgium, Colombia, Croatia, and Uruguay. The broader mechanical audit list is retained in `data/derived/favorite_traps.csv`.
- The 2026 application separately highlights rank-strong non-CCS title names: Spain, Portugal, Brazil, Germany, and Colombia.
- The contender-label permutation simulation controls for the simple "CCS just picks famous teams" explanation by preserving CCS counts inside and outside the traditional title-contender set, then randomizing labels within those groups.
- FIFA rank is retained as an audit field, but the report's main narrative is title-contender recognition plus champion-chain path validation, not a ranking-table model.
