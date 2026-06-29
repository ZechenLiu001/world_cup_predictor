#!/usr/bin/env python3
"""Build the World Cup CCS investment-grade report.

The script intentionally keeps data acquisition and report generation in one
auditable path so the published artifact can be refreshed with a single command.
"""

from __future__ import annotations

import html
import json
import math
import textwrap
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_DERIVED = ROOT / "data" / "derived"
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"

FJELSTUL_BASE = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
FIFA_API = "https://api.fifa.com/api/v3"

WORLD_CUP_STARTS = {
    1998: "1998-06-10",
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
    2026: "2026-06-11",
}

TEAM_ALIASES = {
    "United States": "USA",
    "USA": "USA",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Germany": "Germany",
    "West Germany": "Germany",
}

TEAM_ZH = {
    "Argentina": "阿根廷",
    "Belgium": "比利时",
    "Brazil": "巴西",
    "Chile": "智利",
    "Colombia": "哥伦比亚",
    "Croatia": "克罗地亚",
    "Czech Republic": "捷克",
    "Denmark": "丹麦",
    "England": "英格兰",
    "France": "法国",
    "Germany": "德国",
    "Greece": "希腊",
    "Japan": "日本",
    "Mexico": "墨西哥",
    "Morocco": "摩洛哥",
    "Netherlands": "荷兰",
    "Norway": "挪威",
    "Peru": "秘鲁",
    "Poland": "波兰",
    "Portugal": "葡萄牙",
    "Senegal": "塞内加尔",
    "Spain": "西班牙",
    "Switzerland": "瑞士",
    "Turkey": "土耳其",
    "Ecuador": "厄瓜多尔",
    "Austria": "奥地利",
    "Iran": "伊朗",
    "Italy": "意大利",
    "Uruguay": "乌拉圭",
    "USA": "美国",
}

PERFORMANCE_ZH = {
    "group stage": "小组赛",
    "round of 16": "16强",
    "quarter-finals": "8强",
    "third-place match": "季军赛",
    "final": "决赛",
    "not yet known": "待定",
}

CURATED_HISTORICAL_CONTENDERS = {
    "Argentina",
    "Belgium",
    "Brazil",
    "Colombia",
    "Croatia",
    "England",
    "France",
    "Germany",
    "Italy",
    "Netherlands",
    "Portugal",
    "Spain",
    "Uruguay",
}

CURATED_2026_DOWNGRADE_TEAMS = {
    "Spain",
    "Portugal",
    "Brazil",
    "Germany",
    "Colombia",
}


def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def display_team(name: str, lang: str) -> str:
    return TEAM_ZH.get(name, name) if lang == "zh" else name


def display_performance(value: str, lang: str) -> str:
    return PERFORMANCE_ZH.get(value, value) if lang == "zh" else value


def read_csv_url(name: str) -> pd.DataFrame:
    url = f"{FJELSTUL_BASE}/{name}.csv"
    return pd.read_csv(url)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_dirs() -> None:
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "pdf").mkdir(parents=True, exist_ok=True)


def extract_year(tournament_id: str) -> int:
    return int(str(tournament_id).split("-")[-1])


def team_desc(item: dict) -> str:
    names = item.get("TeamName") or item.get("Name") or []
    if names:
        return names[0].get("Description", "")
    return ""


def get_2026_qualified() -> pd.DataFrame:
    data = fetch_json(f"{FIFA_API}/teamsqualified/season/285023?language=en")["Results"]
    rows = []
    for item in data:
        rows.append(
            {
                "year": 2026,
                "team_name": norm_team(team_desc(item)),
                "team_code": item.get("IdCountry"),
                "performance": "not yet known",
                "source": "FIFA API teamsqualified/season/285023",
            }
        )
    return pd.DataFrame(rows)


def chain_sets(matches: pd.DataFrame, standings: pd.DataFrame) -> dict[int, set[str]]:
    """Return teams in each tournament's champion-chain set.

    For each tournament, include the champion plus every team beaten by either
    finalist in a knockout-stage match. This is the exact information available
    before later tournaments.
    """

    sets: dict[int, set[str]] = {}
    standings = standings[standings["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    standings["year"] = standings["tournament_id"].map(extract_year)
    matches = matches[matches["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    matches["year"] = matches["tournament_id"].map(extract_year)

    for year, rows in standings.groupby("year"):
        finalists = rows.loc[rows["position"].isin([1, 2]), "team_name"].map(norm_team).tolist()
        champion = rows.loc[rows["position"].eq(1), "team_name"].map(norm_team).iloc[0]
        chain = {champion}
        ko = matches[(matches["year"] == year) & (matches["knockout_stage"].eq(1))]
        for _, match in ko.iterrows():
            home = norm_team(match["home_team_name"])
            away = norm_team(match["away_team_name"])
            result = match["result"]
            if result == "home team win":
                winner, loser = home, away
            elif result == "away team win":
                winner, loser = away, home
            else:
                continue
            if winner in finalists:
                chain.add(loser)
        sets[year] = chain
    return sets


def build_team_year() -> pd.DataFrame:
    qualified = read_csv_url("qualified_teams")
    matches = read_csv_url("matches")
    standings = read_csv_url("tournament_standings")

    qualified = qualified[qualified["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    qualified["year"] = qualified["tournament_id"].map(extract_year)
    qualified["team_name"] = qualified["team_name"].map(norm_team)
    qualified = qualified[qualified["year"].between(1998, 2022)][
        ["year", "team_name", "team_code", "performance"]
    ].copy()
    qualified["source"] = "Fjelstul qualified_teams.csv"

    all_teams = pd.concat([qualified, get_2026_qualified()], ignore_index=True)
    chains = chain_sets(matches, standings)

    rows = []
    for _, row in all_teams.iterrows():
        year = int(row["year"])
        prior_years = sorted([y for y in chains if y < year])[-2:]
        ccs_sources = [y for y in prior_years if row["team_name"] in chains.get(y, set())]
        rows.append(
            {
                **row.to_dict(),
                "ccs": int(bool(ccs_sources)),
                "ccs_source_years": ";".join(map(str, ccs_sources)),
                "prior_windows": ";".join(map(str, prior_years)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DERIVED / "ccs_team_year.csv", index=False)
    return out


def schedule_for_world_cup(starts: dict[int, str]) -> pd.DataFrame:
    schedules = fetch_json(
        f"{FIFA_API}/rankingschedules/all?type=0&gender=1&language=en"
    )["Results"]
    sched = pd.DataFrame(schedules)
    sched["official_date"] = pd.to_datetime(sched["OfficialDate"]).dt.tz_localize(None)
    rows = []
    for year, start in starts.items():
        start_dt = pd.to_datetime(start)
        row = sched[sched["official_date"].le(start_dt)].sort_values("official_date").tail(1).iloc[0]
        rows.append(
            {
                "year": year,
                "world_cup_start": start,
                "ranking_schedule_id": row["IdRankingSchedule"],
                "ranking_official_date": row["OfficialDate"][:10],
            }
        )
    return pd.DataFrame(rows)


def build_rankings() -> pd.DataFrame:
    schedule = schedule_for_world_cup(WORLD_CUP_STARTS)
    frames = []
    for _, row in schedule.iterrows():
        data = fetch_json(
            f"{FIFA_API}/rankingsbyschedule?rankingScheduleId={row['ranking_schedule_id']}&language=en"
        )["Results"]
        records = []
        for item in data:
            records.append(
                {
                    "year": int(row["year"]),
                    "ranking_schedule_id": row["ranking_schedule_id"],
                    "ranking_official_date": row["ranking_official_date"],
                    "team_name": norm_team(team_desc(item)),
                    "team_code": item.get("IdCountry"),
                    "fifa_rank": item.get("Rank"),
                    "fifa_points": item.get("DecimalTotalPoints"),
                }
            )
        frames.append(pd.DataFrame(records))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA_DERIVED / "fifa_rankings_pre_wc.csv", index=False)
    return out


def random_benchmark(modern_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    probs = []
    for _, r in modern_summary.iterrows():
        p = r["ccs"] / r["participants"]
        probs.append(p)
        rows.append(
            {
                "year": int(r["year"]),
                "participants": int(r["participants"]),
                "ccs_candidates": int(r["ccs"]),
                "random_hit_probability": p,
                "champion": r["champion"],
                "ccs_hit": int(r["champ_ccs"]),
            }
        )

    dist = [1.0] + [0.0] * len(probs)
    for p in probs:
        new = [0.0] * len(dist)
        for k, v in enumerate(dist):
            new[k] += v * (1 - p)
            if k + 1 < len(dist):
                new[k + 1] += v * p
        dist = new

    out = pd.DataFrame(rows)
    out["expected_random_hits"] = sum(probs)
    out["actual_hits"] = int(modern_summary["champ_ccs"].sum())
    out["prob_random_ge_9_of_10"] = sum(dist[9:])
    rng = np.random.default_rng(20260611)
    simulated_hits = (rng.random((200_000, len(probs))) < np.array(probs)).sum(axis=1)
    out["simulation_runs"] = 200_000
    out["prob_random_ge_9_of_10_simulated"] = float((simulated_hits >= 9).mean())
    out.to_csv(DATA_DERIVED / "random_benchmark.csv", index=False)
    return out


def random_benchmark_from_summary(summary: pd.DataFrame, output_name: str) -> pd.DataFrame:
    sample = summary[summary["evaluable"].eq(1)].copy()
    probs = (sample["ccs_candidates"] / sample["participants"]).tolist()
    actual_hits = int(sample["champ_ccs"].sum())
    dist = poisson_binomial_distribution(probs)
    out = sample[
        ["year", "participants", "ccs_candidates", "random_hit_probability", "champion", "champ_ccs"]
    ].copy()
    out["expected_random_hits"] = sum(probs)
    out["actual_hits"] = actual_hits
    out["prob_random_ge_actual"] = float(dist[actual_hits:].sum())
    rng = np.random.default_rng(20260611)
    simulated_hits = (rng.random((200_000, len(probs))) < np.array(probs)).sum(axis=1)
    out["simulation_runs"] = 200_000
    out["prob_random_ge_actual_simulated"] = float((simulated_hits >= actual_hits).mean())
    out.to_csv(DATA_DERIVED / output_name, index=False)
    return out


def poisson_binomial_distribution(probs: list[float]) -> np.ndarray:
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, np.array([1 - p, p]))
    return dist


def build_historical_knockout_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qualified = read_csv_url("qualified_teams")
    matches = read_csv_url("matches")
    standings = read_csv_url("tournament_standings")

    qualified = qualified[qualified["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    matches = matches[matches["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    standings = standings[standings["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    for frame in [qualified, matches, standings]:
        frame["year"] = frame["tournament_id"].map(extract_year)
    qualified["team_name"] = qualified["team_name"].map(norm_team)
    standings["team_name"] = standings["team_name"].map(norm_team)

    chains = chain_sets(matches, standings)
    knockout_years = sorted(
        int(year)
        for year in standings["year"].unique()
        if int(((matches["year"].eq(year)) & matches["knockout_stage"].eq(1)).sum()) > 0
    )
    all_years = sorted(int(year) for year in standings["year"].unique())

    scope_rows = []
    for year in all_years:
        ko_matches = int(((matches["year"].eq(year)) & matches["knockout_stage"].eq(1)).sum())
        if year == 1950:
            note = "final-round-robin; no knockout-stage path"
        elif year in {1974, 1978, 1982}:
            note = "hybrid format; knockout path exists but is shorter than modern bracket"
        elif ko_matches:
            note = "knockout path available"
        else:
            note = "no knockout path"
        scope_rows.append(
            {
                "year": year,
                "participants": int((qualified["year"].eq(year)).sum()),
                "knockout_matches": ko_matches,
                "has_knockout_path": int(ko_matches > 0),
                "included_in_historical_backtest": int(ko_matches > 0),
                "format_note": note,
            }
        )

    summary_rows = []
    team_rows = []
    for year in knockout_years:
        teams = qualified[qualified["year"].eq(year)][["year", "team_name", "team_code", "performance"]].copy()
        prior_years = [prior for prior in knockout_years if prior < year][-2:]
        prior_chain = set()
        for prior in prior_years:
            prior_chain |= chains.get(prior, set())
        champion = standings.loc[
            standings["year"].eq(year) & standings["position"].eq(1), "team_name"
        ].iloc[0]
        teams["ccs"] = teams["team_name"].isin(prior_chain).astype(int)
        teams["ccs_source_years"] = teams["team_name"].map(
            lambda team: ";".join(str(prior) for prior in prior_years if team in chains.get(prior, set()))
        )
        teams["prior_windows"] = ";".join(map(str, prior_years))
        teams["source"] = "Fjelstul qualified_teams.csv; prior two knockout-path World Cups"
        teams["evaluable"] = int(len(prior_years) == 2)
        team_rows.append(teams)
        ccs_candidates = int(teams["ccs"].sum())
        participants = int(len(teams))
        champ_ccs = int(champion in set(teams.loc[teams["ccs"].eq(1), "team_name"]))
        summary_rows.append(
            {
                "year": year,
                "champion": champion,
                "participants": participants,
                "ccs_candidates": ccs_candidates,
                "ccs_share": ccs_candidates / participants if participants else np.nan,
                "random_hit_probability": ccs_candidates / participants if participants else np.nan,
                "champ_ccs": champ_ccs,
                "evaluable": int(len(prior_years) == 2),
                "prior_windows": ";".join(map(str, prior_years)),
                "champion_prior_ccs_sources": ";".join(
                    str(prior) for prior in prior_years if champion in chains.get(prior, set())
                ),
            }
        )

    team_year = pd.concat(team_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    scope = pd.DataFrame(scope_rows)
    team_year.to_csv(DATA_DERIVED / "historical_knockout_team_year.csv", index=False)
    summary.to_csv(DATA_DERIVED / "historical_knockout_summary.csv", index=False)
    scope.to_csv(DATA_DERIVED / "world_cup_format_scope.csv", index=False)
    return team_year, summary, scope


def path_contact_reason(team: str, year: int, matches: pd.DataFrame, standings: pd.DataFrame) -> tuple[bool, str]:
    year_standings = standings[standings["year"].eq(year)]
    champion = year_standings.loc[year_standings["position"].eq(1), "team_name"].iloc[0]
    finalists = set(year_standings.loc[year_standings["position"].isin([1, 2]), "team_name"])
    if team == champion:
        return True, "own champion"

    stage_mask = matches["knockout_stage"].eq(1) | matches["stage_name"].eq("second group stage")
    rows = matches[
        matches["year"].eq(year)
        & stage_mask
        & ((matches["home_team_name"].eq(team)) | (matches["away_team_name"].eq(team)))
    ]
    reasons = []
    for _, match in rows.iterrows():
        home = norm_team(match["home_team_name"])
        away = norm_team(match["away_team_name"])
        if match["result"] == "home team win":
            winner, loser = home, away
        elif match["result"] == "away team win":
            winner, loser = away, home
        else:
            continue
        if loser == team and winner in finalists:
            role = "champion" if winner == champion else "runner-up"
            reasons.append(f"{year}: lost to {winner} ({role}) in {match['stage_name']}")
    return bool(reasons), "; ".join(reasons)


def build_path_exclusion_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    qualified = read_csv_url("qualified_teams")
    matches = read_csv_url("matches")
    standings = read_csv_url("tournament_standings")

    qualified = qualified[qualified["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    matches = matches[matches["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    standings = standings[standings["tournament_name"].str.contains("Men's World Cup", regex=False)].copy()
    for frame in [qualified, matches, standings]:
        frame["year"] = frame["tournament_id"].map(extract_year)
    qualified["team_name"] = qualified["team_name"].map(norm_team)
    matches["home_team_name"] = matches["home_team_name"].map(norm_team)
    matches["away_team_name"] = matches["away_team_name"].map(norm_team)
    standings["team_name"] = standings["team_name"].map(norm_team)

    years = sorted(int(year) for year in standings["year"].unique())
    champion_by_year = {
        year: standings.loc[standings["year"].eq(year) & standings["position"].eq(1), "team_name"].iloc[0]
        for year in years
    }

    detail_rows = []
    summary_rows = []
    for idx, year in enumerate(years):
        if idx < 2:
            continue
        prior_years = years[idx - 2 : idx]
        target_teams = qualified[qualified["year"].eq(year)].copy()
        champion = champion_by_year[year]
        for team in sorted(target_teams["team_name"].unique()):
            prior_participation = [
                not qualified[qualified["year"].eq(prior) & qualified["team_name"].eq(team)].empty
                for prior in prior_years
            ]
            contact_notes = []
            if all(prior_participation):
                for prior in prior_years:
                    has_contact, reason = path_contact_reason(team, prior, matches, standings)
                    if has_contact:
                        contact_notes.append(f"{prior}: {reason}" if reason == "own champion" else reason)
            has_path_contact = bool(contact_notes)
            clean_non_contact = bool(all(prior_participation) and not has_path_contact)
            detail_rows.append(
                {
                    "year": year,
                    "team_name": team,
                    "champion": int(team == champion),
                    "prior_windows": ";".join(map(str, prior_years)),
                    "prior_participated_both": int(all(prior_participation)),
                    "path_contact": int(has_path_contact),
                    "clean_non_contact": int(clean_non_contact),
                    "contact_reasons": " | ".join(contact_notes),
                }
            )

        year_detail = pd.DataFrame([row for row in detail_rows if row["year"] == year])
        champion_row = year_detail[year_detail["champion"].eq(1)].iloc[0]
        summary_rows.append(
            {
                "year": year,
                "champion_name": champion,
                "participants": int(len(target_teams)),
                "prior_participated_both_teams": int(year_detail["prior_participated_both"].sum()),
                "path_contact_teams": int(
                    (year_detail["prior_participated_both"].eq(1) & year_detail["path_contact"].eq(1)).sum()
                ),
                "clean_non_contact_teams": int(year_detail["clean_non_contact"].sum()),
                "clean_non_contact_champions": int(
                    (year_detail["clean_non_contact"].eq(1) & year_detail["champion"].eq(1)).sum()
                ),
                "champion_prior_participated_both": int(champion_row["prior_participated_both"]),
                "champion_path_contact": int(champion_row["path_contact"]),
                "champion_clean_non_contact": int(champion_row["clean_non_contact"]),
                "champion_contact_reasons": champion_row["contact_reasons"],
                "prior_windows": champion_row["prior_windows"],
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(DATA_DERIVED / "path_exclusion_team_year.csv", index=False)
    summary.to_csv(DATA_DERIVED / "path_exclusion_summary.csv", index=False)
    return detail, summary


def exclusion_random_benchmark(path_summary: pd.DataFrame) -> pd.DataFrame:
    out = path_summary[
        ["year", "participants", "clean_non_contact_teams", "clean_non_contact_champions"]
    ].copy()
    out["random_exclusion_probability"] = out["clean_non_contact_teams"] / out["participants"]
    out["random_keep_champion_probability"] = 1 - out["random_exclusion_probability"]
    out["expected_random_excluded_champions"] = out["random_exclusion_probability"].sum()
    out["prob_random_zero_excluded_champions_exact"] = out["random_keep_champion_probability"].prod()

    rng = np.random.default_rng(20260611)
    probs = out["random_exclusion_probability"].to_numpy()
    simulated_exclusions = (rng.random((200_000, len(probs))) < probs).sum(axis=1)
    out["simulation_runs"] = 200_000
    out["prob_random_zero_excluded_champions_simulated"] = float((simulated_exclusions == 0).mean())
    out.to_csv(DATA_DERIVED / "path_exclusion_random_benchmark.csv", index=False)
    return out


def contender_permutation_benchmark(joined: pd.DataFrame, modern_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control for the simple "it just picks famous teams" explanation.

    Within each tournament, preserve how many CCS teams come from the curated
    title-contender set and how many come from outside it, then randomize the
    CCS label inside those two groups. The champion hit probability is therefore
    the share of CCS labels inside the champion's own group.
    """

    detail_rows = []
    for year in range(1998, 2023, 4):
        teams = joined[joined["year"].eq(year)].copy()
        teams["title_contender_set"] = teams["team_name"].isin(CURATED_HISTORICAL_CONTENDERS)
        champion = modern_summary.loc[modern_summary["year"].eq(year), "champion"].iloc[0]
        champion_row = teams[teams["team_name"].eq(champion)].iloc[0]
        champion_group = bool(champion_row["title_contender_set"])
        group = teams[teams["title_contender_set"].eq(champion_group)]
        ccs_in_group = int(group["ccs"].sum())
        teams_in_group = int(len(group))
        perm_hit_probability = ccs_in_group / teams_in_group
        detail_rows.append(
            {
                "year": year,
                "champion": champion,
                "actual_ccs_hit": int(champion_row["ccs"]),
                "evaluable": not (year == 1998 and champion == "France"),
                "champion_in_title_contender_set": champion_group,
                "ccs_in_champion_group": ccs_in_group,
                "teams_in_champion_group": teams_in_group,
                "permutation_hit_probability": perm_hit_probability,
                "ccs_candidates": int(teams["ccs"].sum()),
                "title_contender_teams": int(teams["title_contender_set"].sum()),
                "ccs_title_contender_teams": int((teams["ccs"].eq(1) & teams["title_contender_set"]).sum()),
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    rng = np.random.default_rng(20260611)
    for scope, sample in [
        ("all_1998_2022", detail),
        ("evaluable_ex_france_1998", detail[detail["evaluable"]]),
    ]:
        probs = sample["permutation_hit_probability"].tolist()
        actual_hits = int(sample["actual_ccs_hit"].sum())
        dist = poisson_binomial_distribution(probs)
        simulated_hits = (rng.random((200_000, len(probs))) < np.array(probs)).sum(axis=1)
        summary_rows.append(
            {
                "scope": scope,
                "champions_considered": int(len(sample)),
                "actual_hits": actual_hits,
                "expected_permutation_hits": float(sum(probs)),
                "probability_ge_actual_exact": float(dist[actual_hits:].sum()),
                "probability_ge_actual_simulated": float((simulated_hits >= actual_hits).mean()),
                "simulation_runs": 200_000,
            }
        )

    summary = pd.DataFrame(summary_rows)
    detail.to_csv(DATA_DERIVED / "contender_permutation_detail.csv", index=False)
    summary.to_csv(DATA_DERIVED / "contender_permutation_summary.csv", index=False)
    return detail, summary


def build_favorite_traps(team_year: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    joined = team_year.merge(
        rankings[["year", "team_name", "team_code", "fifa_rank", "fifa_points", "ranking_official_date"]],
        on=["year", "team_name"],
        how="left",
        suffixes=("", "_rank"),
    )
    # Some FIFA naming differences are easier to recover by country code.
    missing = joined["fifa_rank"].isna()
    by_code = rankings[["year", "team_code", "fifa_rank", "fifa_points", "ranking_official_date"]]
    repaired = joined.loc[missing].drop(columns=["fifa_rank", "fifa_points", "ranking_official_date"]).merge(
        by_code, on=["year", "team_code"], how="left"
    )
    joined.loc[missing, ["fifa_rank", "fifa_points", "ranking_official_date"]] = repaired[
        ["fifa_rank", "fifa_points", "ranking_official_date"]
    ].to_numpy()

    joined["rank_bucket"] = pd.cut(
        joined["fifa_rank"],
        bins=[0, 5, 10, 15, 25, 300],
        labels=["Top 5", "6-10", "11-15", "16-25", "26+"],
    )
    joined.to_csv(DATA_DERIVED / "ccs_ranked_team_year.csv", index=False)

    traps = joined[(joined["year"].between(1998, 2022)) & (joined["fifa_rank"].le(20) & joined["ccs"].eq(0))]
    traps = traps.sort_values(["year", "fifa_rank"])[
        ["year", "team_name", "team_code", "fifa_rank", "fifa_points", "ccs", "performance", "ranking_official_date"]
    ]
    traps.to_csv(DATA_DERIVED / "favorite_traps.csv", index=False)
    headliners = traps.groupby("year", as_index=False).head(1).copy()
    headliners.to_csv(DATA_DERIVED / "favorite_trap_headliners.csv", index=False)
    powerhouses = joined[
        (joined["year"].between(1998, 2022))
        & (joined["ccs"].eq(0))
        & (joined["fifa_rank"].le(20))
        & (joined["team_name"].isin(CURATED_HISTORICAL_CONTENDERS))
        & ~((joined["year"].eq(1998)) & (joined["team_name"].eq("France")))
    ].copy()
    powerhouses = powerhouses.sort_values(["year", "fifa_rank"])[
        ["year", "team_name", "team_code", "fifa_rank", "fifa_points", "ccs", "performance", "ranking_official_date"]
    ]
    powerhouses.to_csv(DATA_DERIVED / "favorite_trap_powerhouses.csv", index=False)

    watch_2026 = joined[joined["year"].eq(2026)].sort_values("fifa_rank")
    watch_2026.to_csv(DATA_DERIVED / "ccs_2026_watchlist.csv", index=False)
    downgrade_2026 = watch_2026[
        watch_2026["ccs"].eq(0) & watch_2026["team_name"].isin(CURATED_2026_DOWNGRADE_TEAMS)
    ].copy()
    downgrade_2026.to_csv(DATA_DERIVED / "ccs_2026_downgrade_giants.csv", index=False)
    return joined


def set_chart_style() -> None:
    for font_path in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Heiti SC", "STHeiti", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d7dce5",
            "axes.labelcolor": "#172033",
            "xtick.color": "#4f5b6d",
            "ytick.color": "#4f5b6d",
            "text.color": "#172033",
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "savefig.dpi": 220,
        }
    )


def savefig(path: str) -> str:
    out = ASSETS / path
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return f"assets/{path}"


def chart_modern_funnel(modern: pd.DataFrame, lang: str = "en") -> str:
    stages = [
        ("All teams", modern["ccs"].sum(), modern["participants"].sum()),
        ("Round of 16", modern["r16_ccs"].sum(), modern["r16_total"].sum()),
        ("Quarter-finals", modern["qf_ccs"].sum(), modern["qf_total"].sum()),
        ("Semi-finals", modern["sf_ccs"].sum(), modern["sf_total"].sum()),
        ("Finalists", modern["final_ccs"].sum(), modern["final_total"].sum()),
        ("Champions", modern["champ_ccs"].sum(), modern["champ_total"].sum()),
    ]
    x = np.arange(len(stages))
    y = [a / b for _, a, b in stages]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(x, y, color="#0b5cad", linewidth=3.5, marker="o", markersize=9)
    ax.fill_between(x, y, color="#0b5cad", alpha=0.10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    labels = [s[0] for s in stages]
    if lang == "zh":
        labels = ["全部参赛队", "16强", "8强", "4强", "决赛队", "冠军"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#e8ebf1", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("CCS 覆盖率随赛事深入而上升" if lang == "zh" else "CCS coverage rises as the tournament gets deeper")
    ax.set_ylabel("该阶段球队中 CCS 占比" if lang == "zh" else "Share of teams at stage carrying CCS")
    for i, (_, a, b) in enumerate(stages):
        ax.annotate(
            f"{a}/{b}\n{a/b:.1%}",
            (i, y[i]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    return savefig(f"01_modern_funnel_{lang}.png")


def chart_historical_scope(summary: pd.DataFrame, scope: pd.DataFrame, lang: str = "en") -> str:
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    title = (
        "No champion comes from the clean non-contact bucket"
        if lang == "en"
        else "历史上没有冠军来自“干净非接触”排除池"
    )
    subtitle = (
        "If a team played both prior World Cups but had no champion/runner-up path contact, it has never won the next one."
        if lang == "en"
        else "若前两届都参赛，却没有任何冠军/亚军路径接触，历史上从未在下一届夺冠。"
    )
    ax.text(0, 4.42, title, fontsize=19, fontweight="bold", va="top")
    ax.text(0, 4.05, subtitle, fontsize=11, color="#5b6575", va="top")

    summary_by_year = summary.set_index("year").to_dict("index")
    scope_by_year = scope.set_index("year").to_dict("index")
    years = sorted(scope["year"].tolist())
    cols = 6
    tile_w, tile_h = 1.02, 0.63
    x_gap, y_gap = 0.13, 0.16
    start_y = 3.22
    for idx, year in enumerate(years):
        col = idx % cols
        row = idx // cols
        x = col * (tile_w + x_gap)
        y = start_y - row * (tile_h + y_gap)
        scope_row = scope_by_year[year]
        rec = summary_by_year.get(year)
        if rec is None:
            face, edge, status = "#eef3fa", "#b7c6da", "warm-up"
            label = "前史不足" if lang == "zh" else "setup"
            champion = ""
        elif int(rec["champion_clean_non_contact"]) == 1:
            face, edge, status = "#fff1ec", "#d15532", "violation"
            label = "反例" if lang == "zh" else "violation"
            champion = display_team(rec["champion_name"], lang)
        elif int(rec["champion_prior_participated_both"]) == 0:
            face, edge, status = "#f3f4f6", "#cfd6e1", "outside rule"
            label = "前两届未全参赛" if lang == "zh" else "not 2-for-2"
            champion = display_team(rec["champion_name"], lang)
        elif int(rec["champion_path_contact"]) == 1:
            face, edge, status = "#e8f2fb", "#0b5cad", "hit"
            label = "路径接触" if lang == "zh" else "path contact"
            champion = display_team(rec["champion_name"], lang)
        else:
            face, edge, status = "#fff8e5", "#c58a00", "review"
            label = "需复核" if lang == "zh" else "review"
            champion = display_team(rec["champion_name"], lang)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                tile_w,
                tile_h,
                boxstyle="round,pad=0.015,rounding_size=0.045",
                linewidth=1.5,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(x + 0.08, y + tile_h - 0.12, str(year), fontsize=12, fontweight="bold", va="top")
        ax.text(x + 0.08, y + 0.28, champion, fontsize=10.5, va="center", color="#172033")
        ax.text(x + 0.08, y + 0.10, label, fontsize=8.5, va="bottom", color="#5b6575")

    clean_total = int(summary["clean_non_contact_teams"].sum())
    contact_total = int(summary["path_contact_teams"].sum())
    clean_champions = int(summary["clean_non_contact_champions"].sum())
    tested = summary[summary["champion_prior_participated_both"].eq(1)]
    protected = int(tested["champion_path_contact"].sum())
    total = int(len(tested))
    note = (
        f"Prior-two participation split: {contact_total} with path contact vs {clean_total} clean non-contact; clean non-contact champions: {clean_champions}/{clean_total}."
        if lang == "en"
        else f"前两届都参赛样本：{contact_total} 个有路径接触 vs {clean_total} 个干净非接触；后者冠军：{clean_champions}/{clean_total}。"
    )
    ax.text(0, 0.18, note, fontsize=12, fontweight="bold", color="#172033")
    return savefig(f"01_historical_scope_{lang}.png")


def chart_random_benchmark(path_summary: pd.DataFrame, lang: str = "en") -> str:
    clean_total = int(path_summary["clean_non_contact_teams"].sum())
    actual_excluded_champions = int(path_summary["clean_non_contact_champions"].sum())
    random_expected = float((path_summary["clean_non_contact_teams"] / path_summary["participants"]).sum())
    random_zero_prob = float(np.prod(1 - path_summary["clean_non_contact_teams"] / path_summary["participants"]))
    labels = ["Actual rule\nclean non-contact", "Random same-size\nexclusion pool"]
    if lang == "zh":
        labels = ["真实规则\n干净非接触池", "同规模随机\n排除池"]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(2)
    bars = ax.bar(x, [actual_excluded_champions, random_expected], color=["#0b5cad", "#b2bbc8"], width=0.52)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(3.5, random_expected + 1))
    ax.set_ylabel("Champions inside exclusion bucket" if lang == "en" else "排除池中的冠军数")
    ax.set_title(
        "The exclusion rule avoids every champion; random same-size exclusion usually would not"
        if lang == "en"
        else "排除规则没有排掉任何冠军；同规模随机排除通常做不到"
    )
    ax.grid(axis="y", color="#e8ebf1")
    ax.spines[["top", "right"]].set_visible(False)
    annotations = [f"0/{clean_total}", f"{random_expected:.1f} expected"]
    if lang == "zh":
        annotations = [f"0/{clean_total}", f"期望 {random_expected:.1f}"]
    for bar, label in zip(bars, annotations):
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    note = (
        f"Random chance of excluding zero champions with the same yearly pool sizes: {random_zero_prob:.1%}"
        if lang == "en"
        else f"按每届同样规模随机排除，却 0 次排到冠军的概率：{random_zero_prob:.1%}"
    )
    ax.text(0.5, 0.34, note, ha="center", fontsize=11, color="#d15532", fontweight="bold")
    return savefig(f"02_random_benchmark_{lang}.png")


def chart_pure_ccs_random_benchmark(summary: pd.DataFrame, random_df: pd.DataFrame, lang: str = "en") -> str:
    sample = summary[summary["evaluable"].eq(1)].copy()
    actual_hits = int(sample["champ_ccs"].sum())
    total = int(len(sample))
    expected = float(random_df["expected_random_hits"].iloc[0])
    exact_prob = float(random_df["prob_random_ge_actual"].iloc[0])
    sim_prob = float(random_df["prob_random_ge_actual_simulated"].iloc[0])
    sim_runs = int(random_df["simulation_runs"].iloc[0])
    labels = ["Actual pure CCS\nchampion hits", "Random same-size\ncandidate pool"]
    if lang == "zh":
        labels = ["纯 CCS 实际\n冠军命中", "同规模随机\n候选池"]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    bars = ax.bar(np.arange(2), [actual_hits, expected], color=["#0b5cad", "#b2bbc8"], width=0.52)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(total + 1.2, actual_hits + 1))
    ax.set_ylabel("Champion hits" if lang == "en" else "冠军命中数")
    ax.set_title(
        "Pure CCS beats a same-size random candidate pool across knockout-path history"
        if lang == "en"
        else "纯 CCS 在全历史淘汰路径样本中显著高于同规模随机候选池"
    )
    ax.grid(axis="y", color="#e8ebf1")
    ax.spines[["top", "right"]].set_visible(False)
    annotations = [f"{actual_hits}/{total}", f"{expected:.1f} expected"]
    if lang == "zh":
        annotations = [f"{actual_hits}/{total}", f"期望 {expected:.1f}"]
    for bar, label in zip(bars, annotations):
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    note = (
        f"Exact random tail probability: {exact_prob:.4%}; Monte Carlo check: {small_tail_pct(sim_prob, sim_runs)}"
        if lang == "en"
        else f"精确随机尾部概率：{exact_prob:.4%}；Monte Carlo 复核：{small_tail_pct(sim_prob, sim_runs)}"
    )
    ax.text(0.5, total + 0.45, note, ha="center", fontsize=11, color="#d15532", fontweight="bold")
    return savefig(f"02a_pure_ccs_random_benchmark_{lang}.png")


def performance_rank(value: str) -> int:
    order = {
        "final": 5,
        "third-place match": 4,
        "quarter-finals": 3,
        "round of 16": 2,
        "group stage": 1,
        "not yet known": 0,
    }
    return order.get(str(value), 0)


def chart_favorite_traps(traps: pd.DataFrame, lang: str = "en") -> str:
    grouped_rows = []
    for year, rows in traps.sort_values(["year", "fifa_rank"]).groupby("year"):
        names = [display_team(name, lang) for name in rows["team_name"].tolist()]
        best = rows.sort_values("performance", key=lambda s: s.map(performance_rank), ascending=False).iloc[0]
        grouped_rows.append(
            {
                "year": int(year),
                "teams": "  ·  ".join(names),
                "count": len(rows),
                "best": f"{display_team(best['team_name'], lang)} · {display_performance(best['performance'], lang)}",
            }
        )

    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(grouped_rows) + 1.25)
    ax.axis("off")
    title = (
        "Recognizable contenders CCS would have downgraded before kickoff"
        if lang == "en"
        else "赛前会被 CCS 降权的一众强队/豪门"
    )
    subtitle = (
        "Curated from FIFA Top-20 non-CCS teams, keeping only sides with a plausible title narrative."
        if lang == "en"
        else "从 FIFA Top 20 且非 CCS 的审计池中筛出：只保留有冠军叙事的强队/豪门。"
    )
    ax.text(0.0, len(grouped_rows) + 1.02, title, fontsize=19, fontweight="bold", va="top")
    ax.text(0.0, len(grouped_rows) + 0.63, subtitle, fontsize=11, color="#5b6575", va="top")
    ax.text(0.02, len(grouped_rows) + 0.10, "Year" if lang == "en" else "年份", fontsize=10, color="#5b6575", fontweight="bold")
    ax.text(0.18, len(grouped_rows) + 0.10, "Non-CCS heavyweights" if lang == "en" else "非 CCS 豪强", fontsize=10, color="#5b6575", fontweight="bold")
    ax.text(0.73, len(grouped_rows) + 0.10, "Deepest run" if lang == "en" else "最深成绩", fontsize=10, color="#5b6575", fontweight="bold")

    for idx, row in enumerate(grouped_rows):
        y = len(grouped_rows) - idx - 0.55
        face = "#f8fafc" if idx % 2 == 0 else "#ffffff"
        ax.add_patch(Rectangle((0, y - 0.34), 1, 0.62, facecolor=face, edgecolor="#e6eaf0", linewidth=0.8))
        ax.text(0.02, y, str(row["year"]), fontsize=15, fontweight="bold", va="center", color="#0b5cad")
        ax.text(0.18, y, row["teams"], fontsize=12.2, va="center", color="#172033")
        ax.text(0.73, y, row["best"], fontsize=11.2, va="center", color="#d15532", fontweight="bold")
        count_text = f"{row['count']} teams" if lang == "en" else f"{row['count']} 队"
        ax.text(0.95, y, count_text, fontsize=10, va="center", ha="right", color="#5b6575")
    return savefig(f"03_favorite_traps_{lang}.png")


def chart_2026_watchlist(downgrade: pd.DataFrame, lang: str = "en") -> str:
    cards = downgrade.sort_values("fifa_rank").copy()
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.75)
    ax.axis("off")
    title = (
        "2026: big-name contenders outside the champion-chain pool"
        if lang == "en"
        else "2026：冠军链候选池之外的响当当强队"
    )
    subtitle = (
        "These are not weak teams. CCS says they need extra evidence to be re-admitted as title picks."
        if lang == "en"
        else "这不是说它们弱，而是说若要押冠军，需要额外证据把它们重新放回候选池。"
    )
    ax.text(0, 1.68, title, fontsize=19, fontweight="bold", va="top")
    ax.text(0, 1.43, subtitle, fontsize=11, color="#5b6575", va="top")
    for idx, r in enumerate(cards.itertuples()):
        x = idx
        y = 0.13
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.08, y),
                0.84,
                1.08,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.4,
                edgecolor="#d15532",
                facecolor="#fff4ef",
            )
        )
        ax.text(x + 0.18, y + 0.88, f"#{int(r.fifa_rank)}", fontsize=14, fontweight="bold", color="#d15532")
        ax.text(x + 0.18, y + 0.58, display_team(r.team_name, lang), fontsize=17, fontweight="bold", color="#172033")
        ax.text(
            x + 0.18,
            y + 0.27,
            "Non-CCS" if lang == "en" else "非 CCS",
            fontsize=11,
            color="#5b6575",
            fontweight="bold",
        )
    return savefig(f"04_2026_watchlist_{lang}.png")


def chart_contender_permutation(summary: pd.DataFrame, lang: str = "en") -> str:
    row = summary[summary["scope"].eq("evaluable_ex_france_1998")].iloc[0]
    actual = float(row["actual_hits"])
    expected = float(row["expected_permutation_hits"])
    total = int(row["champions_considered"])
    tail = float(row["probability_ge_actual_exact"])
    labels = ["CCS actual", "Strong-label permutation\nexpected"]
    if lang == "zh":
        labels = ["CCS 实际", "强队标签置换\n期望"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(labels, [actual, expected], color=["#0b5cad", "#9aa6b2"], width=0.55)
    ax.set_ylim(0, total + 0.8)
    ax.set_ylabel(f"Champion hits out of {total}" if lang == "en" else f"{total} 个可判定冠军中的命中数")
    ax.set_title(
        "CCS still clears a stronger 'famous team' control"
        if lang == "en"
        else "控制“强队标签”后，CCS 仍高于置换基准"
    )
    ax.grid(axis="y", color="#e8ebf1")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, [actual, expected]):
        ax.annotate(
            f"{value:.1f}/{total}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    note = (
        f"Permutation reaches {int(actual)}/{total}: {tail:.1%}"
        if lang == "en"
        else f"置换达到 {int(actual)}/{total} 的概率：{tail:.1%}"
    )
    ax.text(0.5, total + 0.25, note, ha="center", fontsize=10, color="#5b6575")
    return savefig(f"05_contender_permutation_{lang}.png")


def chart_rank_bucket(joined: pd.DataFrame, lang: str = "en") -> str:
    hist = joined[joined["year"].between(1998, 2022)].copy()
    bucket = hist.groupby(["rank_bucket", "ccs"], observed=True).size().unstack(fill_value=0)
    bucket = bucket.reindex(["Top 5", "6-10", "11-15", "16-25", "26+"])
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(bucket))
    width = 0.36
    ax.bar(x - width / 2, bucket.get(1, 0), width, label="CCS", color="#0b5cad")
    ax.bar(x + width / 2, bucket.get(0, 0), width, label="Non-CCS", color="#9aa6b2")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket.index)
    ax.set_ylabel("球队-届次" if lang == "zh" else "Team-tournaments")
    ax.set_title("CCS 与强队属性重叠，但并不等同于 FIFA 排名" if lang == "zh" else "CCS overlaps with strength, but it is not just the FIFA ranking table")
    ax.grid(axis="y", color="#e8ebf1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(["CCS", "非 CCS"] if lang == "zh" else ["CCS", "Non-CCS"], frameon=False)
    return savefig(f"05_rank_bucket_control_{lang}.png")


def pct(x: float) -> str:
    return f"{x:.1%}"


def small_tail_pct(x: float, runs: int, digits: int = 4) -> str:
    if x == 0:
        return f"<{1 / runs:.{digits}%}"
    return f"{x:.{digits}%}"


def table_html(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, limit: int | None = None) -> str:
    view = df[columns].head(limit) if limit else df[columns]
    view = view.rename(columns=rename or {})
    return view.to_html(index=False, classes="data-table", border=0, escape=False)


def report_css() -> str:
    return """
    :root { --ink:#111827; --muted:#5b6575; --line:#e6eaf0; --blue:#0b5cad; --red:#d15532; --bg:#f6f8fb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color:var(--ink); background:#fff; line-height:1.55; }
    .page { max-width: 1080px; margin: 0 auto; padding: 44px 48px 72px; }
    .eyebrow { color:var(--blue); text-transform: uppercase; letter-spacing:.12em; font-size:12px; font-weight:800; }
    h1 { font-size: 38px; line-height:1.12; margin: 8px 0 14px; letter-spacing:0; }
    h2 { font-size: 23px; margin: 42px 0 12px; border-top: 1px solid var(--line); padding-top: 24px; }
    h3 { font-size: 17px; margin: 24px 0 8px; }
    p { margin: 0 0 13px; }
    .subhead { color:var(--muted); font-size:16px; max-width: 880px; }
    .summary { background:var(--bg); border:1px solid var(--line); padding:22px 24px; margin:28px 0; border-radius:12px; }
    .summary h2 { border:0; padding:0; margin:0 0 12px; }
    .summary ul { margin:0; padding-left:20px; }
    .summary li { margin: 9px 0; }
    .lens-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:14px; margin:20px 0 24px; }
    .lens { border:1px solid var(--line); border-radius:10px; padding:17px 18px; background:#fff; }
    .lens .tag { color:var(--blue); font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
    .lens .metric { font-size:26px; font-weight:800; margin:7px 0 4px; color:var(--ink); }
    .lens p { color:var(--muted); font-size:13px; margin:0; }
    .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin: 24px 0 30px; }
    .kpi { border:1px solid var(--line); border-radius:10px; padding:14px 15px; background:#fff; }
    .kpi .value { font-size:25px; font-weight:800; color:var(--blue); }
    .kpi .label { color:var(--muted); font-size:12px; margin-top:3px; }
    .figure { margin: 22px 0 26px; }
    .figure img { width:100%; border:1px solid var(--line); border-radius:10px; }
    .caption { color:var(--muted); font-size:12px; margin-top:7px; }
    .callout { border-left: 4px solid var(--blue); background:#f4f8ff; padding:14px 16px; margin:18px 0; }
    .warn { border-left-color: var(--red); background:#fff6f2; }
    .data-table { width:100%; border-collapse: collapse; margin: 16px 0 24px; font-size:13px; }
    .data-table th { background:#15233a; color:#fff; text-align:left; padding:9px; }
    .data-table td { border-bottom:1px solid var(--line); padding:8px 9px; vertical-align:top; }
    .data-table tr:nth-child(even) td { background:#fafbfe; }
    .footer { color:var(--muted); font-size:12px; margin-top:40px; border-top:1px solid var(--line); padding-top:16px; }
    @media print { .page { padding: 24px 34px; } .figure img { break-inside: avoid; } h2 { break-after: avoid; } }
    @page { size: A4; margin: 14mm 12mm; }
    """


def prepare_tables(context: dict, lang: str) -> tuple[str, str, str]:
    top_traps = context["trap_powerhouses"].copy()
    top_traps["fifa_rank"] = top_traps["fifa_rank"].map(lambda x: f"#{int(x)}")
    downgrade_2026 = context["downgrade_2026"].copy()
    downgrade_2026["fifa_rank"] = downgrade_2026["fifa_rank"].map(lambda x: f"#{int(x)}")
    watch = context["watch_2026"].copy()
    watch = watch[watch["fifa_rank"].le(16)].copy()
    watch["fifa_rank"] = watch["fifa_rank"].map(lambda x: f"#{int(x)}")
    watch["ccs_source_years"] = watch["ccs_source_years"].replace("", "none").fillna("none")
    if lang == "zh":
        top_traps["team_name"] = top_traps["team_name"].map(lambda x: display_team(x, "zh"))
        top_traps["performance"] = top_traps["performance"].map(lambda x: display_performance(x, "zh"))
        downgrade_2026["team_name"] = downgrade_2026["team_name"].map(lambda x: display_team(x, "zh"))
        downgrade_2026["降权理由"] = "排名/声望强，但 2018/2022 无 CCS 来源"
        watch["team_name"] = watch["team_name"].map(lambda x: display_team(x, "zh"))
        top_traps = top_traps.rename(columns={"team_name": "球队", "performance": "最终成绩"})
        watch["CCS状态"] = np.where(watch["ccs"].eq(1), "CCS 候选", "非 CCS，需降权")
        traps_html = table_html(
            top_traps,
            ["year", "球队", "team_code", "fifa_rank", "最终成绩", "ranking_official_date"],
            {"year": "年份", "team_code": "代码", "fifa_rank": "赛前排名", "ranking_official_date": "排名日期"},
            28,
        )
        watch_html = table_html(
            watch,
            ["fifa_rank", "team_name", "team_code", "CCS状态", "ccs_source_years"],
            {"fifa_rank": "排名", "team_name": "球队", "team_code": "代码", "ccs_source_years": "CCS 来源届次"},
            16,
        )
        downgrade_2026_html = table_html(
            downgrade_2026,
            ["fifa_rank", "team_name", "team_code", "fifa_points", "降权理由"],
            {"fifa_rank": "排名", "team_name": "球队", "team_code": "代码", "fifa_points": "FIFA积分"},
        )
    else:
        top_traps = top_traps.rename(columns={"team_name": "Team", "performance": "Final outcome"})
        downgrade_2026["Downgrade reason"] = "Rank/reputation strong, but no CCS source from 2018/2022"
        watch["CCS status"] = np.where(watch["ccs"].eq(1), "CCS candidate", "Non-CCS downgrade")
        traps_html = table_html(
            top_traps,
            ["year", "Team", "team_code", "fifa_rank", "Final outcome", "ranking_official_date"],
            {"year": "Year", "team_code": "Code", "fifa_rank": "FIFA rank", "ranking_official_date": "Ranking date"},
            28,
        )
        watch_html = table_html(
            watch,
            ["fifa_rank", "team_name", "team_code", "CCS status", "ccs_source_years"],
            {"team_name": "Team", "team_code": "Code", "ccs_source_years": "CCS source World Cup"},
            16,
        )
        downgrade_2026_html = table_html(
            downgrade_2026,
            ["fifa_rank", "team_name", "team_code", "fifa_points", "Downgrade reason"],
            {"team_name": "Team", "team_code": "Code", "fifa_points": "FIFA points"},
        )
    return traps_html, watch_html, downgrade_2026_html


def historical_misses_html(summary: pd.DataFrame, lang: str) -> str:
    misses = summary[(summary["evaluable"].eq(1)) & (summary["champ_ccs"].eq(0))].copy()
    misses["champion_display"] = misses["champion"].map(lambda x: display_team(x, lang))
    if lang == "zh":
        miss_notes = {
            1978: "1970 未进正赛；1974 为第二阶段小组，不是连续两届淘汰赛强队。",
            1982: "1974 小组赛；1978 进三四名赛，但不在 1974/1978 冠军链来源中。",
            1998: "法国缺席 1990/1994 正赛，是明确的无前史主办国例外。",
        }
        misses["note"] = misses["year"].map(miss_notes)
        return table_html(
            misses,
            ["year", "champion_display", "prior_windows", "ccs_candidates", "participants", "note"],
            {
                "year": "年份",
                "champion_display": "冠军",
                "prior_windows": "回看窗口",
                "ccs_candidates": "CCS候选",
                "participants": "参赛队",
                "note": "为什么不是简单“连续淘汰赛强队”故事",
            },
        )
    miss_notes = {
        1978: "Argentina missed 1970; 1974 was a second-group-stage run, not two straight knockout-path validations.",
        1982: "Italy exited in the 1974 group stage; 1978 reached the third-place match but was not in the 1974/1978 champion-chain source.",
        1998: "France missed both 1990 and 1994, making it an explicit no-prior-history host exception.",
    }
    misses["note"] = misses["year"].map(miss_notes)
    return table_html(
        misses,
        ["year", "champion_display", "prior_windows", "ccs_candidates", "participants", "note"],
        {
            "year": "Year",
            "champion_display": "Champion",
            "prior_windows": "Lookback window",
            "ccs_candidates": "CCS pool",
            "participants": "Teams",
            "note": "Why this is not a simple repeated-knockout-team story",
        },
    )


def path_exclusion_examples_html(summary: pd.DataFrame, lang: str) -> str:
    examples = summary[
        summary["year"].isin([1978, 1982, 1998, 2022])
    ].copy()
    examples["champion_display"] = examples["champion_name"].map(lambda x: display_team(x, lang))
    if lang == "zh":
        zh_evidence = {
            1978: "1970 未进决赛圈；因此不适用排除命题。",
            1982: "1978 第二阶段小组输给最终亚军荷兰。",
            1998: "1990 和 1994 均未进决赛圈；因此不适用排除命题。",
            2022: "2014 决赛输给冠军德国；2018 16强输给冠军法国。",
        }
        examples["状态"] = np.where(
            examples["champion_prior_participated_both"].eq(0),
            "前两届未都参加，不适用排除命题",
            "前两届都参加，且有冠军/亚军路径接触",
        )
        examples["证据"] = examples["year"].map(zh_evidence)
        return table_html(
            examples,
            ["year", "champion_display", "prior_windows", "状态", "证据"],
            {"year": "年份", "champion_display": "冠军", "prior_windows": "前两届窗口"},
        )
    examples["Status"] = np.where(
        examples["champion_prior_participated_both"].eq(0),
        "Not covered by exclusion rule: did not play both prior finals",
        "Played both prior finals and had champion/runner-up path contact",
    )
    examples["Evidence"] = examples["champion_contact_reasons"].replace(
        "", "No path contact, but did not play both prior finals"
    )
    return table_html(
        examples,
        ["year", "champion_display", "prior_windows", "Status", "Evidence"],
        {"year": "Year", "champion_display": "Champion", "prior_windows": "Prior two World Cups"},
    )


def render_report_en(context: dict) -> str:
    css = report_css()
    traps_html, watch_html, downgrade_2026_html = prepare_tables(context, "en")
    examples_html = path_exclusion_examples_html(context["path_summary"], "en")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>World Cup CCS Investment Report</title>
<style>{css}</style>
</head>
<body>
<main class="page">
<div class="eyebrow">World Cup predictor research memo</div>
<h1>Champion-chain signal: a pre-tournament filter for real title contenders</h1>
<p class="subhead">A reproducible backtest with two lenses: pure CCS as a broad champion-chain candidate pool, and a stricter prior-two-participation exclusion rule for teams with no champion/runner-up path contact.</p>

<section class="summary">
<h2>Executive Summary</h2>
<ul>
<li><strong>The report now carries two explicit lenses.</strong> Lens A is pure CCS: no prior-participation filter, just whether a team is in the prior two champion-chain pools. Lens B adds the stricter question: among teams that played both prior World Cups, did any clean non-contact team ever win next?</li>
<li><strong>Pure CCS is the broad candidate-pool claim.</strong> Across the full knockout-path history, pure CCS covers {context['historical_champions_short']} evaluable champions; a same-size random candidate pool would expect {context['historical_random_expected']} hits and reach that result with probability {context['historical_random_prob']}.</li>
<li><strong>The prior-two-participation lens is the sharper exclusion claim.</strong> From 1938 to 2022, {context['path_prior_both_total']} team-tournaments played both prior World Cups; {context['path_contact_total']} had champion/runner-up path contact, while {context['clean_non_contact_teams']} had none. That clean non-contact bucket produced {context['clean_non_contact_champions']} next champions.</li>
<li><strong>This reframes the historical “misses.”</strong> Argentina 1978 and France 1998 are not counterexamples because they did not play both prior World Cups. Italy 1982 is not a counterexample because it played both, and in 1978 it lost to runner-up Netherlands in the second group stage.</li>
<li><strong>The random benchmark is just probability multiplication, and that is the right first test.</strong> If each year randomly excluded the same number of teams as the clean non-contact bucket, the expected number of excluded champions is {context['path_random_expected']}; the exact probability of excluding zero champions is {context['path_random_zero_prob']}.</li>
<li><strong>The Monte Carlo simulation is a sanity check, not a black box.</strong> A 200,000-run random redraw gives {context['path_random_zero_sim_prob']}, close to the exact probability. The report uses the exact value as the headline and keeps simulation as a reproducibility check.</li>
<li><strong>The most useful reader experience is the pre-tournament downgrade list.</strong> From 1998 onward, a long list of recognizable contenders were highly ranked but non-CCS at kickoff. Many still advanced, including finalists, but none won in the modern sample.</li>
<li><strong>The mechanism is partly strength, but more specific than strength.</strong> The rule asks a narrower question than ranking: did this team recently prove it could survive, win, or be eliminated by finalist-level opposition in the World Cup cycle?</li>
</ul>
</section>

<div class="kpis">
<div class="kpi"><div class="value">{context['historical_champions_short']}</div><div class="label">Pure CCS historical champion coverage</div></div>
<div class="kpi"><div class="value">{context['historical_random_prob']}</div><div class="label">Random same-size pool reaches pure CCS</div></div>
<div class="kpi"><div class="value">{context['clean_non_contact_champions_short']}</div><div class="label">Strict clean non-contact champions</div></div>
<div class="kpi"><div class="value">{context['path_contact_vs_clean']}</div><div class="label">Path-contact vs clean among prior-two participants</div></div>
</div>

<h2>1. Two lenses, two uses</h2>
<div class="lens-grid">
<div class="lens"><div class="tag">Lens A · pure CCS</div><div class="metric">{context['historical_champions_short']} champions</div><p>No requirement that a team played both prior tournaments. This is the broader candidate-pool lens: did the next champion appear in either of the previous two champion-chain sets?</p></div>
<div class="lens"><div class="tag">Lens B · prior-two participation</div><div class="metric">{context['clean_non_contact_champions_short']} champions</div><p>Only teams that played both prior World Cups enter the stricter denominator. If they still had no champion/runner-up path contact, the historical winner count is zero.</p></div>
</div>
<p><strong>These lenses answer different questions.</strong> Pure CCS is better for building a candidate pool. The prior-two-participation rule is better for explaining why certain apparently strong teams should be downgraded before kickoff.</p>

<h2>2. Lens A: pure CCS, without the prior-participation filter</h2>
<p><strong>Pure CCS asks the broad question first.</strong> Ignore whether the team played both previous World Cups. If it is in the champion-chain set from either of the prior two knockout-path tournaments, it is a CCS candidate. On the full historical knockout-path sample, this covers {context['historical_champions_short']} evaluable champions.</p>
<div class="figure"><img src="{context['fig_pure_random_en']}" alt="Pure CCS random benchmark"><div class="caption">Pure CCS is compared against a same-size random candidate pool for each tournament. Exact probability and Monte Carlo simulation are both reported.</div></div>
<p><strong>Modern standard-format evidence points in the same direction.</strong> From 1986 to 2022, CCS covers {context['champ_all']} champions overall, or {context['champ_evaluable']} after treating France 1998 as a no-prior-history exception.</p>
<div class="figure"><img src="{context['fig_funnel_en']}" alt="CCS modern stage funnel"><div class="caption">Modern era is 1986-2022. Champion coverage is 9/10 on the all-champion denominator and 9/9 after removing France 1998, which had no prior-two-World-Cup finals history.</div></div>

<h2>3. Lens B: the strict exclusion rule has produced zero champions</h2>
<p><strong>The rule is intentionally narrow.</strong> A team is put in the exclusion bucket only when it played both prior World Cups and, across all knockout or championship-phase appearances in those two tournaments, it neither won the tournament nor lost to that tournament's champion or runner-up. In 1974, 1978, and 1982, the second group stage is treated as a championship phase because those formats did not use a modern round-of-16 bracket.</p>
<p><strong>The prior-two-participation gate is deliberately strict, so the right comparison is inside that same gate.</strong> Across the 1938-2022 target sample, there are {context['path_team_tournaments_total']} team-tournaments. Only {context['path_prior_both_total']} played both prior World Cups; among those, {context['path_contact_total']} had champion/runner-up path contact and {context['clean_non_contact_teams']} did not.</p>
<p><strong>The 65 count is team-tournaments, not unique teams.</strong> It is the clean side of a {context['path_contact_total']} vs {context['clean_non_contact_teams']} split among teams that passed the strict prior-participation gate, not a hand-picked list of isolated cases. If counted more literally, {context['path_lost_to_finalist_total']} of the 177 lost to a champion or runner-up in the prior two tournaments; {context['path_own_champion_total']} had their own prior title; {context['path_contact_overlap_total']} had both, so the union is {context['path_contact_total']}.</p>
<div class="figure"><img src="{context['fig_history_en']}" alt="Historical path exclusion scope"><div class="caption">Blue tiles show champions that had prior path contact after playing both prior tournaments. Gray tiles show champions outside the rule because they did not play both prior tournaments. There are no red violation tiles.</div></div>
<p><strong>The apparent exceptions disappear under this stricter definition.</strong> Argentina 1978 did not play the 1970 finals. France 1998 did not play either 1990 or 1994. Italy 1982 did play both prior tournaments, but in 1978 it lost to the eventual runner-up, the Netherlands, during the second group stage.</p>
{examples_html}

<h2>4. Two simulations: pure candidate-pool hit and strict exclusion miss</h2>
<p><strong>Both benchmarks use the same philosophy: preserve each year's pool size, randomize the labels, then compare the observed result.</strong> For pure CCS, the event is “random candidate pool hits at least as many champions as CCS.” For the strict lens, the event is “random exclusion pool avoids every champion.”</p>
<p><strong>The pure CCS random benchmark is a hit-rate test.</strong> Same-size random pools would expect {context['historical_random_expected']} champion hits across the full historical knockout-path sample; reaching {context['historical_champions_short']} has exact probability {context['historical_random_prob']} and Monte Carlo probability {context['historical_random_sim_prob']}.</p>
<p><strong>The strict exclusion benchmark is an exclusion test.</strong> The script redraws same-size random exclusion pools 200,000 times. The simulated probability of excluding zero champions is {context['path_random_zero_sim_prob']}, while the exact probability is {context['path_random_zero_prob']}.</p>
<div class="figure"><img src="{context['fig_random_en']}" alt="Random benchmark chart"><div class="caption">The rule excludes {context['clean_non_contact_teams']} team-tournaments and zero champions. A same-size random exclusion pool would exclude {context['path_random_expected']} champions in expectation.</div></div>
<p><strong>The stronger control asks whether CCS is merely a famous-team label.</strong> This simulation is intentionally limited to 1998-2022, where the report has a consistent pre-tournament FIFA ranking layer and a defensible modern title-contender set. For each tournament, we preserve the number of CCS labels inside the traditional title-contender set and outside it, then randomly permute the labels inside those two groups. Excluding the explicit France 1998 no-prior-history exception, CCS hit 6/6 evaluable champions; the strong-label permutation expects {context['strong_sim_expected']}/6, and reaches 6/6 with probability {context['strong_sim_prob']}.</p>
<div class="figure"><img src="{context['fig_sim_en']}" alt="Strong-team permutation simulation"><div class="caption">This is deliberately not a FIFA-ranking simulation. It controls for the broader fact that CCS often overlaps with obvious football powers, then asks whether the specific champion-chain label still carries information.</div></div>
<div class="callout warn"><strong>Interpretation discipline:</strong> these simulations support CCS as a credible screening heuristic. They do not prove it beats Elo, betting odds, or a full multivariate model. The next bar is incremental value versus those stronger baselines.</div>

<h2>5. The pre-tournament experience: recognizable favorites CCS would downgrade</h2>
<p><strong>This is the most intuitive way to use the method.</strong> Before kickoff, a team can be highly ranked, historically recognizable, and still lack a recent champion-chain connection. The main exhibit is curated from a Top-20 non-CCS audit pool to show the teams a modern audience would naturally treat as title-relevant: Argentina, Germany, England, Spain, Portugal, Netherlands, Belgium, Colombia, Croatia, and Uruguay.</p>
<div class="figure"><img src="{context['fig_traps_en']}" alt="Recognizable non-CCS title contenders"><div class="caption">Curated from qualified teams that were FIFA Top 20 and non-CCS at kickoff. The full mechanical Top-20 audit table is retained in data/derived/favorite_traps.csv; the curated list is retained in data/derived/favorite_trap_powerhouses.csv.</div></div>
{traps_html}
<p><strong>The pattern is useful but not absolute.</strong> Non-CCS strong teams can go deep: 2002 Germany, 2010 Netherlands, and 2014 Argentina reached finals. The historical point is narrower and stronger: in the modern sample, the champion almost always came from the CCS side of the field.</p>

<h2>6. 2026 application: separate rank strength from champion-chain strength</h2>
<p><strong>The 2026 view is a live-use case, not a backtest result.</strong> The ranking snapshot is frozen at FIFA's June 11, 2026 official ranking and the qualified-team list is from FIFA's 2026 season endpoint. The headline application is clear: Spain, Portugal, Brazil, Germany, and Colombia are rank-strong, reputation-strong, but non-CCS before kickoff.</p>
{downgrade_2026_html}
<p>The broader watchlist below keeps the full top-ranked context visible, so the downgrade call is not hidden inside a hand-picked list.</p>
<div class="figure"><img src="{context['fig_2026_en']}" alt="2026 ranked non-CCS contenders"><div class="caption">Five high-reputation 2026 qualifiers that are outside the CCS pool before kickoff. The table below keeps the broader top-ranked context.</div></div>
{watch_html}

<h2>7. Why this happens: strength matters, but path matters too</h2>
<p><strong>CCS partly captures strength, and that should be acknowledged.</strong> Champions, finalists, and teams beaten by finalists are usually strong teams. A signal built from those events will naturally overlap with FIFA ranking, odds, and historical reputation.</p>
<p><strong>But CCS is not simply 'teams that often qualify' or 'teams that are highly ranked.'</strong> It requires a specific recent relationship to the champion path: either winning the World Cup or being knocked out by a finalist. A team can be ranked highly, qualify regularly, or have reached a quarter-final and still be non-CCS if it did not touch that path.</p>
<p><strong>The proposed mechanism is tournament-cycle validation.</strong> Recent champion-chain contact is a proxy for having already met World Cup knockout intensity against finalist-level opposition. It is not causal proof; it is a compact historical state variable that seems especially relevant for champions rather than finalists.</p>

<h2>Recommended Next Steps</h2>
<ol>
<li><strong>Use CCS as the first-layer title filter.</strong> Start the champion pool with CCS candidates, then require extra evidence to re-admit non-CCS teams.</li>
<li><strong>Add a stronger model benchmark.</strong> Compare CCS against Elo, odds-implied probabilities, FIFA ranking, and a combined model to test incremental value.</li>
<li><strong>Track 2026 without moving the goalposts.</strong> Freeze the pre-tournament CCS/ranking table and evaluate it only after the tournament is complete.</li>
</ol>

<h2>Further Questions</h2>
<ul>
<li>Does CCS add predictive lift after controlling for Elo or betting odds?</li>
<li>Is a one-, two-, or three-tournament lookback optimal once tested out of sample?</li>
<li>Should hosts, major squad-cycle upgrades, and no-prior-history teams receive a separate override flag rather than being treated as ordinary non-CCS teams?</li>
</ul>

<h2>Caveats And Assumptions</h2>
<ul>
<li>The headline exclusion test uses team-tournaments, not unique national teams. A country can appear multiple times if it satisfies the rule in multiple World Cup cycles.</li>
<li>1950 remains in the target-year audit because the rule evaluates whether the next champion had prior-two-World-Cup path contact; the target tournament itself does not need a knockout bracket for that question.</li>
<li>For 1974, 1978, and 1982, second group stage matches are treated as championship-phase matches because those tournaments used hybrid formats rather than a modern bracket.</li>
<li>The random benchmark is an exact same-size random exclusion calculation. The Monte Carlo simulation is only a reproducibility check of that arithmetic.</li>
<li>The strong-team permutation simulation remains a modern 1998-2022 control, not a full-history simulation, because it depends on consistent modern ranking context and curated title-contender labels.</li>
<li>FIFA ranking is used lightly as a public audit proxy; odds and Elo would be better next-step baselines.</li>
<li>2026 outcomes are not used in the 2026 watchlist. The ranking date is frozen at June 11, 2026.</li>
</ul>

<div class="footer">
Generated by scripts/build_report.py. Sources: Fjelstul World Cup Database; FIFA API ranking schedules, rankings, and 2026 qualified-team endpoint. Original corrected CCS report is preserved in reports/pdf/original_ccs_report_corrected.pdf.
</div>
</main>
</body>
</html>"""


def render_report_zh(context: dict) -> str:
    css = """
    """ + report_css() + """
    body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", Arial, sans-serif; }
    """
    traps_html, watch_html, downgrade_2026_html = prepare_tables(context, "zh")
    examples_html = path_exclusion_examples_html(context["path_summary"], "zh")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>世界杯 CCS 投研报告</title>
<style>{css}</style>
</head>
<body>
<main class="page">
<div class="eyebrow">世界杯预测研究备忘录</div>
<h1>冠军链信号：一个赛前冠军候选过滤器</h1>
<p class="subhead">本报告同时保留两个口径：纯 CCS 作为宽口径冠军候选池；“前两届都参赛”作为更严格的赛前排除规则。</p>

<section class="summary">
<h2>执行摘要</h2>
<ul>
<li><strong>报告现在明确分成两个口径。</strong> 口径 A 是纯 CCS：不要求前两届都参赛，只看球队是否在前两届冠军链候选池里。口径 B 是更严格的“前两届都参赛”排除规则：若一支队参赛履历完整，但没有任何冠亚军路径接触，历史上没有下一届夺冠。</li>
<li><strong>纯 CCS 是宽口径候选池结论。</strong> 在全历史淘汰路径样本里，纯 CCS 覆盖 {context['historical_champions_short']} 个可判定冠军；同规模随机候选池期望命中 {context['historical_random_expected']} 个冠军，达到该结果的概率是 {context['historical_random_prob']}。</li>
<li><strong>前两届参赛口径是更锋利的排除结论。</strong> 1938-2022 年间，{context['path_prior_both_total']} 个球队-届次满足“前两届都参赛”；其中 {context['path_contact_total']} 个有冠军/亚军路径接触，{context['clean_non_contact_teams']} 个没有。后者成为下一届冠军的次数是 {context['clean_non_contact_champions']}。</li>
<li><strong>这会重新解释所谓历史漏点。</strong> 1978 阿根廷和 1998 法国不是反例，因为它们前两届没有都正常参加决赛圈。1982 意大利也不是反例，因为它前两届都参赛，且 1978 年第二阶段小组输给了当届亚军荷兰。</li>
<li><strong>随机基准本质就是概率乘法，而且这是正确的第一道检验。</strong> 如果每届随机排除与 clean non-contact 池同样数量的球队，期望会误排 {context['path_random_expected']} 个冠军；精确算出来，随机排除却 0 次排到冠军的概率是 {context['path_random_zero_prob']}。</li>
<li><strong>Monte Carlo simulation 只是复核，不是黑箱。</strong> 20 万次随机重抽得到 {context['path_random_zero_sim_prob']}，与精确概率接近。报告用精确值作 headline，把 simulation 作为可复现 sanity check。</li>
<li><strong>最有体感的用法，是赛前热门队降权清单。</strong> 1998 年以来，一众排名高、名气大、今天读者也会认为有冠军叙事的强队，在开赛前并非 CCS。它们并不一定弱，甚至可能进决赛，但现代样本里没有最终夺冠。</li>
<li><strong>机制上，它有强队效应，但比“强队”更窄。</strong> 它问的不是球队是否有名、排名是否高，而是它最近两届是否已经在世界杯周期里被冠军级或亚军级对手验证过。</li>
</ul>
</section>

<div class="kpis">
<div class="kpi"><div class="value">{context['historical_champions_short']}</div><div class="label">纯 CCS 全历史冠军覆盖</div></div>
<div class="kpi"><div class="value">{context['historical_random_prob']}</div><div class="label">同规模随机达到纯 CCS</div></div>
<div class="kpi"><div class="value">{context['clean_non_contact_champions_short']}</div><div class="label">严格无接触池冠军数</div></div>
<div class="kpi"><div class="value">{context['path_contact_vs_clean']}</div><div class="label">前两届参赛样本：路径接触 vs 无接触</div></div>
</div>

<h2>1. 两个口径，两种用法</h2>
<div class="lens-grid">
<div class="lens"><div class="tag">口径 A · 纯 CCS</div><div class="metric">{context['historical_champions_short']} 个冠军</div><p>不要求球队前两届都参赛。只问下一届冠军是否出现在前两届任一冠军链候选池里，用来构建宽口径候选池。</p></div>
<div class="lens"><div class="tag">口径 B · 前两届都参赛</div><div class="metric">{context['clean_non_contact_champions_short']} 个冠军</div><p>只看前两届都参赛的球队。如果它仍然没有冠亚军路径接触，就进入严格排除池；历史冠军数为零。</p></div>
</div>
<p><strong>两者回答的问题不一样。</strong> 纯 CCS 适合做“候选池”；前两届参赛规则适合做“排除/降权”。这样既不因为门槛过严丢掉法国 1998 这类无前史样本，也能保留最锋利的排除结论。</p>

<h2>2. 口径 A：不考虑前两届都参赛的纯 CCS</h2>
<p><strong>纯 CCS 先回答宽口径问题。</strong> 不管一支队前两届是否都参加，只要它出现在前两届任一冠军链候选池里，就算 CCS 候选。在全历史淘汰路径样本中，这个口径覆盖 {context['historical_champions_short']} 个可判定冠军。</p>
<div class="figure"><img src="{context['fig_pure_random_zh']}" alt="Pure CCS random benchmark"><div class="caption">纯 CCS 与每届同规模随机候选池对比；图中同时给出精确概率和 Monte Carlo 复核。</div></div>
<p><strong>现代标准赛制样本也支持这个方向。</strong> 1986-2022 年，纯 CCS 覆盖全部冠军口径 {context['champ_all']}；如果剔除 1998 法国这个无前史例外，则为 {context['champ_evaluable']}。</p>
<div class="figure"><img src="{context['fig_funnel_zh']}" alt="CCS modern stage funnel"><div class="caption">现代时代定义为 1986-2022。全部冠军口径为 9/10；剔除 1998 法国这一前两届无世界杯决赛圈前史样本后为 9/9。</div></div>

<h2>3. 口径 B：前两届都参赛的严格排除规则</h2>
<p><strong>这条规则刻意很窄。</strong> 只有当一支球队前两届都参加了世界杯决赛圈，并且这两届所有淘汰赛/争冠阶段经历中，既没有自己夺冠，也没有输给当届冠军或亚军，才进入排除池。1974、1978、1982 的第二阶段小组被视为争冠阶段，因为这些年份不是现代 16 强淘汰赛结构。</p>
<p><strong>“前两届都参赛”这个门槛确实苛刻，所以要在同一个门槛内对比。</strong> 1938-2022 的目标样本共有 {context['path_team_tournaments_total']} 个球队-届次；其中只有 {context['path_prior_both_total']} 个满足“前两届都参赛”。在这 {context['path_prior_both_total']} 个里面，{context['path_contact_total']} 个有过冠军/亚军路径接触，{context['clean_non_contact_teams']} 个没有。</p>
<p><strong>65 个不是 65 支唯一球队，而是 65 个球队-届次。</strong> 它是严格参赛门槛之后的“干净无接触”一侧，对照组是同样满足前两届参赛条件、但已经有冠亚军路径接触的 {context['path_contact_total']} 个球队-届次。更细地拆，严格意义上“前两届输给当届冠军/亚军”的是 {context['path_lost_to_finalist_total']} 个；“前两届自己当过冠军”的是 {context['path_own_champion_total']} 个；两者重叠 {context['path_contact_overlap_total']} 个，所以合并后是 {context['path_contact_total']} 个。</p>
<div class="figure"><img src="{context['fig_history_zh']}" alt="Historical path exclusion scope"><div class="caption">蓝色代表冠军在前两届都参赛且已有路径接触；灰色代表冠军不适用排除命题，因为前两届没有都参赛。图中没有红色反例。</div></div>
<p><strong>看似的例外，在这个定义下会消失。</strong> 1978 阿根廷没有参加 1970 决赛圈；1998 法国没有参加 1990 和 1994 决赛圈；1982 意大利虽然前两届都参赛，但 1978 年第二阶段小组输给了最终亚军荷兰。</p>
{examples_html}

<h2>4. 两套 simulation：宽口径命中与严格口径排除</h2>
<p><strong>两套基准的思想一致：每届保留同样规模，然后随机化标签。</strong> 对纯 CCS 来说，检验的是“同规模随机候选池能否命中至少同样多冠军”；对严格口径来说，检验的是“同规模随机排除池能否一次都不排掉冠军”。</p>
<p><strong>纯 CCS 随机基准是命中率检验。</strong> 全历史淘汰路径样本中，同规模随机候选池期望命中 {context['historical_random_expected']} 个冠军；达到 {context['historical_champions_short']} 的精确概率是 {context['historical_random_prob']}，Monte Carlo 概率是 {context['historical_random_sim_prob']}。</p>
<p><strong>严格排除随机基准是误伤检验。</strong> 脚本做 20 万次同规模随机重抽；模拟得到 0 次误排冠军的概率为 {context['path_random_zero_sim_prob']}，精确概率为 {context['path_random_zero_prob']}。</p>
<div class="figure"><img src="{context['fig_random_zh']}" alt="Random benchmark chart"><div class="caption">规则排除 {context['clean_non_contact_teams']} 个球队-届次，且 0 次排到冠军。同规模随机排除的期望误排冠军数为 {context['path_random_expected']}。</div></div>
<p><strong>更强的控制，是检验 CCS 是否只是“强队标签”。</strong> 这个 simulation 明确限制在 1998-2022，因为这一段有一致的赛前 FIFA 排名语境，也更适合人工定义“现代冠军叙事队”。我们保留 CCS 在“传统冠军叙事队”和普通队中的数量，再在两个组内随机置换 CCS 标签。剔除法国 1998 这个明确无前史例外后，CCS 实际命中 6/6 个可判定冠军；强队标签置换的期望为 {context['strong_sim_expected']}/6，达到 6/6 的概率为 {context['strong_sim_prob']}。</p>
<div class="figure"><img src="{context['fig_sim_zh']}" alt="Strong-team permutation simulation"><div class="caption">这不是世界排名 simulation。它控制的是“CCS 本来就会和传统强队重叠”这件事，再检验具体的冠军链标签是否还有额外信息。</div></div>
<div class="callout warn"><strong>解释边界：</strong> 这些 simulation 支持 CCS 是一个可信的筛选启发式，但尚不能证明它优于 Elo、赔率或多变量模型。下一步应检验相对于强基准的增量价值。</div>

<h2>5. 赛前使用体验：哪些强队/豪门应被 CCS 降权</h2>
<p><strong>这是最容易让读者理解的方法使用场景。</strong> 开赛前，一支球队可以排名很高、历史声望很强、舆论很热，但仍然缺少最近两届的冠军链连接。主图从“FIFA Top 20 且非 CCS”的审计池里人工策展，重点保留今天读者也会自然认为与冠军叙事相关的强队：阿根廷、德国、英格兰、西班牙、葡萄牙、荷兰、比利时、哥伦比亚、克罗地亚、乌拉圭。</p>
<div class="figure"><img src="{context['fig_traps_zh']}" alt="Recognizable non-CCS title contenders"><div class="caption">样本来自开赛前 FIFA Top 20 且非 CCS 的入围队；主图展示人工策展的强队/豪门清单。完整机械 Top 20 审计表保留在 data/derived/favorite_traps.csv；策展清单保留在 data/derived/favorite_trap_powerhouses.csv。</div></div>
{traps_html}
<p><strong>这个信号有用，但不是绝对排除。</strong> 非 CCS 强队可以走很远：2002 德国、2010 荷兰、2014 阿根廷都进入决赛。更准确的结论是：现代样本里，最终冠军几乎总来自 CCS 一侧。</p>

<h2>6. 2026 应用：区分排名强与冠军链强</h2>
<p><strong>2026 是实时应用场景，不是回测结果。</strong> 本报告将排名快照冻结在 FIFA 2026 年 6 月 11 日官方排名，并使用 FIFA 2026 赛季接口中的入围队名单。最直接的赛前结论是：西班牙、葡萄牙、巴西、德国、哥伦比亚，都是排名强、声望强，但开赛前非 CCS 的豪强。</p>
{downgrade_2026_html}
<p>下方完整观察表保留头部排名上下文，避免把 2026 的降权判断藏在人工挑选名单里。</p>
<div class="figure"><img src="{context['fig_2026_zh']}" alt="2026 ranked non-CCS contenders"><div class="caption">2026 已入围球队中五支高声望但非 CCS 的豪强；下方表格保留更完整的头部排名上下文。</div></div>
{watch_html}

<h2>7. 为什么会这样：强队重要，但路径也重要</h2>
<p><strong>首先要承认，CCS 确实部分捕捉了强队效应。</strong> 冠军、亚军，以及被冠亚军淘汰的球队，本来就往往是强队。因此 CCS 与 FIFA 排名、赔率、历史声望存在天然重叠。</p>
<p><strong>但 CCS 并不等同于“经常入围”或“排名很高”。</strong> 它要求球队在最近两届与冠军路径发生过具体关系：自己夺冠，或被最终冠亚军淘汰。一个队可以排名高、经常参赛、甚至进过 8 强，但如果没有触碰冠军路径，仍然可能是非 CCS。</p>
<p><strong>更合理的机制解释是“锦标赛周期验证”。</strong> 最近两届与冠军链发生联系，意味着球队已经在世界杯淘汰赛强度下被冠军级对手验证过。这不是因果证明，而是一个紧凑的历史状态变量；它对冠军列尤其有解释力。</p>

<h2>建议下一步</h2>
<ol>
<li><strong>把 CCS 作为第一层冠军过滤器。</strong> 先用 CCS 构建冠军候选池；非 CCS 球队若要重新纳入，需要赔率、Elo、阵容或签表给出额外强证据。</li>
<li><strong>补强正式模型基准。</strong> 下一版应与 Elo、赔率隐含概率、FIFA 排名和组合模型比较，检验 CCS 的增量价值。</li>
<li><strong>固定 2026 赛前版本，赛后再评估。</strong> 不应随着赛果移动口径；应冻结赛前 CCS/排名表，等比赛结束后统一复盘。</li>
</ol>

<h2>待回答问题</h2>
<ul>
<li>控制 Elo 或赔率后，CCS 是否仍有预测增量？</li>
<li>回看 1 届、2 届还是 3 届最优？是否存在过拟合？</li>
<li>主办国、阵容周期突变、无前史样本，是否应建立单独 override 规则？</li>
</ul>

<h2>限制与假设</h2>
<ul>
<li>主结论使用的是球队-届次，不是唯一国家队；同一国家在多个周期满足条件，会被计为多个样本点。</li>
<li>1950 年保留在目标年份审计中，因为这条规则检验的是“下一届冠军在前两届是否有路径接触”；目标届本身不一定需要现代淘汰赛结构。</li>
<li>1974、1978、1982 的第二阶段小组视为争冠阶段，因为这些年份采用混合赛制，不是现代淘汰赛签表。</li>
<li>随机基准是同规模随机排除的精确概率计算；Monte Carlo simulation 只是复核这个乘法，不是黑箱模型。</li>
<li>强队标签置换 simulation 仍是 1998-2022 的现代控制实验，不是全历史 simulation，因为它依赖一致的现代排名语境和人工冠军叙事标签。</li>
<li>FIFA 排名只是轻量公开审计变量；赔率与 Elo 是下一步更强基准。</li>
<li>2026 观察清单不使用 2026 赛果；排名快照冻结在 2026 年 6 月 11 日。</li>
</ul>

<div class="footer">
由 scripts/build_report.py 生成。数据源：Fjelstul World Cup Database；FIFA API 排名日程、排名数据和 2026 入围球队接口。原修正版 CCS 报告保存在 reports/pdf/original_ccs_report_corrected.pdf。
</div>
</main>
</body>
</html>"""


def main() -> None:
    ensure_dirs()
    set_chart_style()

    team_year = build_team_year()
    historical_team_year, historical_summary, format_scope = build_historical_knockout_backtest()
    path_detail, path_summary = build_path_exclusion_backtest()
    path_random = exclusion_random_benchmark(path_summary)
    rankings = build_rankings()
    modern = pd.read_csv(DATA_RAW / "modern_ccs.csv")
    random_df = random_benchmark(modern)
    historical_random = random_benchmark_from_summary(historical_summary, "historical_random_benchmark.csv")
    joined = build_favorite_traps(team_year, rankings)
    permutation_detail, permutation_summary = contender_permutation_benchmark(joined, modern)
    traps = pd.read_csv(DATA_DERIVED / "favorite_traps.csv")
    trap_headliners = pd.read_csv(DATA_DERIVED / "favorite_trap_headliners.csv")
    trap_powerhouses = pd.read_csv(DATA_DERIVED / "favorite_trap_powerhouses.csv")
    watch = pd.read_csv(DATA_DERIVED / "ccs_2026_watchlist.csv")
    downgrade_2026 = pd.read_csv(DATA_DERIVED / "ccs_2026_downgrade_giants.csv")

    total_ccs = modern["ccs"].sum()
    total_participants = modern["participants"].sum()
    champ_hits = modern["champ_ccs"].sum()
    champion_total = modern["champ_total"].sum()
    evaluable_total = champion_total - 1
    historical_evaluable = historical_summary[historical_summary["evaluable"].eq(1)]
    historical_hits = int(historical_evaluable["champ_ccs"].sum())
    historical_total = int(len(historical_evaluable))
    historical_pool_share = historical_evaluable["ccs_candidates"].sum() / historical_evaluable["participants"].sum()
    path_team_tournaments_total = int(path_summary["participants"].sum())
    path_prior_both_total = int(path_summary["prior_participated_both_teams"].sum())
    path_contact_total = int(path_summary["path_contact_teams"].sum())
    prior_both_detail = path_detail[path_detail["prior_participated_both"].eq(1)].copy()
    contact_reasons = prior_both_detail["contact_reasons"].fillna("")
    path_lost_to_finalist_total = int(contact_reasons.str.contains("lost to").sum())
    path_own_champion_total = int(contact_reasons.str.contains("own champion").sum())
    path_contact_overlap_total = int(
        (contact_reasons.str.contains("lost to") & contact_reasons.str.contains("own champion")).sum()
    )
    clean_non_contact_total = int(path_summary["clean_non_contact_teams"].sum())
    clean_non_contact_champions = int(path_summary["clean_non_contact_champions"].sum())
    path_champions_with_prior_both = int(path_summary["champion_prior_participated_both"].sum())
    path_champions_with_contact = int(
        (
            path_summary["champion_prior_participated_both"].eq(1)
            & path_summary["champion_path_contact"].eq(1)
        ).sum()
    )

    context = {
        "ccs_pool_share": pct(total_ccs / total_participants),
        "champ_all": f"{int(champ_hits)}/{int(champion_total)} ({pct(champ_hits / champion_total)})",
        "champ_evaluable": f"{int(champ_hits)}/{int(evaluable_total)} (100.0%)",
        "champ_evaluable_short": f"{int(champ_hits)}/{int(evaluable_total)}",
        "historical_champions": f"{historical_hits}/{historical_total} ({pct(historical_hits / historical_total)})",
        "historical_champions_short": f"{historical_hits}/{historical_total}",
        "historical_evaluable_total": historical_total,
        "historical_actual_hits": historical_hits,
        "historical_pool_share": pct(historical_pool_share),
        "historical_random_expected": f"{historical_random['expected_random_hits'].iloc[0]:.1f}",
        "historical_random_prob": f"{historical_random['prob_random_ge_actual'].iloc[0]:.4%}",
        "historical_random_sim_prob": small_tail_pct(
            float(historical_random["prob_random_ge_actual_simulated"].iloc[0]),
            int(historical_random["simulation_runs"].iloc[0]),
        ),
        "path_total_target_years": int(len(path_summary)),
        "path_team_tournaments_total": path_team_tournaments_total,
        "path_prior_both_total": path_prior_both_total,
        "path_contact_total": path_contact_total,
        "path_lost_to_finalist_total": path_lost_to_finalist_total,
        "path_own_champion_total": path_own_champion_total,
        "path_contact_overlap_total": path_contact_overlap_total,
        "path_contact_vs_clean": f"{path_contact_total}/{clean_non_contact_total}",
        "clean_non_contact_teams": clean_non_contact_total,
        "clean_non_contact_champions": clean_non_contact_champions,
        "clean_non_contact_champions_short": f"{clean_non_contact_champions}/{clean_non_contact_total}",
        "path_champion_contact_short": f"{path_champions_with_contact}/{path_champions_with_prior_both}",
        "path_random_expected": f"{path_random['expected_random_excluded_champions'].iloc[0]:.1f}",
        "path_random_zero_prob": f"{path_random['prob_random_zero_excluded_champions_exact'].iloc[0]:.1%}",
        "path_random_zero_sim_prob": f"{path_random['prob_random_zero_excluded_champions_simulated'].iloc[0]:.1%}",
        "random_expected": f"{random_df['random_hit_probability'].sum():.1f}",
        "random_prob": f"{random_df['prob_random_ge_9_of_10'].iloc[0]:.3%}",
        "strong_sim_expected": f"{permutation_summary.loc[permutation_summary['scope'].eq('evaluable_ex_france_1998'), 'expected_permutation_hits'].iloc[0]:.1f}",
        "strong_sim_prob": f"{permutation_summary.loc[permutation_summary['scope'].eq('evaluable_ex_france_1998'), 'probability_ge_actual_exact'].iloc[0]:.1%}",
        "fig_history_en": chart_historical_scope(path_summary, format_scope, "en"),
        "fig_pure_random_en": chart_pure_ccs_random_benchmark(historical_summary, historical_random, "en"),
        "fig_funnel_en": chart_modern_funnel(modern, "en"),
        "fig_random_en": chart_random_benchmark(path_summary, "en"),
        "fig_traps_en": chart_favorite_traps(trap_powerhouses, "en"),
        "fig_2026_en": chart_2026_watchlist(downgrade_2026, "en"),
        "fig_sim_en": chart_contender_permutation(permutation_summary, "en"),
        "fig_history_zh": chart_historical_scope(path_summary, format_scope, "zh"),
        "fig_pure_random_zh": chart_pure_ccs_random_benchmark(historical_summary, historical_random, "zh"),
        "fig_funnel_zh": chart_modern_funnel(modern, "zh"),
        "fig_random_zh": chart_random_benchmark(path_summary, "zh"),
        "fig_traps_zh": chart_favorite_traps(trap_powerhouses, "zh"),
        "fig_2026_zh": chart_2026_watchlist(downgrade_2026, "zh"),
        "fig_sim_zh": chart_contender_permutation(permutation_summary, "zh"),
        "historical_team_year": historical_team_year,
        "historical_summary": historical_summary,
        "path_detail": path_detail,
        "path_summary": path_summary,
        "path_random": path_random,
        "format_scope": format_scope,
        "historical_random": historical_random,
        "traps": traps,
        "trap_headliners": trap_headliners,
        "trap_powerhouses": trap_powerhouses,
        "permutation_detail": permutation_detail,
        "permutation_summary": permutation_summary,
        "watch_2026": watch,
        "downgrade_2026": downgrade_2026,
    }

    report_en = render_report_en(context)
    report_zh = render_report_zh(context)
    (REPORTS / "world_cup_ccs_investment_report_en.html").write_text(report_en, encoding="utf-8")
    (REPORTS / "world_cup_ccs_investment_report_zh.html").write_text(report_zh, encoding="utf-8")
    (REPORTS / "world_cup_ccs_investment_report.html").write_text(report_en, encoding="utf-8")
    print(f"Wrote {REPORTS / 'world_cup_ccs_investment_report_en.html'}")
    print(f"Wrote {REPORTS / 'world_cup_ccs_investment_report_zh.html'}")


if __name__ == "__main__":
    main()
