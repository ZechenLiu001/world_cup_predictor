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
    out["prob_random_ge_9_of_10"] = sum(dist[9:])
    out.to_csv(DATA_DERIVED / "random_benchmark.csv", index=False)
    return out


def poisson_binomial_distribution(probs: list[float]) -> np.ndarray:
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, np.array([1 - p, p]))
    return dist


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


def chart_random_benchmark(random_df: pd.DataFrame, lang: str = "en") -> str:
    ccs_hits = int(random_df["ccs_hit"].sum())
    exp_hits = float(random_df["random_hit_probability"].sum())
    prob_ge9 = float(random_df["prob_random_ge_9_of_10"].iloc[0])
    labels = ["CCS actual", "Random same-size pool\nexpected", "Random chance of\n≥9 hits"]
    if lang == "zh":
        labels = ["CCS 实际命中", "同规模随机池\n期望命中", "随机达到\n≥9次命中"]
    vals = [ccs_hits, exp_hits, prob_ge9 * 10]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#0b5cad", "#9aa6b2", "#d15532"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylim(0, 10)
    ax.set_ylabel("10届冠军命中次数\n（第三柱为概率缩放展示）" if lang == "zh" else "Champion hits out of 10\n(third bar scaled to 10)")
    ax.set_title("CCS 的 9/10 命中不是随机三成筛选能轻易做到的" if lang == "zh" else "CCS coverage is not what random one-third screening would produce")
    ax.grid(axis="y", color="#e8ebf1")
    ax.spines[["top", "right"]].set_visible(False)
    annotations = [f"{ccs_hits}/10", f"{exp_hits:.1f}/10", f"{prob_ge9:.3%} probability"]
    for bar, label in zip(bars, annotations):
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.text(
        2,
        1.15,
        "红色柱为概率值，缩放后仅用于可视化。" if lang == "zh" else "The red bar is a probability, scaled only so it remains visible.",
        ha="center",
        fontsize=9,
        color="#5b6575",
    )
    return savefig(f"02_random_benchmark_{lang}.png")


def chart_favorite_traps(traps: pd.DataFrame, lang: str = "en") -> str:
    top = traps.copy()
    top["label"] = top["year"].astype(str) + " " + top["team_name"].map(lambda x: display_team(x, lang))
    top = top.sort_values(["year", "fifa_rank"])
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    y = np.arange(len(top))
    colors = np.where(top["performance"].str.lower().eq("final"), "#d15532", "#67758a")
    ax.barh(y, np.ones(len(top)), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.28)
    ax.set_xticks([])
    ax.set_title("一众强队/豪门但非 CCS：赛前可降权名单" if lang == "zh" else "A wall of recognizable contenders that CCS would have downgraded")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(
            1.02,
            i,
            f"#{int(r['fifa_rank'])} · {display_performance(r['performance'], lang)}",
            va="center",
            fontsize=10,
        )
    return savefig(f"03_favorite_traps_{lang}.png")


def chart_2026_watchlist(watch: pd.DataFrame, lang: str = "en") -> str:
    top = watch[watch["fifa_rank"].le(24)].copy().sort_values("fifa_rank")
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    y = np.arange(len(top))
    colors = np.where(top["ccs"].eq(1), "#0b5cad", "#d15532")
    ax.barh(y, np.ones(len(top)), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r.fifa_rank)} {display_team(r.team_name, lang)}" for r in top.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    ax.set_title("2026 赛前观察：哪些强队缺少 CCS 支持" if lang == "zh" else "2026 pre-tournament watchlist: which contenders lack CCS support")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    for i, r in enumerate(top.itertuples()):
        status = "CCS" if r.ccs else ("非CCS" if lang == "zh" else "Non-CCS")
        ax.text(1.02, i, status, va="center", fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#0b5cad", label="CCS 候选" if lang == "zh" else "CCS candidate"),
            Patch(color="#d15532", label="排名强但非 CCS" if lang == "zh" else "Ranked strong, non-CCS"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
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


def render_report_en(context: dict) -> str:
    css = report_css()
    traps_html, watch_html, downgrade_2026_html = prepare_tables(context, "en")
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
<p class="subhead">A reproducible backtest of the Champion-Chain Signal (CCS), plus a practical 1998 to 2026 pre-tournament lens for identifying recognizable contenders that deserve a champion-probability downgrade.</p>

<section class="summary">
<h2>Executive Summary</h2>
<ul>
<li><strong>CCS is a candidate-pool filter, not a champion picker.</strong> In the 1986-2022 modern knockout era, CCS retained only {context['ccs_pool_share']} of team-tournaments but covered {context['champ_all']} of all champions; after excluding the one no-prior-history case, coverage was {context['champ_evaluable']}.</li>
<li><strong>The result is not a random one-third screen.</strong> A random candidate pool with each year's same size would expect only {context['random_expected']} champion hits out of 10; the probability of randomly reaching at least 9 hits is {context['random_prob']}.</li>
<li><strong>A stronger simulation still supports the signal, but with humility.</strong> After preserving the mix of traditional title contenders and ordinary teams, a label-permutation control expects {context['strong_sim_expected']} hits out of 6 evaluable champions; reaching 6/6 is a {context['strong_sim_prob']} event.</li>
<li><strong>The most useful reader experience is the pre-tournament downgrade list.</strong> From 1998 onward, a long list of recognizable contenders were highly ranked but non-CCS at kickoff. Many still advanced, including finalists, but none won in the modern sample.</li>
<li><strong>The mechanism is partly strength, but more specific than strength.</strong> CCS overlaps with elite teams, yet it asks a narrower question: has this team recently been on, or directly removed from, the champion path?</li>
</ul>
</section>

<div class="kpis">
<div class="kpi"><div class="value">{context['champ_evaluable']}</div><div class="label">Evaluable modern champions covered</div></div>
<div class="kpi"><div class="value">{context['champ_all']}</div><div class="label">All modern champions covered</div></div>
<div class="kpi"><div class="value">{context['ccs_pool_share']}</div><div class="label">Modern candidate-pool share</div></div>
<div class="kpi"><div class="value">{context['random_prob']}</div><div class="label">Random same-size pool reaches ≥9/10</div></div>
</div>

<h2>1. CCS gets more concentrated as the tournament gets serious</h2>
<p><strong>The primary backtest pattern is monotonic.</strong> CCS teams are about three-tenths of the field, but their share rises at every deeper stage: round of 16, quarter-finals, semi-finals, final, and champion. This makes the signal more useful as a championship filter than as a generic knockout-stage prediction tool.</p>
<div class="figure"><img src="{context['fig_funnel_en']}" alt="CCS modern stage funnel"><div class="caption">Modern era is 1986-2022. Champion coverage is 9/10 on the all-champion denominator and 9/9 after removing France 1998, which had no prior-two-World-Cup finals history.</div></div>

<h2>2. Method and simulation: stronger than random, not proof by itself</h2>
<p><strong>The method has two layers.</strong> First, CCS is defined before the tournament from the prior two World Cups only: a team qualifies if it recently won the World Cup or was knocked out by a later finalist. Second, the report tests whether that pre-tournament label covers champions better than no-information and strong-team controls.</p>
<p><strong>A same-size random pool is the first hurdle.</strong> If 1998 has 11 CCS candidates out of 32 teams, the random benchmark also randomly selects 11 of 32. Repeating that across the ten modern tournaments yields an expected {context['random_expected']} champion hits, not nine. Hitting at least nine by chance is a {context['random_prob']} event.</p>
<div class="figure"><img src="{context['fig_random_en']}" alt="Random benchmark chart"><div class="caption">The benchmark preserves each year's candidate-pool size, so it tests information content rather than simply rewarding CCS for selecting more teams.</div></div>
<p><strong>The stronger control asks whether CCS is merely a famous-team label.</strong> For each tournament from 1998 to 2022, we preserve the number of CCS labels inside the traditional title-contender set and outside it, then randomly permute the labels inside those two groups. Excluding the explicit France 1998 no-prior-history exception, CCS hit 6/6 evaluable champions; the strong-label permutation expects {context['strong_sim_expected']}/6, and reaches 6/6 with probability {context['strong_sim_prob']}.</p>
<div class="figure"><img src="{context['fig_sim_en']}" alt="Strong-team permutation simulation"><div class="caption">This is deliberately not a FIFA-ranking simulation. It controls for the broader fact that CCS often overlaps with obvious football powers, then asks whether the specific champion-chain label still carries information.</div></div>
<div class="callout warn"><strong>Interpretation discipline:</strong> these simulations support CCS as a credible screening heuristic. They do not prove it beats Elo, betting odds, or a full multivariate model. The next bar is incremental value versus those stronger baselines.</div>

<h2>3. The pre-tournament experience: recognizable favorites CCS would downgrade</h2>
<p><strong>This is the most intuitive way to use the method.</strong> Before kickoff, a team can be highly ranked, historically recognizable, and still lack a recent champion-chain connection. The main exhibit is curated from a Top-20 non-CCS audit pool to show the teams a modern audience would naturally treat as title-relevant: Argentina, Germany, England, Spain, Portugal, Netherlands, Belgium, Colombia, Croatia, and Uruguay.</p>
<div class="figure"><img src="{context['fig_traps_en']}" alt="Recognizable non-CCS title contenders"><div class="caption">Curated from qualified teams that were FIFA Top 20 and non-CCS at kickoff. The full mechanical Top-20 audit table is retained in data/derived/favorite_traps.csv; the curated list is retained in data/derived/favorite_trap_powerhouses.csv.</div></div>
{traps_html}
<p><strong>The pattern is useful but not absolute.</strong> Non-CCS strong teams can go deep: 2002 Germany, 2010 Netherlands, and 2014 Argentina reached finals. The historical point is narrower and stronger: in the modern sample, the champion almost always came from the CCS side of the field.</p>

<h2>4. 2026 application: separate rank strength from champion-chain strength</h2>
<p><strong>The 2026 view is a live-use case, not a backtest result.</strong> The ranking snapshot is frozen at FIFA's June 11, 2026 official ranking and the qualified-team list is from FIFA's 2026 season endpoint. The headline application is clear: Spain, Portugal, Brazil, Germany, and Colombia are rank-strong, reputation-strong, but non-CCS before kickoff.</p>
{downgrade_2026_html}
<p>The broader watchlist below keeps the full top-ranked context visible, so the downgrade call is not hidden inside a hand-picked list.</p>
<div class="figure"><img src="{context['fig_2026_en']}" alt="2026 ranked watchlist"><div class="caption">Top 24 ranked qualified teams in the 2026 field. Blue teams have CCS support from 2018 or 2022; red teams are rank-strong but non-CCS.</div></div>
{watch_html}

<h2>5. Why this happens: strength matters, but path matters too</h2>
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
<li>The modern champion sample has only 10 observations. The 100% evaluable number is 9/9, not a stable law.</li>
<li>France 1998 is treated as a no-prior-history exception because it missed the prior two World Cup finals tournaments.</li>
<li>FIFA ranking is used as a public strong-team proxy; odds and Elo would be better next-step baselines.</li>
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
<p class="subhead">本报告对世界杯 Champion-Chain Signal（CCS）做可复现回测，并把它转化为 1998 至 2026 的赛前使用框架：哪些声望强、舆论热、看起来像冠军候选的球队，实际上应在夺冠概率上被降权。</p>

<section class="summary">
<h2>执行摘要</h2>
<ul>
<li><strong>CCS 是冠军候选池过滤器，不是单点冠军预测器。</strong> 在 1986-2022 现代淘汰赛时代，CCS 只保留 {context['ccs_pool_share']} 的球队-届次，却覆盖全部现代冠军中的 {context['champ_all']}；若剔除唯一“前两届无可判定世界杯前史”的法国 1998，则为 {context['champ_evaluable']}。</li>
<li><strong>这不是随机挑三成球队就能得到的结果。</strong> 若每届随机抽取与 CCS 同等规模的候选池，10 届期望只命中 {context['random_expected']} 个冠军；随机达到至少 9 次命中的概率只有 {context['random_prob']}。</li>
<li><strong>更强的 simulation 也支持这个信号，但要克制解读。</strong> 在保留“传统强队/普通队”结构后随机置换 CCS 标签，6 个可判定冠军的期望命中为 {context['strong_sim_expected']} 个；达到 6/6 的概率为 {context['strong_sim_prob']}。</li>
<li><strong>最有体感的用法，是赛前热门队降权清单。</strong> 1998 年以来，一众排名高、名气大、今天读者也会认为有冠军叙事的强队，在开赛前并非 CCS。它们并不一定弱，甚至可能进决赛，但现代样本里没有最终夺冠。</li>
<li><strong>机制上，CCS 有强队效应，但比“强队”更窄。</strong> 它问的不是球队是否有名、排名是否高，而是它最近两届是否已经进入过冠军路径，或被冠军/亚军直接淘汰验证过。</li>
</ul>
</section>

<div class="kpis">
<div class="kpi"><div class="value">{context['champ_evaluable']}</div><div class="label">可判定现代冠军覆盖</div></div>
<div class="kpi"><div class="value">{context['champ_all']}</div><div class="label">全部现代冠军覆盖</div></div>
<div class="kpi"><div class="value">{context['ccs_pool_share']}</div><div class="label">现代候选池占比</div></div>
<div class="kpi"><div class="value">{context['random_prob']}</div><div class="label">随机同规模池达到 ≥9/10</div></div>
</div>

<h2>1. 赛事越深入，CCS 越集中</h2>
<p><strong>核心回测形态是单调上升。</strong> CCS 球队约占全部参赛队三成，但从 16 强、8 强、4 强、决赛到冠军，其占比逐层上升。这意味着 CCS 更像“冠军过滤器”，而不是普通的淘汰赛晋级预测器。</p>
<div class="figure"><img src="{context['fig_funnel_zh']}" alt="CCS modern stage funnel"><div class="caption">现代时代定义为 1986-2022。全部冠军口径为 9/10；剔除 1998 法国这一前两届无世界杯决赛圈前史样本后为 9/9。</div></div>

<h2>2. 方法与 simulation：强于随机，但不是单独证明</h2>
<p><strong>方法分两层。</strong> 第一，CCS 只使用赛前已知的前两届世界杯信息：球队若最近夺冠，或被后来的冠亚军淘汰，就进入冠军链候选池。第二，报告检验这个赛前标签是否比无信息随机池和“强队标签”控制更能覆盖冠军。</p>
<p><strong>同规模随机候选池是第一道合理门槛。</strong> 如果 1998 年 CCS 候选池是 32 队中的 11 队，那么随机基准也只随机抽 11 队。把这个逻辑重复到 10 届现代世界杯，随机期望命中只有 {context['random_expected']} 个冠军，而不是 9 个；随机达到至少 9 次命中的概率仅 {context['random_prob']}。</p>
<div class="figure"><img src="{context['fig_random_zh']}" alt="Random benchmark chart"><div class="caption">该基准保留每届实际 CCS 候选池规模，因此检验的是“信息量”，不是简单奖励候选池更大。</div></div>
<p><strong>更强的控制，是检验 CCS 是否只是“强队标签”。</strong> 对 1998-2022 每届世界杯，我们保留 CCS 在“传统冠军叙事队”和普通队中的数量，再在两个组内随机置换 CCS 标签。剔除法国 1998 这个明确无前史例外后，CCS 实际命中 6/6 个可判定冠军；强队标签置换的期望为 {context['strong_sim_expected']}/6，达到 6/6 的概率为 {context['strong_sim_prob']}。</p>
<div class="figure"><img src="{context['fig_sim_zh']}" alt="Strong-team permutation simulation"><div class="caption">这不是世界排名 simulation。它控制的是“CCS 本来就会和传统强队重叠”这件事，再检验具体的冠军链标签是否还有额外信息。</div></div>
<div class="callout warn"><strong>解释边界：</strong> 这些 simulation 支持 CCS 是一个可信的筛选启发式，但尚不能证明它优于 Elo、赔率或多变量模型。下一步应检验相对于强基准的增量价值。</div>

<h2>3. 赛前使用体验：哪些强队/豪门应被 CCS 降权</h2>
<p><strong>这是最容易让读者理解的方法使用场景。</strong> 开赛前，一支球队可以排名很高、历史声望很强、舆论很热，但仍然缺少最近两届的冠军链连接。主图从“FIFA Top 20 且非 CCS”的审计池里人工策展，重点保留今天读者也会自然认为与冠军叙事相关的强队：阿根廷、德国、英格兰、西班牙、葡萄牙、荷兰、比利时、哥伦比亚、克罗地亚、乌拉圭。</p>
<div class="figure"><img src="{context['fig_traps_zh']}" alt="Recognizable non-CCS title contenders"><div class="caption">样本来自开赛前 FIFA Top 20 且非 CCS 的入围队；主图展示人工策展的强队/豪门清单。完整机械 Top 20 审计表保留在 data/derived/favorite_traps.csv；策展清单保留在 data/derived/favorite_trap_powerhouses.csv。</div></div>
{traps_html}
<p><strong>这个信号有用，但不是绝对排除。</strong> 非 CCS 强队可以走很远：2002 德国、2010 荷兰、2014 阿根廷都进入决赛。更准确的结论是：现代样本里，最终冠军几乎总来自 CCS 一侧。</p>

<h2>4. 2026 应用：区分排名强与冠军链强</h2>
<p><strong>2026 是实时应用场景，不是回测结果。</strong> 本报告将排名快照冻结在 FIFA 2026 年 6 月 11 日官方排名，并使用 FIFA 2026 赛季接口中的入围队名单。最直接的赛前结论是：西班牙、葡萄牙、巴西、德国、哥伦比亚，都是排名强、声望强，但开赛前非 CCS 的豪强。</p>
{downgrade_2026_html}
<p>下方完整观察表保留头部排名上下文，避免把 2026 的降权判断藏在人工挑选名单里。</p>
<div class="figure"><img src="{context['fig_2026_zh']}" alt="2026 ranked watchlist"><div class="caption">2026 已入围球队中排名前 24 的队伍。蓝色代表 2018 或 2022 提供 CCS 支持；红色代表排名强但非 CCS。</div></div>
{watch_html}

<h2>5. 为什么会这样：强队重要，但路径也重要</h2>
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
<li>现代冠军样本只有 10 个观测；“100% 可判定覆盖”是 9/9，不应被理解为稳定规律。</li>
<li>1998 法国被视为无前史例外，因为它缺席此前两届世界杯决赛圈。</li>
<li>FIFA 排名只是公开强队代理变量；赔率与 Elo 是下一步更强基准。</li>
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
    rankings = build_rankings()
    modern = pd.read_csv(DATA_RAW / "modern_ccs.csv")
    random_df = random_benchmark(modern)
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

    context = {
        "ccs_pool_share": pct(total_ccs / total_participants),
        "champ_all": f"{int(champ_hits)}/{int(champion_total)} ({pct(champ_hits / champion_total)})",
        "champ_evaluable": f"{int(champ_hits)}/{int(evaluable_total)} (100.0%)",
        "random_expected": f"{random_df['random_hit_probability'].sum():.1f}",
        "random_prob": f"{random_df['prob_random_ge_9_of_10'].iloc[0]:.3%}",
        "strong_sim_expected": f"{permutation_summary.loc[permutation_summary['scope'].eq('evaluable_ex_france_1998'), 'expected_permutation_hits'].iloc[0]:.1f}",
        "strong_sim_prob": f"{permutation_summary.loc[permutation_summary['scope'].eq('evaluable_ex_france_1998'), 'probability_ge_actual_exact'].iloc[0]:.1%}",
        "fig_funnel_en": chart_modern_funnel(modern, "en"),
        "fig_random_en": chart_random_benchmark(random_df, "en"),
        "fig_traps_en": chart_favorite_traps(trap_powerhouses, "en"),
        "fig_2026_en": chart_2026_watchlist(watch, "en"),
        "fig_sim_en": chart_contender_permutation(permutation_summary, "en"),
        "fig_funnel_zh": chart_modern_funnel(modern, "zh"),
        "fig_random_zh": chart_random_benchmark(random_df, "zh"),
        "fig_traps_zh": chart_favorite_traps(trap_powerhouses, "zh"),
        "fig_2026_zh": chart_2026_watchlist(watch, "zh"),
        "fig_sim_zh": chart_contender_permutation(permutation_summary, "zh"),
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
