"""Comeback dashboard API.

Start:  python -m uvicorn main:app --port 8000
Docs:   http://localhost:8000/docs

The data is loaded once at import time. Two kinds of work are split:
  * the server aggregates across the whole league (47 teams x 32 seasons,
    ~24,000 team-matches) for the team overview and the league baseline;
  * the client receives one team's ~1,200 rows and filters them itself,
    so dragging the season brush never waits for the network.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATA = Path(__file__).parent / "data"
appearances_path = DATA / "appearances.csv"
if not appearances_path.exists():
    raise SystemExit("data/appearances.csv not found - run `python clean.py` first")

A = pd.read_csv(appearances_path)
M = pd.read_csv(DATA / "matches.csv")

SEASON_MIN = int(A.season.min())
SEASON_MAX = int(A.season.max())
LATEST_PARTIAL = int(M.season.max())  # the season still being played
TEAMS = sorted(A.team.unique())

Venue = Literal["all", "H", "A"]

app = FastAPI(title="La Liga comebacks API", version="1.0")

# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS=https://<user>.github.io
# Defaults to "*" so localhost development works without configuration.
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"]
)


def label(y: int) -> str:
    return f"{y}/{str(y + 1)[2:]}"


def check_range(season_min: int, season_max: int) -> None:
    if season_min > season_max:
        raise HTTPException(400, "season_min must not be after season_max")


def subset(season_min: int, season_max: int, venue: Venue) -> pd.DataFrame:
    x = A[(A.season >= season_min) & (A.season <= season_max)]
    if venue != "all":
        x = x[x.venue == venue]
    return x


def outcome_counts(x: pd.DataFrame) -> dict:
    """W/D/L counts and rates for the matches in x that were trailing at half-time."""
    tr = x[x.ht == "trail"]
    n = len(tr)
    w, d, l = (int((tr.ft == r).sum()) for r in ("W", "D", "L"))
    return {
        "matches": int(len(x)),
        "trailing": n,
        "trail_rate": round(n / len(x), 4) if len(x) else None,
        "w": w, "d": d, "l": l,
        "win_rate": round(w / n, 4) if n else None,
        "pts_per_trailing": round((3 * w + d) / n, 3) if n else None,
    }


def matrix_counts(x: pd.DataFrame) -> dict:
    """Half-time state x full-time result counts, e.g. {"trail": {"W": 7, "D": 9, "L": 40}}."""
    ct = pd.crosstab(x.ht, x.ft)
    return {
        ht: {ft: int(ct.at[ht, ft]) if ht in ct.index and ft in ct.columns else 0 for ft in ("W", "D", "L")}
        for ht in ("lead", "level", "trail")
    }


@app.get("/api/health")
def health():
    return {"ok": True, "team_matches": int(len(A))}


@app.get("/api/meta")
def meta():
    """Seasons, teams and defaults the dashboard needs to start."""
    seasons = [
        {"season": s, "label": label(s), "partial": s == LATEST_PARTIAL}
        for s in range(SEASON_MIN, SEASON_MAX + 1)
    ]
    per_team = A.groupby("team").agg(matches=("match_id", "size"), seasons=("season", "nunique"))
    teams = [
        {"team": t, "matches": int(r.matches), "seasons": int(r.seasons)}
        for t, r in per_team.sort_index().iterrows()
    ]
    return {
        "seasons": seasons,
        "season_min": SEASON_MIN,
        "season_max": SEASON_MAX,
        "partial_season": LATEST_PARTIAL,
        "teams": teams,
        "default_team": "Real Madrid",
        "source": "football-data.co.uk, La Liga (SP1) results with half-time scores",
    }


@lru_cache(maxsize=512)
def _summary(season_min: int, season_max: int, venue: str) -> dict:
    x = subset(season_min, season_max, venue)  # type: ignore[arg-type]
    teams = []
    for team, g in x.groupby("team"):
        row = outcome_counts(g)
        row["team"] = team
        row["seasons"] = int(g.season.nunique())
        teams.append(row)
    return {
        "season_min": season_min,
        "season_max": season_max,
        "venue": venue,
        "league": outcome_counts(x),
        "league_matrix": matrix_counts(x),
        "teams": teams,
    }


@app.get("/api/teams/summary")
def teams_summary(
    season_min: int = Query(SEASON_MIN, ge=SEASON_MIN, le=SEASON_MAX),
    season_max: int = Query(SEASON_MAX, ge=SEASON_MIN, le=SEASON_MAX),
    venue: Venue = "all",
):
    """Every team's half-time trailing record over a season range (server-side aggregation)."""
    check_range(season_min, season_max)
    return _summary(season_min, season_max, venue)


@app.get("/api/teams/{team}/appearances")
def team_appearances(team: str):
    """All of one team's matches with half-time and full-time scores (both venues)."""
    if team not in TEAMS:
        raise HTTPException(404, f"Unknown team {team!r}. See /api/meta for the list.")
    x = A[A.team == team]
    cols = ["match_id", "date", "season", "opponent", "venue", "gf_ht", "ga_ht", "gf", "ga", "ht", "ft"]
    return {"team": team, "rows": x[cols].to_dict(orient="records")}


@lru_cache(maxsize=4)
def _league_seasons(venue: str) -> list[dict]:
    x = A if venue == "all" else A[A.venue == venue]
    out = []
    for s, g in x.groupby("season"):
        row = outcome_counts(g)
        row["season"] = int(s)
        out.append(row)
    return out


@app.get("/api/league/seasons")
def league_seasons(venue: Venue = "all"):
    """League-wide baseline per season: how often teams trail at half-time and what follows."""
    return {"venue": venue, "seasons": _league_seasons(venue)}
