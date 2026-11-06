"""Build tidy tables from the raw football-data.co.uk La Liga files.

Run once (or whenever the raw files change):
    python clean.py

Inputs : data/raw/SP1_<season>.csv  (one file per season, as downloaded)
Outputs: data/matches.csv      one row per match
         data/appearances.csv  one row per team per match (the team's point of view)

Two quirks in the raw files are handled explicitly:
  1. Some rows carry extra trailing empty fields (e.g. 112 rows in 2004/05).
     pandas would reject or silently skip them, so rows are padded/trimmed
     to the header length with the csv module instead.
  2. The season comes from the file, not from the match date: 2019/20
     finished in July 2020 after the COVID pause, so a date cutoff would
     put 57 of its matches in the wrong season.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"

# football-data abbreviations -> names a reader recognises
DISPLAY_NAMES = {
    "Villareal": "Villarreal",  # spelled both ways across seasons
    "Ath Madrid": "Atlético Madrid",
    "Ath Bilbao": "Athletic Club",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "La Coruna": "Deportivo La Coruña",
    "Sp Gijon": "Sporting Gijón",
    "Celta": "Celta Vigo",
    "Betis": "Real Betis",
    "Santander": "Racing Santander",
    "Hercules": "Hércules",
    "Alaves": "Alavés",
    "Malaga": "Málaga",
    "Almeria": "Almería",
    "Cadiz": "Cádiz",
    "Leganes": "Leganés",
    "Logrones": "Logroñés",
    "Merida": "Mérida",
    "Compostela": "Compostela",
    "Extremadura": "Extremadura",
    "Gimnastic": "Gimnàstic",
    "Xerez": "Xerez",
    "Numancia": "Numancia",
    "Tenerife": "Tenerife",
    "Salamanca": "Salamanca",
    "Albacete": "Albacete",
    "Oviedo": "Real Oviedo",
    "Valladolid": "Real Valladolid",
    "Zaragoza": "Real Zaragoza",
    "Murcia": "Real Murcia",
    "Mallorca": "Mallorca",
    "Granada": "Granada",
    "Huesca": "Huesca",
    "Girona": "Girona",
    "Eibar": "Eibar",
    "Elche": "Elche",
    "Osasuna": "Osasuna",
    "Getafe": "Getafe",
    "Levante": "Levante",
    "Las Palmas": "Las Palmas",
    "Cordoba": "Córdoba",
}

KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]


def parse_date(s: str) -> pd.Timestamp:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {s!r}")


def load_season(path: Path) -> pd.DataFrame:
    with path.open(encoding="latin-1", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if any(x.strip() for x in r)]
    df = pd.DataFrame(body, columns=header)
    df = df.loc[:, [c in KEEP for c in df.columns]]  # also drops blank header cells
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reindex(columns=KEEP)  # older seasons have no HT columns -> NaN
    df["Date"] = df["Date"].map(parse_date)
    first = df["Date"].min()
    df["season"] = first.year if first.month >= 7 else first.year - 1
    for c in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def season_label(y: int) -> str:
    return f"{y}/{str(y + 1)[2:]}"


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(RAW.glob("SP1_*.csv"))
    if not files:
        raise SystemExit(f"No raw files found in {RAW}")
    m = pd.concat([load_season(f) for f in files], ignore_index=True)
    m["HomeTeam"] = m["HomeTeam"].str.strip().replace(DISPLAY_NAMES)
    m["AwayTeam"] = m["AwayTeam"].str.strip().replace(DISPLAY_NAMES)
    m = m.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    m.insert(0, "match_id", range(1, len(m) + 1))

    # --- sanity checks: fail loudly instead of silently losing rows ---
    assert not m.duplicated(["Date", "HomeTeam", "AwayTeam"]).any(), "duplicate matches"
    per_season = m.groupby("season").size()
    teams = m.groupby("season")["HomeTeam"].nunique()
    latest = per_season.index.max()
    for s, n in per_season.items():
        expected = teams[s] * (teams[s] - 1)
        if s != latest:  # the current season is still in progress
            assert n == expected, f"{season_label(s)}: {n} matches, expected {expected}"
    has_ht = m.dropna(subset=["HTHG", "HTAG"])
    assert ((has_ht.HTHG <= has_ht.FTHG) & (has_ht.HTAG <= has_ht.FTAG)).all(), "HT goals exceed FT"

    m["season_label"] = m["season"].map(season_label)

    # --- one row per team per match, from that team's point of view ---
    ht = m.dropna(subset=["HTHG", "HTAG"])
    home = pd.DataFrame({
        "match_id": ht.match_id, "date": ht.Date, "season": ht.season,
        "team": ht.HomeTeam, "opponent": ht.AwayTeam, "venue": "H",
        "gf_ht": ht.HTHG, "ga_ht": ht.HTAG, "gf": ht.FTHG, "ga": ht.FTAG,
    })
    away = pd.DataFrame({
        "match_id": ht.match_id, "date": ht.Date, "season": ht.season,
        "team": ht.AwayTeam, "opponent": ht.HomeTeam, "venue": "A",
        "gf_ht": ht.HTAG, "ga_ht": ht.HTHG, "gf": ht.FTAG, "ga": ht.FTHG,
    })
    a = pd.concat([home, away], ignore_index=True)
    for c in ["gf_ht", "ga_ht", "gf", "ga"]:
        a[c] = a[c].astype(int)
    d_ht = a.gf_ht - a.ga_ht
    d_ft = a.gf - a.ga
    a["ht"] = pd.cut(d_ht, [-99, -1, 0, 99], labels=["trail", "level", "lead"]).astype(str)
    a["ft"] = pd.cut(d_ft, [-99, -1, 0, 99], labels=["L", "D", "W"]).astype(str)
    a["pts"] = a["ft"].map({"W": 3, "D": 1, "L": 0})
    a["date"] = a["date"].dt.strftime("%Y-%m-%d")
    a = a.sort_values(["date", "match_id", "venue"], ascending=[True, True, False])

    m_out = m.rename(columns={"Date": "date"})
    m_out["date"] = m_out["date"].dt.strftime("%Y-%m-%d")
    return m_out, a.reset_index(drop=True)


if __name__ == "__main__":
    matches, appearances = build()
    matches.to_csv(DATA / "matches.csv", index=False)
    appearances.to_csv(DATA / "appearances.csv", index=False)
    print(f"matches.csv      {len(matches):>6} rows, seasons "
          f"{matches.season.min()}-{matches.season.max()}")
    print(f"appearances.csv  {len(appearances):>6} rows (seasons with half-time scores: "
          f"{appearances.season.min()}-{appearances.season.max()})")
    print(f"teams            {appearances.team.nunique():>6}")
