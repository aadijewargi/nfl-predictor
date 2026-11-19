"""Download nflverse data and collapse play-by-play into one row per team per game.

Sources (all public, from the nflverse project):
  * schedules / results / betting lines: github.com/nflverse/nfldata  (data/games.csv)
  * play-by-play with EPA:               github.com/nflverse/nflverse-data releases (pbp)
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz"
FIRST_SEASON = 2016

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
            "epa", "success", "pass", "rush", "qb_dropback", "wp", "passer_player_id",
            "passer_player_name", "qb_epa", "special_teams_play", "interception",
            "fumble_lost", "sack"]

# nflverse team codes have been stable since 2020; map relocated franchises to current code
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX", "LAR": "LA"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # fall back to curl (respects proxies configured in the shell)
        subprocess.run(["curl", "-sSLf", "-o", str(dest), url], check=True)


def load_games(refresh: bool = False) -> pd.DataFrame:
    path = DATA / "games.csv"
    if refresh or not path.exists():
        _download(GAMES_URL, path)
    g = pd.read_csv(path)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g[g.season >= FIRST_SEASON - 1].reset_index(drop=True)


def load_pbp_season(year: int, refresh: bool = False) -> pd.DataFrame:
    path = DATA / f"pbp_{year}.csv.gz"
    if refresh or not path.exists():
        _download(PBP_URL.format(y=year), path)
    return pd.read_csv(path, usecols=PBP_COLS, low_memory=False)


def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with offensive and defensive efficiency."""
    p = pbp[(pbp.play_type.isin(["pass", "run"])) & pbp.epa.notna() & pbp.posteam.notna()].copy()
    for c in ("posteam", "defteam"):
        p[c] = p[c].replace(TEAM_FIX)
    # drop garbage time: plays where the game is essentially decided
    p["live"] = p.wp.between(0.05, 0.95)
    p = p[p.live]
    p["pass_epa"] = np.where(p["pass"] == 1, p.epa, np.nan)
    p["rush_epa"] = np.where(p["rush"] == 1, p.epa, np.nan)
    p["to"] = p.interception.fillna(0) + p.fumble_lost.fillna(0)

    off = p.groupby(["game_id", "season", "week", "posteam"]).agg(
        off_plays=("epa", "size"), off_epa=("epa", "mean"), off_sr=("success", "mean"),
        off_pass_epa=("pass_epa", "mean"), off_rush_epa=("rush_epa", "mean"),
        off_pass_rate=("pass", "mean"), off_to=("to", "sum"), off_sack=("sack", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(
        def_epa=("epa", "mean"), def_sr=("success", "mean"),
        def_pass_epa=("pass_epa", "mean"), def_rush_epa=("rush_epa", "mean"),
        def_to=("to", "sum"), def_sack=("sack", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    # starting QB = most dropbacks; his EPA/dropback this game
    db = pbp[(pbp.qb_dropback == 1) & pbp.passer_player_id.notna()].copy()
    db["posteam"] = db.posteam.replace(TEAM_FIX)
    qb = (db.groupby(["game_id", "posteam", "passer_player_id", "passer_player_name"])
            .agg(n=("qb_epa", "size"), qb_epa=("qb_epa", "mean")).reset_index()
            .sort_values("n").groupby(["game_id", "posteam"]).tail(1)
            .rename(columns={"posteam": "team", "passer_player_id": "qb_id",
                             "passer_player_name": "qb_name", "n": "qb_dropbacks"}))
    out = off.merge(dfn, on=["game_id", "team"], how="left").merge(qb, on=["game_id", "team"], how="left")
    return out


def build_team_games(seasons=None, refresh_current: bool = False) -> pd.DataFrame:
    games = load_games(refresh=refresh_current)
    seasons = seasons or sorted(games.season.unique())
    cur = int(games.season.max())
    frames = []
    for y in seasons:
        if y < FIRST_SEASON:
            continue
        cache = DATA / f"team_games_{y}.csv"
        if cache.exists() and not (refresh_current and y == cur):
            frames.append(pd.read_csv(cache))
            continue
        tg = team_game_stats(load_pbp_season(y, refresh=refresh_current and y == cur))
        tg.to_csv(cache, index=False)
        frames.append(tg)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    tg = build_team_games()
    print(tg.shape)
    print(tg.tail())
