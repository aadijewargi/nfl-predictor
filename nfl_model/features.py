"""Leak-free pre-game features.

For every game, each team's ratings are computed only from games that kicked off
*before* it. Ratings are recency-weighted season-to-date averages, shrunk toward a
regressed version of the team's previous-season rating (strong early in the
season, fading as games accumulate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass_epa", "def_pass_epa",
         "off_rush_epa", "def_rush_epa", "pts_for", "pts_against"]

DECAY = 0.90         # weight multiplier per game back in time
PRIOR_GAMES = 4.0    # previous-season prior counts as this many games
REGRESS = 0.55       # keep this fraction of last season's deviation from average
OPP_ADJUST = True    # also build opponent-adjusted EPA ratings
OPP_WEIGHT = 1.0     # how much of the opponent's deviation to remove
ADJ_STATS = ["off_epa_adj", "def_epa_adj"]
QB_HALF_LIFE = 365.0  # days; older QB games count half as much per this many days
QB_PRIOR_MEAN = -0.05 # EPA/dropback a QB with no history is assumed to have (below average)
QB_PRIOR_N = 300.0    # ...worth this many dropbacks of evidence

# ---- Elo (FiveThirtyEight-style NFL Elo) ----
ELO_K = 20.0
ELO_HFA = 48.0
ELO_CARRY = 2 / 3    # keep 2/3 of rating between seasons


def _long_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) with points for/against."""
    g = games.copy()
    home = g.assign(team=g.home_team, opp=g.away_team, is_home=1,
                    pts_for=g.home_score, pts_against=g.away_score)
    away = g.assign(team=g.away_team, opp=g.home_team, is_home=0,
                    pts_for=g.away_score, pts_against=g.home_score)
    cols = ["game_id", "season", "week", "game_type", "gameday", "team", "opp", "is_home",
            "pts_for", "pts_against"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def _opponent_adjust(long: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-game EPA corrected for the opponent's pre-game strength.

    Gaining 0.10 EPA/play against a defense that usually allows -0.05 is better than
    the same 0.10 against one that allows +0.10. Uses the opponent's rating going
    into the game, so nothing from the future leaks in.
    """
    opp = ratings[["game_id", "team", "r_off_epa", "r_def_epa"]].rename(
        columns={"team": "opp", "r_off_epa": "opp_off", "r_def_epa": "opp_def"})
    out = long.merge(opp, on=["game_id", "opp"], how="left")
    # league average EPA per play is close to 0 in this dataset; centre per season to be safe
    mean_off = out.groupby("season").opp_off.transform("mean")
    mean_def = out.groupby("season").opp_def.transform("mean")
    out["off_epa_adj"] = out.off_epa - OPP_WEIGHT * (out.opp_def - mean_def)
    out["def_epa_adj"] = out.def_epa - OPP_WEIGHT * (out.opp_off - mean_off)
    return out


def _rolling_ratings(long: pd.DataFrame, stats=None) -> pd.DataFrame:
    """Pre-game rating for each team-game row, plus an end-of-season row per team."""
    STATS = stats or globals()["STATS"]  # noqa: N806
    out_rows = []
    season_end = {}  # (team, season) -> final ratings
    first = long.season.min()
    for season in sorted(long.season.unique()):
        # league average from completed earlier seasons only (no peeking at this season)
        hist = long[long.played & (long.season < season)]
        league_mean = (hist if len(hist) else long[long.played & (long.season == first)])[STATS].mean()
        s = long[long.season == season]
        for team, tg in s.groupby("team"):
            tg = tg.sort_values("gameday")
            last = season_end.get((team, season - 1))
            prior = league_mean if last is None else league_mean + REGRESS * (last - league_mean)
            num = prior * PRIOR_GAMES
            den = PRIOR_GAMES
            n_played = 0
            for _, row in tg.iterrows():
                rating = num / den
                out_rows.append({"game_id": row.game_id, "team": team, "games_played": n_played,
                                 **{f"r_{k}": rating[k] for k in STATS}})
                if row.played:
                    x = row[STATS].astype(float)
                    x = x.fillna(rating)          # e.g. a game with no rushing plays
                    num = num * DECAY + x
                    den = den * DECAY + 1.0
                    # the prior also decays, so it fades as the season goes on
                    n_played += 1
            season_end[(team, season)] = num / den
    return pd.DataFrame(out_rows), season_end


def _elo(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo for home and away team in every game."""
    elo: dict[str, float] = {}
    rows = []
    cur_season = None
    for g in games.sort_values(["gameday", "game_id"]).itertuples():
        if g.season != cur_season:
            elo = {t: 1505 + ELO_CARRY * (r - 1505) for t, r in elo.items()}
            cur_season = g.season
        h, a = elo.get(g.home_team, 1505.0), elo.get(g.away_team, 1505.0)
        rows.append({"game_id": g.game_id, "home_elo": h, "away_elo": a})
        if pd.isna(g.result):
            continue
        hfa = 0.0 if g.location == "Neutral" else ELO_HFA
        diff = h + hfa - a
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        actual = 1.0 if g.result > 0 else 0.0 if g.result < 0 else 0.5
        mov = np.log(abs(g.result) + 1) * (2.2 / ((diff if g.result > 0 else -diff) * 0.001 + 2.2))
        delta = ELO_K * mov * (actual - exp_home)
        elo[g.home_team] = h + delta
        elo[g.away_team] = a - delta
    return pd.DataFrame(rows)


def _qb_ratings(team_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted EPA/dropback of each QB, from games before the given date.

    Returns, for each team-game, the rating of that game's starting QB (or for
    future games, the team's most recent starter, or the listed QB if nflverse has one).
    """
    qb = team_games.dropna(subset=["qb_id"]).merge(games[["game_id", "gameday"]], on="game_id")
    qb = qb.sort_values("gameday")
    prior_mean, prior_n = QB_PRIOR_MEAN, QB_PRIOR_N
    hist: dict[str, list] = {}
    rating_at = {}
    for r in qb.itertuples():
        h = hist.get(r.qb_id, [])
        rating_at[(r.game_id, r.team)] = _qb_value(h, r.gameday, prior_mean, prior_n)
        h.append((r.gameday, r.qb_dropbacks, r.qb_epa))
        hist[r.qb_id] = h
    return rating_at, hist, (prior_mean, prior_n)


def _qb_value(h, when, prior_mean, prior_n):
    if not h:
        return prior_mean
    d = np.array([(when - t).days for t, _, _ in h], float)
    n = np.array([n for _, n, _ in h], float)
    e = np.array([e for _, _, e in h], float)
    w = n * 0.5 ** (d / QB_HALF_LIFE)
    return float((w @ e + prior_mean * prior_n) / (w.sum() + prior_n))


def build_features(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    games = games[games.season >= team_games.season.min()].copy()
    games["played"] = games.result.notna()
    long = _long_games(games)
    long["played"] = long.pts_for.notna()
    long = long.merge(team_games.drop(columns=["season", "week"]), on=["game_id", "team"], how="left")
    ratings, _ = _rolling_ratings(long)
    if OPP_ADJUST:
        long = _opponent_adjust(long, ratings)
        adj, _ = _rolling_ratings(long, stats=ADJ_STATS)
        ratings = ratings.merge(adj.drop(columns=["games_played"]), on=["game_id", "team"])
    elo = _elo(games)
    _, qb_hist, (pm, pn) = _qb_ratings(team_games, games)
    qb_name_by_id = dict(zip(team_games.qb_id, team_games.qb_name))
    qb_name_by_id.update(dict(zip(games.home_qb_id, games.home_qb_name)))
    qb_name_by_id.update(dict(zip(games.away_qb_id, games.away_qb_name)))

    # Starting QB for each team-game, using only what is known before kickoff:
    # the starter listed in the schedule data if there is one, otherwise the
    # team's most recent known starter (listed or, failing that, most dropbacks).
    actual = {(r.game_id, r.team): r.qb_id for r in team_games.dropna(subset=["qb_id"]).itertuples()}
    seq = pd.concat([
        games[["game_id", "gameday", "home_team", "home_qb_id"]].set_axis(["game_id", "gameday", "team", "listed"], axis=1),
        games[["game_id", "gameday", "away_team", "away_qb_id"]].set_axis(["game_id", "gameday", "team", "listed"], axis=1),
    ]).sort_values(["gameday", "game_id"])
    starter, last_known = {}, {}
    for r in seq.itertuples():
        listed = r.listed if isinstance(r.listed, str) else None
        qid = listed or last_known.get(r.team)
        starter[(r.game_id, r.team)] = qid
        known_after = listed or actual.get((r.game_id, r.team))
        if known_after:
            last_known[r.team] = known_after

    def qb_for(game_id, team, gameday):
        qid = starter.get((game_id, team))
        return _qb_value([x for x in qb_hist.get(qid, []) if x[0] < gameday], gameday, pm, pn)

    f = games.merge(elo, on="game_id")
    for side in ("home", "away"):
        r = ratings.rename(columns={c: f"{side}_{c}" for c in ratings.columns if c != "game_id"})
        f = f.merge(r, left_on=["game_id", f"{side}_team"], right_on=["game_id", f"{side}_team"], how="left")
        f[f"{side}_qb"] = [qb_for(g, t, d) for g, t, d in zip(f.game_id, f[f"{side}_team"], f.gameday)]
        f[f"upcoming_{side}_qb"] = [qb_name_by_id.get(starter.get((g, t))) for g, t in zip(f.game_id, f[f"{side}_team"])]
    if OPP_ADJUST:
        f["d_off_epa_adj"] = f.home_r_off_epa_adj - f.away_r_off_epa_adj
        f["d_def_epa_adj"] = -(f.home_r_def_epa_adj - f.away_r_def_epa_adj)
    f["outdoor"] = f.roof.isin(["outdoors", "open"]).astype(int)
    f["wind_x"] = f.wind.fillna(0) * f.outdoor

    # differences (home minus away). For defense, lower EPA allowed is better, so flip sign.
    f["d_elo"] = f.home_elo - f.away_elo
    for k in STATS:
        f[f"d_{k}"] = f[f"home_r_{k}"] - f[f"away_r_{k}"]
    for k in ("def_epa", "def_sr", "def_pass_epa", "def_rush_epa", "pts_against"):
        f[f"d_{k}"] = -f[f"d_{k}"]
    # matchup: my offense vs your defense
    f["d_matchup_epa"] = (f.home_r_off_epa - f.away_r_def_epa) - (f.away_r_off_epa - f.home_r_def_epa)
    f["d_qb"] = f.home_qb - f.away_qb
    f["d_rest"] = (f.home_rest - f.away_rest).clip(-7, 7)
    f["home_field"] = (f.location != "Neutral").astype(int)
    f["div"] = f.div_game.fillna(0)
    f["min_gp"] = np.minimum(f.home_games_played, f.away_games_played)
    return f


# Chosen by walk-forward validation on 2018-2023 (2024-2026 held out); see experiments.py.
# Success rate, pass/rush splits, points, opponent-adjusted EPA, weather and division
# games were all tried and did not improve held-out log loss.
FEATURES = ["d_elo", "d_off_epa", "d_def_epa", "d_qb", "d_rest", "home_field"]
# Market-aware model: the betting line plus the stats that still carry a little signal.
BLEND_FEATURES = ["spread_line", "d_qb", "d_elo"]
ALL_FEATURES = ["d_elo", "d_off_epa", "d_def_epa", "d_off_sr", "d_def_sr", "d_off_pass_epa",
                "d_def_pass_epa", "d_off_rush_epa", "d_def_rush_epa", "d_pts_for",
                "d_pts_against", "d_qb", "d_rest", "home_field"]
