"""Fit on every completed game and predict the rest of the current season.

Usage:  python predict.py            (uses cached data)
        python predict.py --refresh  (re-downloads this season's schedule + play-by-play first)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_model.data import build_team_games, load_games
from nfl_model.features import BLEND_FEATURES, build_features
from nfl_model.model import MarginModel, vegas_proba

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main(refresh: bool = False):
    games = load_games(refresh=refresh)
    tg = build_team_games(refresh_current=refresh)
    f = build_features(games, tg)
    cur = int(f.season.max())
    played = f[f.result.notna() & (f.season >= 2017)]
    upcoming = f[(f.season == cur) & f.result.isna()].copy()

    m = MarginModel().fit(played)
    upcoming["pred_margin"] = m.predict_margin(upcoming)
    upcoming["p_home"] = m.predict_proba(upcoming)
    upcoming["p_vegas_home"] = np.where(upcoming.spread_line.notna(), vegas_proba(upcoming), np.nan)
    upcoming["pick"] = np.where(upcoming.p_home >= 0.5, upcoming.home_team, upcoming.away_team)
    upcoming["pick_prob"] = np.maximum(upcoming.p_home, 1 - upcoming.p_home)

    # market-aware blend, only where a line is posted
    blend = MarginModel(features=BLEND_FEATURES).fit(played[played.spread_line.notna()])
    has = upcoming.spread_line.notna()
    upcoming["p_blend_home"] = np.nan
    upcoming["blend_margin"] = np.nan
    if has.any():
        upcoming.loc[has, "p_blend_home"] = blend.predict_proba(upcoming[has])
        upcoming.loc[has, "blend_margin"] = blend.predict_margin(upcoming[has])
    upcoming["blend_pick"] = np.where(~has, None, np.where(upcoming.p_blend_home >= 0.5, upcoming.home_team,
                                                           upcoming.away_team))
    upcoming["blend_prob"] = np.where(has, np.maximum(upcoming.p_blend_home, 1 - upcoming.p_blend_home), np.nan)
    cols = ["season", "week", "gameday", "gametime", "away_team", "home_team", "upcoming_away_qb",
            "upcoming_home_qb", "pred_margin", "p_home", "pick", "pick_prob", "spread_line",
            "p_vegas_home", "blend_margin", "p_blend_home", "blend_pick", "blend_prob",
            "home_elo", "away_elo", "home_r_off_epa", "home_r_def_epa",
            "away_r_off_epa", "away_r_def_epa"]
    out = upcoming[cols].sort_values(["week", "gameday", "gametime"])
    out.to_csv(OUT / f"predictions_{cur}.csv", index=False)

    # current power ratings: neutral-field margin vs an average team
    last = pd.concat([
        f[f.season == cur][["gameday", "home_team", "home_elo", "home_r_off_epa", "home_r_def_epa", "home_qb"]]
        .set_axis(["gameday", "team", "elo", "off_epa", "def_epa", "qb"], axis=1),
        f[f.season == cur][["gameday", "away_team", "away_elo", "away_r_off_epa", "away_r_def_epa", "away_qb"]]
        .set_axis(["gameday", "team", "elo", "off_epa", "def_epa", "qb"], axis=1),
    ])
    # the first unplayed game holds each team's latest pre-game rating
    nxt = upcoming[["gameday"]].min().iloc[0]
    latest = last[last.gameday >= nxt].sort_values("gameday").groupby("team").head(1).set_index("team")
    mean = latest.mean(numeric_only=True)
    probe = pd.DataFrame({
        "d_elo": latest.elo - mean.elo, "d_off_epa": latest.off_epa - mean.off_epa,
        "d_def_epa": -(latest.def_epa - mean.def_epa), "d_qb": latest.qb - mean.qb,
        "d_rest": 0.0, "home_field": 0,
    })
    # points better than an average team on a neutral field (linear model, so no intercept)
    latest["power"] = probe[m.features].to_numpy(float) @ m.coefficients().to_numpy()
    latest = latest.sort_values("power", ascending=False)
    latest.to_csv(OUT / f"power_ratings_{cur}.csv")
    print(f"sigma={m.sigma:.2f}")
    print(m.coefficients().round(3))
    print(out[out.week == out.week.min()][["away_team", "home_team", "upcoming_away_qb", "upcoming_home_qb",
                                          "pick", "pick_prob", "blend_pick", "blend_prob",
                                          "spread_line"]].round(2).to_string(index=False))
    return out, latest, m


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
