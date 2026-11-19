"""Small synthetic league so the tests run offline in seconds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
STRENGTH = {"AAA": 7.0, "BBB": 2.0, "CCC": -2.0, "DDD": -7.0}


def make_league(seasons=(2016, 2017, 2018), weeks=12, seed=0):
    rng = np.random.default_rng(seed)
    games, team_games = [], []
    for s in seasons:
        start = pd.Timestamp(f"{s}-09-10")
        for w in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, 4, 2):
                h, a = order[i], order[i + 1]
                gid = f"{s}_{w:02d}_{a}_{h}"
                margin = STRENGTH[h] - STRENGTH[a] + 2 + rng.normal(0, 10)
                hs = int(max(0, round(21 + margin / 2)))
                as_ = int(max(0, round(21 - margin / 2)))
                games.append(dict(
                    game_id=gid, season=s, game_type="REG", week=w,
                    gameday=start + pd.Timedelta(days=7 * (w - 1)), gametime="13:00",
                    away_team=a, home_team=h, away_score=as_, home_score=hs, result=hs - as_,
                    location="Home", home_rest=7, away_rest=7, div_game=0,
                    spread_line=round(STRENGTH[h] - STRENGTH[a] + 2, 1), roof="outdoors", wind=5,
                    home_qb_id=f"QB_{h}", away_qb_id=f"QB_{a}", home_qb_name=f"{h} QB", away_qb_name=f"{a} QB"))
                for t, o, sign in ((h, a, 1), (a, h, -1)):
                    e = 0.02 * (STRENGTH[t] - STRENGTH[o]) + rng.normal(0, 0.1)
                    team_games.append(dict(
                        game_id=gid, season=s, week=w, team=t, off_plays=60, off_epa=e,
                        off_sr=0.45 + e / 2, off_pass_epa=e, off_rush_epa=e, off_pass_rate=0.6,
                        off_to=1, off_sack=0.06, def_epa=-e, def_sr=0.45 - e / 2, def_pass_epa=-e,
                        def_rush_epa=-e, def_to=1, def_sack=0.06, qb_id=f"QB_{t}", qb_name=f"{t} QB",
                        qb_dropbacks=35, qb_epa=e + 0.05))
    return pd.DataFrame(games), pd.DataFrame(team_games)


@pytest.fixture(scope="session")
def league():
    return make_league()
