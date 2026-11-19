import numpy as np
import pandas as pd
import pytest

from nfl_model.data import team_game_stats
from nfl_model.features import FEATURES, _elo, build_features
from nfl_model.model import MarginModel, metrics, vegas_proba


# ---------- play-by-play aggregation ----------

def _pbp():
    rows = []
    # HOM offense: three live plays (EPA 0.3, 0.0, -0.3) and one garbage-time play (EPA 5)
    for epa, wp, passer in ((0.3, 0.5, "QB1"), (0.0, 0.6, "QB1"), (-0.3, 0.4, "QB2"), (5.0, 0.99, "QB1")):
        rows.append(dict(game_id="g1", season=2026, week=1, season_type="REG", posteam="HOM", defteam="AWY",
                         play_type="pass", epa=epa, success=float(epa > 0), **{"pass": 1, "rush": 0},
                         qb_dropback=1, wp=wp, passer_player_id=passer, passer_player_name=passer,
                         qb_epa=epa, special_teams_play=0, interception=0, fumble_lost=0, sack=0))
    # AWY offense: one run, plus a punt that must be ignored
    rows.append(dict(rows[0], posteam="AWY", defteam="HOM", play_type="run", epa=0.2, **{"pass": 0, "rush": 1},
                     qb_dropback=0, passer_player_id=None, passer_player_name=None, qb_epa=None))
    rows.append(dict(rows[0], posteam="AWY", defteam="HOM", play_type="punt", epa=-1.0))
    return pd.DataFrame(rows)


def test_team_game_stats_drops_garbage_time_and_non_plays():
    tg = team_game_stats(_pbp()).set_index("team")
    assert tg.loc["HOM", "off_plays"] == 3
    assert tg.loc["HOM", "off_epa"] == pytest.approx(0.0)
    assert tg.loc["AWY", "def_epa"] == pytest.approx(0.0)      # AWY defence faced HOM's live plays
    assert tg.loc["AWY", "off_epa"] == pytest.approx(0.2)      # punt excluded
    assert tg.loc["HOM", "qb_id"] == "QB1"                     # most dropbacks (incl. garbage time)


# ---------- Elo ----------

def test_elo_is_zero_sum_and_rewards_winner(league):
    games, _ = league
    g = games[games.season == 2016].copy()
    elo = _elo(g)
    first = elo.iloc[0]
    assert first.home_elo == first.away_elo == 1505
    # after the whole season, ratings still average 1505 (each update is zero-sum)
    final = {}
    for r in g.merge(elo, on="game_id").itertuples():
        final[r.home_team], final[r.away_team] = r.home_elo, r.away_elo
    assert np.mean(list(final.values())) == pytest.approx(1505, abs=1e-6)
    assert final["AAA"] > final["DDD"]


# ---------- leakage ----------

def test_features_do_not_use_future_results(league):
    games, tg = league
    full = build_features(games, tg)
    cut = (games.season == 2018) & (games.week >= 6)
    hidden = games.copy()
    hidden.loc[cut, ["home_score", "away_score", "result"]] = np.nan
    trunc = build_features(hidden, tg[~((tg.season == 2018) & (tg.week >= 6))])
    a = full[(full.season == 2018) & (full.week == 6)].set_index("game_id")[FEATURES].sort_index()
    b = trunc[(trunc.season == 2018) & (trunc.week == 6)].set_index("game_id")[FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_ratings_order_teams_by_true_strength(league):
    games, tg = league
    f = build_features(games, tg)
    last = f[(f.season == 2018) & (f.week == 12)]
    r = pd.concat([last.set_index("home_team").home_r_off_epa, last.set_index("away_team").away_r_off_epa])
    assert r["AAA"] > r["DDD"]


# ---------- model ----------

def test_margin_model_recovers_linear_signal():
    rng = np.random.default_rng(1)
    n = 4000
    df = pd.DataFrame({k: rng.normal(size=n) for k in FEATURES})
    df["home_field"] = 1
    df["result"] = 3 * df.d_elo - 2 * df.d_qb + 2.0 + rng.normal(0, 5, n)
    m = MarginModel(alpha=0.01).fit(df)
    c = m.coefficients()
    assert c["d_elo"] == pytest.approx(3, abs=0.2)
    assert c["d_qb"] == pytest.approx(-2, abs=0.2)
    assert m.sigma == pytest.approx(5, abs=0.3)
    p = m.predict_proba(df)
    assert ((p > 0) & (p < 1)).all()


def test_model_beats_coin_flip_on_synthetic_league(league):
    games, tg = league
    f = build_features(games, tg)
    train, test = f[f.season == 2017], f[f.season == 2018]
    m = MarginModel().fit(train)
    assert metrics(test, m.predict_proba(test))["log_loss"] < np.log(2)


def test_vegas_proba():
    df = pd.DataFrame({"spread_line": [-7.0, 0.0, 7.0]})
    p = vegas_proba(df)
    assert p[1] == pytest.approx(0.5)
    assert p[0] == pytest.approx(1 - p[2])
    assert p[2] > 0.65


def test_metrics_known_values():
    df = pd.DataFrame({"result": [3, -3, 7, 0]})
    out = metrics(df, np.array([0.8, 0.4, 0.3, 0.5]))
    assert out["accuracy"] == pytest.approx(2 / 3)    # tie excluded
    assert out["brier"] == pytest.approx((0.04 + 0.16 + 0.49 + 0.25) / 4)
