"""Walk-forward backtest of both models against the Vegas favorite.

* Each past season S (2018-2025): train on games from seasons before S, predict
  every game of S using only pre-game features.
* Current season: for each week W, train on all games before W, predict week W.

Models:
  stats  - team efficiency, Elo, starting QB, rest, home field (no betting data)
  blend  - the closing spread plus QB and Elo differences
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_model.data import build_team_games, load_games
from nfl_model.features import BLEND_FEATURES, FEATURES, build_features
from nfl_model.model import MarginModel, metrics, vegas_proba

OUT = Path(__file__).resolve().parent / "output"
TRAIN_FROM = 2017   # 2016 has no previous-season prior
FIRST_TEST = 2018


def walk_forward(f: pd.DataFrame, features, alpha: float = 10.0) -> pd.DataFrame:
    played = f[f.result.notna() & (f.season >= TRAIN_FROM)]
    cur = int(f.season.max())
    splits = [(played.season < s, played.season == s) for s in range(FIRST_TEST, cur)]
    cs = played.season == cur
    splits += [((played.season < cur) | (cs & (played.week < w)), cs & (played.week == w))
               for w in sorted(played[cs].week.unique())]
    preds = []
    for tr, te in splits:
        train, test = played[tr], played[te]
        m = MarginModel(alpha=alpha, features=features).fit(train)
        preds.append(test.assign(p=m.predict_proba(test), margin=m.predict_margin(test)))
    return pd.concat(preds)


def run(f: pd.DataFrame | None = None, quiet: bool = False):
    if f is None:
        f = build_features(load_games(), build_team_games())
    OUT.mkdir(exist_ok=True)
    lined = f[f.spread_line.notna()]
    stats = walk_forward(f, FEATURES)
    blend = walk_forward(lined, BLEND_FEATURES)
    bt = stats[stats.spread_line.notna()].rename(columns={"p": "p_model", "margin": "m_model"})
    bt = bt.merge(blend[["game_id", "p", "margin"]].rename(columns={"p": "p_blend", "margin": "m_blend"}), on="game_id")
    bt["p_vegas"] = vegas_proba(bt)

    rows = []
    for s, d in list(bt.groupby("season")) + [("all", bt)]:
        for who, p, m in (("model", "p_model", "m_model"), ("blend", "p_blend", "m_blend"),
                          ("vegas", "p_vegas", "spread_line")):
            rows.append({"season": str(s), "who": who, **metrics(d, d[p].values, d[m].values)})
    res = pd.DataFrame(rows)
    if not quiet:
        print(res.pivot(index="season", columns="who", values=["accuracy", "log_loss"]).round(3).to_string())
    bt.to_csv(OUT / "backtest_predictions.csv", index=False)
    res.to_csv(OUT / "backtest_summary.csv", index=False)
    return res, bt


if __name__ == "__main__":
    run()
