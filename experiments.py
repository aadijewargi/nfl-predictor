"""Compare model variants with the same walk-forward backtest.

Variants are judged on log loss over 2018-2023 ("tune"); 2024-2026 ("holdout") is
reported to confirm a change helps on seasons that played no part in choosing it.

    python experiments.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data import build_team_games, load_games
from nfl_model.features import build_features
from nfl_model.model import MarginModel, metrics, vegas_proba

TRAIN_FROM = 2017


def walk_forward(f: pd.DataFrame, make_model) -> pd.DataFrame:
    played = f[f.result.notna() & (f.season >= TRAIN_FROM)]
    cur = int(f.season.max())
    preds = []
    for s in range(2018, cur):
        train, test = played[played.season < s], played[played.season == s]
        m = make_model().fit(train)
        preds.append(test.assign(p=m.predict_proba(test)))
    cs = played[played.season == cur]
    for w in sorted(cs.week.unique()):
        train = played[(played.season < cur) | ((played.season == cur) & (played.week < w))]
        test = cs[cs.week == w]
        m = make_model().fit(train)
        preds.append(test.assign(p=m.predict_proba(test)))
    return pd.concat(preds)


def score(bt: pd.DataFrame, col="p") -> dict:
    bt = bt[bt.spread_line.notna()]
    t, h = bt[bt.season <= 2023], bt[bt.season >= 2024]
    mt, mh = metrics(t, t[col].values), metrics(h, h[col].values)
    return {"tune_ll": mt["log_loss"], "tune_acc": mt["accuracy"],
            "hold_ll": mh["log_loss"], "hold_acc": mh["accuracy"]}


VARIANTS = {
    "v1 (raw EPA)": ["d_elo", "d_off_epa", "d_def_epa", "d_qb", "d_rest", "home_field"],
    "opp-adjusted EPA": ["d_elo", "d_off_epa_adj", "d_def_epa_adj", "d_qb", "d_rest", "home_field"],
    "raw + adjusted": ["d_elo", "d_off_epa", "d_def_epa", "d_off_epa_adj", "d_def_epa_adj", "d_qb", "d_rest", "home_field"],
    "adjusted + pass/rush": ["d_elo", "d_off_epa_adj", "d_def_epa_adj", "d_off_pass_epa", "d_def_pass_epa",
                             "d_qb", "d_rest", "home_field"],
    "adjusted + weather/div": ["d_elo", "d_off_epa_adj", "d_def_epa_adj", "d_qb", "d_rest", "home_field", "div"],
}


def main():
    games, tg = load_games(), build_team_games()
    f = build_features(games, tg)
    rows = []
    for name, feats in VARIANTS.items():
        for alpha in (10.0, 100.0):
            bt = walk_forward(f, lambda: MarginModel(alpha=alpha, features=feats))
            rows.append({"variant": name, "alpha": alpha, **score(bt)})
            print(rows[-1], flush=True)
    bt["p_vegas"] = vegas_proba(bt)
    rows.append({"variant": "Vegas favorite", "alpha": None, **score(bt, "p_vegas")})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
