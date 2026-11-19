"""Write the dashboard's data files (JSON {rows, meta}) from the model outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
DASH = ROOT / "dashboard"
DASH.mkdir(exist_ok=True)

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}


def code(*names):
    return [{"label": n, "value": (ROOT / n).read_text(), "type": "code", "lang": "python"} for n in names]


def clean(rows):
    out = []
    for r in rows:
        out.append({k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in r.items()})
    return out


def dump(name, df, meta):
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4)
    (DASH / name).write_text(json.dumps({"rows": clean(df.to_dict("records")), "meta": meta}, default=str))


def main():
    gen = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M %Z")
    src = {"label": "Input", "value": "nflverse games.csv (schedule, scores, closing lines) and "
                                      "play-by-play 2016-2026 (EPA), through 2026 week 10"}
    gen_m = {"label": "Generated", "value": gen}

    p = pd.read_csv(OUT / "predictions_2026.csv")
    p["gameday"] = p.gameday.str[:10]
    p["away_name"] = p.away_team.map(TEAM_NAMES)
    p["home_name"] = p.home_team.map(TEAM_NAMES)
    p["p_away"] = 1 - p.p_home
    p["vegas_pick"] = np.where(p.spread_line.isna(), None,
                               np.where(p.spread_line > 0, p.home_team,
                                        np.where(p.spread_line < 0, p.away_team, "Pick'em")))
    p["agrees"] = np.where(p.spread_line.isna(), None, p.pick == p.vegas_pick)
    p["game"] = p.away_team + " @ " + p.home_team
    p["key"] = p.week.astype(str) + "-" + p.away_team + "-" + p.home_team
    cols = ["key", "week", "gameday", "gametime", "game", "away_team", "home_team", "away_name", "home_name",
            "upcoming_away_qb", "upcoming_home_qb", "pred_margin", "p_home", "p_away", "pick", "pick_prob",
            "spread_line", "p_vegas_home", "vegas_pick", "agrees", "blend_pick", "blend_prob", "blend_margin"]
    dump("predictions.json", p[cols], code("predict.py", "nfl_model/model.py", "nfl_model/features.py") + [src, gen_m])

    pw = pd.read_csv(OUT / "power_ratings_2026.csv")
    pw["name"] = pw.team.map(TEAM_NAMES)
    pw["rank"] = np.arange(1, len(pw) + 1)
    pw = pw[["rank", "team", "name", "power", "elo", "off_epa", "def_epa", "qb"]]
    dump("power.json", pw, code("predict.py", "nfl_model/features.py") + [src, gen_m])

    s = pd.read_csv(OUT / "backtest_summary.csv")
    wide = s.pivot(index="season", columns="who", values=["accuracy", "log_loss", "brier", "games"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["games"] = wide.games_model.astype(int)
    wide = wide.drop(columns=[c for c in wide.columns if c.startswith("games_")])
    wide["season"] = wide.season.astype(str)
    order = [str(y) for y in range(2018, 2027)] + ["all"]
    wide = wide.set_index("season").loc[order].reset_index()
    wide["label"] = wide.season.replace({"all": "2018–26", "2026": "2026 (wk 1–10)"})
    dump("backtest.json", wide, code("backtest.py", "nfl_model/model.py") + [src, gen_m])

    bt = pd.read_csv(OUT / "backtest_predictions.csv")
    bt = bt[bt.spread_line.notna() & (bt.result != 0)]
    bt["model_right"] = ((bt.p_model > 0.5) == (bt.result > 0)).astype(int)
    bt["vegas_right"] = ((bt.spread_line > 0) == (bt.result > 0)).astype(int)
    bt["blend_right"] = ((bt.p_blend > 0.5) == (bt.result > 0)).astype(int)
    cur = bt[bt.season == 2026].groupby("week").agg(
        games=("model_right", "size"), model_right=("model_right", "sum"),
        blend_right=("blend_right", "sum"), vegas_right=("vegas_right", "sum")).reset_index()
    dump("weekly_2026.json", cur, code("backtest.py") + [src, gen_m])

    bt["conf"] = np.maximum(bt.p_model, 1 - bt.p_model)
    bt["bucket"] = pd.cut(bt.conf, [0.5, 0.6, 0.7, 0.8, 1.0], labels=["50–60%", "60–70%", "70–80%", "80%+"])
    cal = bt.groupby("bucket", observed=True).agg(
        games=("model_right", "size"), predicted=("conf", "mean"), actual=("model_right", "mean")).reset_index()
    cal["bucket"] = cal.bucket.astype(str)
    dump("calibration.json", cal, code("backtest.py") + [src, gen_m])
    print("wrote", sorted(x.name for x in DASH.iterdir()))


if __name__ == "__main__":
    main()
