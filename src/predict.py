"""
Saudi Pro League - Match Outcome Predictor & Season Simulator
Uses historical rolling form, head-to-head, and team strength features
(no score leakage from the match being predicted).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────
def load_data(path="data/spl_matches.csv"):
    df = pd.read_csv(path)
    results = df[df["section"] == "results"].copy()

    # Parse date for chronological ordering
    results["date"] = pd.to_datetime(
        results["date_label"].str.extract(r"(\d{1,2} \w{3} \d{4})")[0],
        format="%d %b %Y", errors="coerce"
    )
    results = results.sort_values(["season", "gameweek", "date"]).reset_index(drop=True)
    results["result_enc"] = results["result"].map({"H": 0, "D": 1, "A": 2})

    return results


# ──────────────────────────────────────────────────────────
# 2. BUILD HISTORICAL FEATURES (rolling form, points, etc.)
#    These are computed using ONLY matches played BEFORE
#    the current match — no leakage.
# ──────────────────────────────────────────────────────────
def build_team_history(results):
    """
    For every team, build a chronological log of:
    points earned, goals scored, goals conceded per match.
    Returns a long-format DataFrame with one row per team per match.
    """
    records = []

    for idx, row in results.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]

        if row["result"] == "H":
            h_pts, a_pts = 3, 0
        elif row["result"] == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1

        records.append({"match_idx": idx, "team": h, "is_home": 1,
                         "points": h_pts, "scored": hg, "conceded": ag,
                         "win": 1 if row["result"] == "H" else 0})
        records.append({"match_idx": idx, "team": a, "is_home": 0,
                         "points": a_pts, "scored": ag, "conceded": hg,
                         "win": 1 if row["result"] == "A" else 0})

    team_log = pd.DataFrame(records)
    return team_log


def add_rolling_features(team_log, window=5):
    """
    For each team-match row, compute the rolling average
    (last `window` matches) of points, goals scored, goals
    conceded, win rate — using only PRIOR matches (shifted by 1).
    """
    team_log = team_log.sort_values(["team", "match_idx"]).reset_index(drop=True)

    for col, new_col in [("points", "form_points"),
                          ("scored", "form_scored"),
                          ("conceded", "form_conceded"),
                          ("win", "form_win_rate")]:
        # Shift first so the rolling window never includes the current match
        shifted = team_log.groupby("team")[col].shift(1)
        team_log[new_col] = (
            shifted.groupby(team_log["team"])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Fill first match of each team with league-average defaults
    form_cols = ["form_points", "form_scored", "form_conceded", "form_win_rate"]
    team_log[form_cols] = team_log[form_cols].fillna(team_log[form_cols].mean())

    return team_log


def attach_features(results, team_log):
    """
    Attach home & away team rolling features back onto the
    match-level results table.
    """
    home_feats = team_log[team_log["is_home"] == 1][
        ["match_idx", "form_points", "form_scored", "form_conceded", "form_win_rate"]
    ].rename(columns={
        "form_points": "home_form_points",
        "form_scored": "home_form_scored",
        "form_conceded": "home_form_conceded",
        "form_win_rate": "home_form_win_rate"
    })

    away_feats = team_log[team_log["is_home"] == 0][
        ["match_idx", "form_points", "form_scored", "form_conceded", "form_win_rate"]
    ].rename(columns={
        "form_points": "away_form_points",
        "form_scored": "away_form_scored",
        "form_conceded": "away_form_conceded",
        "form_win_rate": "away_form_win_rate"
    })

    out = results.merge(home_feats, left_index=True, right_on="match_idx")
    out = out.merge(away_feats, on="match_idx")

    return out


# ──────────────────────────────────────────────────────────
# 3. PREPARE FEATURES FOR MODEL
# ──────────────────────────────────────────────────────────
def prepare_features(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()

    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    le_home.fit(all_teams)
    le_away.fit(all_teams)

    df["home_team_enc"] = le_home.transform(df["home_team"])
    df["away_team_enc"] = le_away.transform(df["away_team"])
    return df, le_home, le_away


FEATURE_COLS = [
    "home_team_enc", "away_team_enc",
    "home_form_points", "away_form_points",
    "home_form_scored", "away_form_scored",
    "home_form_conceded", "away_form_conceded",
    "home_form_win_rate", "away_form_win_rate"
]


# ──────────────────────────────────────────────────────────
# 4. TRAIN MODEL
# ──────────────────────────────────────────────────────────
def train_model(df):
    X = df[FEATURE_COLS]
    y = df["result_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(
        y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]
    ))

    return model


# ──────────────────────────────────────────────────────────
# 5. PREDICT NEXT SEASON'S CHAMPION
#    Simulates every team playing every other team home & away
#    (round-robin), using each team's CURRENT form as of the
#    end of the latest season.
# ──────────────────────────────────────────────────────────
def get_latest_form(team_log):
    """Get each team's most recent rolling form values (most recent match)."""
    latest = (
        team_log.sort_values("match_idx")
        .groupby("team")[["form_points", "form_scored", "form_conceded", "form_win_rate"]]
        .last()
        .reset_index()
    )

    # Also need to compute the NEXT form for each team — i.e. the form
    # AFTER incorporating their final season's results. Re-run rolling
    # to get the "next" form using their last `window` matches including
    # the most recent ones.
    return latest


def simulate_next_season(model, latest_form, le_home, le_away, teams):
    """
    Simulate a full round-robin season (each team plays every
    other team home & away) and tally predicted points.
    """
    table = {team: 0 for team in teams}
    sim_results = []

    for home in teams:
        for away in teams:
            if home == away:
                continue

            if home not in le_home.classes_ or away not in le_away.classes_:
                continue

            h_enc = le_home.transform([home])[0]
            a_enc = le_away.transform([away])[0]

            h_form_row = latest_form[latest_form["team"] == home]
            a_form_row = latest_form[latest_form["team"] == away]

            if h_form_row.empty or a_form_row.empty:
                continue

            h_form = h_form_row.iloc[0]
            a_form = a_form_row.iloc[0]

            features = pd.DataFrame([[
                h_enc, a_enc,
                h_form["form_points"], a_form["form_points"],
                h_form["form_scored"], a_form["form_scored"],
                h_form["form_conceded"], a_form["form_conceded"],
                h_form["form_win_rate"], a_form["form_win_rate"]
            ]], columns=FEATURE_COLS)

            pred = model.predict(features)[0]  # 0=Home,1=Draw,2=Away

            if pred == 0:
                table[home] += 3
            elif pred == 1:
                table[home] += 1
                table[away] += 1
            else:
                table[away] += 3

            sim_results.append((home, away, ["H", "D", "A"][pred]))

    standings = pd.Series(table).sort_values(ascending=False)
    return standings, sim_results


# ──────────────────────────────────────────────────────────
# 6. PIPELINE ENTRY POINT
#    Call this from predict.py (CLI) or app.py (Streamlit)
#    to get everything needed for predictions + simulation.
# ──────────────────────────────────────────────────────────
def build_pipeline(path="data/spl_matches.csv"):
    """
    Loads data, builds features, trains the model, and returns
    everything needed to predict matches or simulate a season.
    """
    results = load_data(path)
    team_log = build_team_history(results)
    team_log = add_rolling_features(team_log, window=5)
    df = attach_features(results, team_log)
    df, le_home, le_away = prepare_features(df)
    model = train_model(df)
    latest_form = get_latest_form(team_log)

    latest_season = results["season"].max()
    current_teams = sorted(
        pd.concat([
            results[results["season"] == latest_season]["home_team"],
            results[results["season"] == latest_season]["away_team"]
        ]).unique()
    )

    return {
        "results": results,
        "model": model,
        "le_home": le_home,
        "le_away": le_away,
        "latest_form": latest_form,
        "current_teams": current_teams,
        "latest_season": latest_season,
    }


# ──────────────────────────────────────────────────────────
# 7. MAIN (CLI mode — prints accuracy + predicted table)
# ──────────────────────────────────────────────────────────
def main():
    print("Loading data and training model...")
    pipeline = build_pipeline()

    print("\n" + "=" * 50)
    print("SIMULATING NEXT SEASON")
    print("=" * 50)

    standings, _ = simulate_next_season(
        pipeline["model"], pipeline["latest_form"],
        pipeline["le_home"], pipeline["le_away"], pipeline["current_teams"]
    )

    print(f"\nPredicted final table for the season after {pipeline['latest_season']}:\n")
    for i, (team, pts) in enumerate(standings.items(), start=1):
        print(f"{i:>2}. {team:<15} {pts:>3} pts")

    print(f"\n🏆 Predicted Champion: {standings.index[0]}")


if __name__ == "__main__":
    main()