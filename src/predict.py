"""
Saudi Pro League - Match Outcome Predictor
Predicts match results using only pre-match team statistics (no score leakage)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_data(path="C:\\Users\\sheed\\Documents\\github\\saudi-pro-league-analyzer\\data\\spl_matches.csv"):
    df = pd.read_csv(path)

    # Get match results only
    matches = df[df["section"] == "result"].copy()

    # Get team standings
    standings = df[df["section"] == "standings_overall"][
        ["team", "points", "win_pct", "avg_scored", "avg_conceded"]
    ].copy()

    # Merge home team stats
    matches = matches.merge(standings, left_on="home_team", right_on="team")
    matches = matches.rename(columns={
    "points_x": "home_points",
    "win_pct_x": "home_win_pct",
    "avg_scored_x": "home_avg_scored",
    "avg_conceded_x": "home_avg_conceded",
    "points_y": "away_points",
    "win_pct_y": "away_win_pct",
    "avg_scored_y": "away_avg_scored",
    "avg_conceded_y": "away_avg_conceded"
})
    matches = matches.drop(columns=["team"], errors="ignore")

    # Merge away team stats
    matches = matches.merge(standings, left_on="away_team", right_on="team", suffixes=("", "_away"))
    matches = matches.rename(columns={
    "points_x": "home_points",
    "win_pct_x": "home_win_pct",
    "avg_scored_x": "home_avg_scored",
    "avg_conceded_x": "home_avg_conceded",
    "points_y": "away_points",
    "win_pct_y": "away_win_pct",
    "avg_scored_y": "away_avg_scored",
    "avg_conceded_y": "away_avg_conceded"
})
    matches = matches.drop(columns=["team"], errors="ignore")

    return matches


def prepare_features(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df["home_team_enc"] = le_home.fit_transform(df["home_team"])
    df["away_team_enc"] = le_away.fit_transform(df["away_team"])
    df["result_enc"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    return df


def train_model(df):
    X = df[[
        "home_team_enc", "away_team_enc",
        "home_points", "away_points",
        "home_win_pct", "away_win_pct",
        "home_avg_scored", "away_avg_scored",
        "home_avg_conceded", "away_avg_conceded"
    ]]
    y = df["result_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(
        y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]
    ))
    return model


def main():
    df = load_data()
    print(f"Loaded {len(df)} matches")
    print("\nResult distribution:")
    print(df["result"].value_counts())
    df = prepare_features(df)
    print(df.columns.tolist())
    train_model(df)


if __name__ == "__main__":
    main()