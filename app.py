import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Pro League Analyzer",
    page_icon="⚽",
    layout="wide"
)

# ── Load & prepare data ──────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("data/spl_matches.csv")

    matches = df[df["section"] == "result"].copy()
    
    standings = df[df["section"] == "standings_overall"][
        ["team", "points", "win_pct", "avg_scored", "avg_conceded"]
    ].copy()

    # Rename BEFORE merging so no suffix conflicts
    home_stats = standings.rename(columns={
        "team": "home_team",
        "points": "home_points",
        "win_pct": "home_win_pct",
        "avg_scored": "home_avg_scored",
        "avg_conceded": "home_avg_conceded"
    })

    away_stats = standings.rename(columns={
        "team": "away_team",
        "points": "away_points",
        "win_pct": "away_win_pct",
        "avg_scored": "away_avg_scored",
        "avg_conceded": "away_avg_conceded"
    })

    matches = matches.merge(home_stats, on="home_team")
    matches = matches.merge(away_stats, on="away_team")

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    matches["home_team_enc"] = le_home.fit_transform(matches["home_team"])
    matches["away_team_enc"] = le_away.fit_transform(matches["away_team"])
    matches["result_enc"] = matches["result"].map({"H": 0, "D": 1, "A": 2})

    X = matches[[
        "home_team_enc", "away_team_enc",
        "home_points", "away_points",
        "home_win_pct", "away_win_pct",
        "home_avg_scored", "away_avg_scored",
        "home_avg_conceded", "away_avg_conceded"
    ]]
    y = matches["result_enc"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, matches, standings, le_home, le_away

model, matches, standings, le_home, le_away = load_and_train()
teams = sorted(matches["home_team"].unique())

# ── Header ───────────────────────────────────────────────
st.title("⚽ Saudi Pro League Analyzer")
st.markdown("Predict match outcomes and explore team stats — 2025/26 Season")
st.divider()

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "📊 Team Stats", "📈 League Overview"])

# ── TAB 1: Predictor ─────────────────────────────────────
with tab1:
    st.subheader("Predict a Match Result")
    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox("🏠 Home Team", teams, index=teams.index("Al Hilal"))
    with col2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("✈️ Away Team", away_options)

    if st.button("Predict", use_container_width=True):
        if home == away:
            st.warning("Please select two different teams.")
        else:
            h_stats = standings[standings["team"] == home].iloc[0]
            a_stats = standings[standings["team"] == away].iloc[0]

            try:
                h_enc = le_home.transform([home])[0]
                a_enc = le_away.transform([away])[0]
            except:
                st.error("Team encoding error. Make sure both teams have match data.")
                st.stop()

            features = [[
                h_enc, a_enc,
                h_stats["points"], a_stats["points"],
                h_stats["win_pct"], a_stats["win_pct"],
                h_stats["avg_scored"], a_stats["avg_scored"],
                h_stats["avg_conceded"], a_stats["avg_conceded"]
            ]]

            proba = model.predict_proba(features)[0]
            labels = ["🏠 Home Win", "🤝 Draw", "✈️ Away Win"]

            st.markdown("### Prediction")
            c1, c2, c3 = st.columns(3)
            c1.metric(labels[0], f"{proba[0]*100:.0f}%")
            c2.metric(labels[1], f"{proba[1]*100:.0f}%")
            c3.metric(labels[2], f"{proba[2]*100:.0f}%")

            winner_idx = proba.argmax()
            st.success(f"Most likely result: **{labels[winner_idx]}**")

# ── TAB 2: Team Stats ────────────────────────────────────
with tab2:
    st.subheader("Team Statistics")
    selected = st.selectbox("Select a team", teams)
    t = standings[standings["team"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", int(t["points"]))
    c2.metric("Win %", f"{t['win_pct']:.1f}%")
    c3.metric("Avg Scored", f"{t['avg_scored']:.2f}")
    c4.metric("Avg Conceded", f"{t['avg_conceded']:.2f}")

    team_matches = matches[(matches["home_team"] == selected) | (matches["away_team"] == selected)]
    results = team_matches["result"].value_counts().rename({"H": "Home Win", "A": "Away Win", "D": "Draw"})

    fig, ax = plt.subplots(figsize=(5, 3))
    results.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c", "#3498db"])
    ax.set_title(f"{selected} — Match Results")
    ax.set_xlabel("")
    ax.set_ylabel("Matches")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ── TAB 3: League Overview ───────────────────────────────
with tab3:
    st.subheader("League Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Teams by Points**")
        top = standings.sort_values("points", ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.barplot(data=top, x="points", y="team", palette="Blues_r", ax=ax1)
        ax1.set_xlabel("Points")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Result Distribution (All Matches)**")
        dist = matches["result"].value_counts().rename({"H": "Home Win", "A": "Away Win", "D": "Draw"})
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(dist, labels=dist.index, autopct="%1.0f%%",
                colors=["#2ecc71", "#3498db", "#e74c3c"], startangle=90)
        st.pyplot(fig2)

    st.markdown("**Full Standings**")
    st.dataframe(
        standings.sort_values("points", ascending=False).reset_index(drop=True),
        use_container_width=True
    )