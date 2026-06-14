import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from predict import build_pipeline, simulate_next_season, FEATURE_COLS

# ── Theme ──────────────────────────────────────────────
st.set_page_config(page_title="Saudi Pro League · Data Hub", page_icon="🟢", layout="wide")

GREEN = "#00B140"; GOLD = "#C9A84C"; DARK = "#0D0D0D"
CARD = "#161616"; MUTED = "#777777"; WHITE = "#F0F0F0"
RED = "#CC3333"; BLUE = "#2980b9"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{DARK};color:{WHITE};}}
.block-container{{padding:1.2rem 2rem 3rem;}}
[data-testid="stSidebar"]{{background:#0a0a0a;border-right:1px solid #1e1e1e;}}
[data-testid="stSidebar"] *{{color:{WHITE}!important;}}
[data-testid="metric-container"]{{background:{CARD};border:1px solid #222;border-radius:10px;padding:12px 16px;}}
[data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;}}
[data-testid="stMetricValue"]{{color:{WHITE}!important;font-size:1.6rem;font-weight:800;}}
[data-testid="stTabs"] [role="tab"]{{color:{MUTED};font-weight:600;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;padding:.5rem 1rem;}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{color:{GREEN};border-bottom:2px solid {GREEN};}}
div[data-testid="stSelectbox"] > div > div{{background:{CARD};border:1px solid #2a2a2a;color:{WHITE};}}
.stButton > button{{background:{GREEN};color:#000;font-weight:700;border:none;border-radius:8px;padding:.6rem 1.4rem;font-size:.85rem;letter-spacing:.04em;text-transform:uppercase;}}
.stButton > button:hover{{background:#00cc4a;}}
/* Cards */
.card{{background:{CARD};border:1px solid #222;border-radius:12px;padding:1.2rem 1.5rem;}}
.section-label{{font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{GREEN};margin-bottom:.6rem;}}
/* Hero */
.hero{{background:linear-gradient(120deg,#071a07 0%,#0D0D0D 70%);border:1px solid #1a2e1a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}}
.hero h1{{font-size:2.4rem;font-weight:900;color:{WHITE};margin:0 0 .3rem;letter-spacing:-.02em;}}
.hero .sub{{color:{MUTED};font-size:.88rem;}}
.hero::after{{content:"⚽";position:absolute;right:2rem;top:50%;transform:translateY(-50%);font-size:7rem;opacity:.05;}}
/* Badges */
.badge{{display:inline-block;background:{GREEN};color:#000;font-size:.65rem;font-weight:800;padding:2px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.08em;margin-right:5px;}}
.badge-gold{{background:{GOLD};color:#000;}}
/* Standings */
.srow{{display:flex;align-items:center;padding:9px 14px;border-radius:8px;margin-bottom:3px;border:1px solid #1e1e1e;}}
.srow:hover{{background:#1a1a1a;}}
.spos{{width:26px;font-weight:700;color:{MUTED};font-size:.82rem;}}
.spos.g{{color:{GOLD};}}
.sname{{flex:1;font-weight:600;font-size:.88rem;}}
.spts{{background:#0f2e0f;color:{GREEN};font-weight:700;font-size:.78rem;padding:2px 10px;border-radius:20px;}}
/* Bracket */
.bm{{background:{CARD};border:1px solid #252525;border-radius:8px;padding:10px 13px;margin:5px 0;font-size:.84rem;}}
.bm .bw{{color:{GREEN};font-weight:700;}}
.bm .bl{{color:{MUTED};}}
.brlabel{{font-size:.64rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:{GREEN};margin:1rem 0 .4rem;text-align:center;}}
/* Champion */
.champ{{background:linear-gradient(135deg,#1a1200 0%,#261c00 100%);border:1px solid {GOLD};border-radius:14px;padding:1.8rem 2rem;text-align:center;margin-bottom:1rem;}}
.champ .ctrophy{{font-size:3.2rem;}}
.champ .cname{{font-size:2.1rem;font-weight:900;color:{GOLD};margin:.3rem 0;letter-spacing:-.02em;}}
.champ .csub{{color:{MUTED};font-size:.85rem;}}
/* H2H */
.h2hbox{{background:{CARD};border:1px solid #222;border-radius:10px;padding:1rem;text-align:center;}}
.h2hnum{{font-size:2.2rem;font-weight:800;}}
.h2hlbl{{font-size:.68rem;color:{MUTED};text-transform:uppercase;letter-spacing:.07em;}}
/* Pred */
.predbox{{background:{CARD};border:1px solid #222;border-radius:12px;padding:1.5rem;text-align:center;}}
.predpct{{font-size:2.8rem;font-weight:900;}}
.predlbl{{font-size:.78rem;color:{MUTED};margin-top:.3rem;}}
/* Record */
.reccard{{background:{CARD};border:1px solid #252525;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.6rem;}}
.reccard .rtitle{{font-size:.7rem;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem;}}
.reccard .rval{{font-size:1.1rem;font-weight:700;}}
/* Scroll */
::-webkit-scrollbar{{width:3px;}} ::-webkit-scrollbar-thumb{{background:#2a2a2a;border-radius:4px;}}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading 15 seasons of SPL data…")
def get_pipeline():
    return build_pipeline("data/spl_matches.csv")

pipeline = get_pipeline()
model = pipeline["model"]
le_home = pipeline["le_home"]
le_away = pipeline["le_away"]
latest_form = pipeline["latest_form"]
teams = pipeline["current_teams"]
latest_season = pipeline["latest_season"]
results = pipeline["results"]

@st.cache_data
def get_sim():
    return simulate_next_season(model, latest_form, le_home, le_away, teams)

sim_standings, sim_results_list = get_sim()

# ── Precompute records ───────────────────────────────
@st.cache_data
def compute_records(results_df):
    title_map = {}
    for season in sorted(results_df["season"].unique()):
        s = results_df[results_df["season"] == season]
        pts = {}
        for _, r in s.iterrows():
            h, a = r["home_team"], r["away_team"]
            pts.setdefault(h, 0); pts.setdefault(a, 0)
            if r["result"] == "H": pts[h] += 3
            elif r["result"] == "A": pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
        title_map[season] = max(pts, key=pts.get)

    from collections import Counter
    titles = Counter(title_map.values())

    def get_max_streak(team, kind="W"):
        tm = results_df[(results_df["home_team"] == team) | (results_df["away_team"] == team)].sort_values(["season", "gameweek"])
        best, cur = 0, 0
        for _, r in tm.iterrows():
            is_home = r["home_team"] == team
            if kind == "W":
                hit = (is_home and r["result"] == "H") or (not is_home and r["result"] == "A")
            elif kind == "U":
                hit = not ((is_home and r["result"] == "A") or (not is_home and r["result"] == "H"))
            else:
                hit = False
            if hit: cur += 1; best = max(best, cur)
            else: cur = 0
        return best

    all_t = pd.concat([results_df["home_team"], results_df["away_team"]]).unique()
    win_streaks = {t: get_max_streak(t, "W") for t in all_t}

    biggest_wins = results_df.copy()
    biggest_wins["gdiff"] = abs(biggest_wins["home_goals"] - biggest_wins["away_goals"])
    biggest = biggest_wins.nlargest(10, "gdiff")[["season", "home_team", "away_team", "score", "city", "gdiff"]]

    top_scorers_home = results_df.groupby("home_team")["home_goals"].sum()
    top_scorers_away = results_df.groupby("away_team")["away_goals"].sum()
    total_goals_all = (top_scorers_home.add(top_scorers_away, fill_value=0)).sort_values(ascending=False)

    return title_map, titles, win_streaks, biggest, total_goals_all

title_map, titles, win_streaks, biggest_wins, total_goals_all = compute_records(results)

# ── Helper functions ─────────────────────────────────
def mpl_dark(fig):
    fig.patch.set_facecolor(CARD)
    for ax in fig.axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors="#aaa", labelsize=8.5)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color(WHITE)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")

def predict_match(home, away):
    h = latest_form[latest_form["team"] == home]
    a = latest_form[latest_form["team"] == away]
    if h.empty or a.empty: return None
    h, a = h.iloc[0], a.iloc[0]
    h_enc = le_home.transform([home])[0]
    a_enc = le_away.transform([away])[0]
    feat = pd.DataFrame([[h_enc, a_enc,
        h["form_points"], a["form_points"],
        h["form_scored"], a["form_scored"],
        h["form_conceded"], a["form_conceded"],
        h["form_win_rate"], a["form_win_rate"]]], columns=FEATURE_COLS)
    return model.predict_proba(feat)[0]

def get_team_form_dots(team, n=5):
    s = results[results["season"] == latest_season]
    tm = s[(s["home_team"] == team) | (s["away_team"] == team)].sort_values("gameweek").tail(n)
    dots = ""
    for _, r in tm.iterrows():
        is_home = r["home_team"] == team
        if (is_home and r["result"] == "H") or (not is_home and r["result"] == "A"):
            dots += f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{GREEN};margin:0 2px;"></span>'
        elif r["result"] == "D":
            dots += f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{GOLD};margin:0 2px;"></span>'
        else:
            dots += f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{RED};margin:0 2px;"></span>'
    return dots

def team_season_pts(team, season):
    s = results[results["season"] == season]
    pts = 0
    for _, r in s.iterrows():
        if r["home_team"] == team:
            if r["result"] == "H": pts += 3
            elif r["result"] == "D": pts += 1
        elif r["away_team"] == team:
            if r["result"] == "A": pts += 3
            elif r["result"] == "D": pts += 1
    return pts


# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="section-label">Saudi Pro League</div>', unsafe_allow_html=True)
    st.markdown("## Data Hub")
    st.markdown(f'<span class="badge">ML</span><span class="badge">15 Seasons</span><br><br>', unsafe_allow_html=True)
    st.metric("Total Matches", f"{len(results):,}")
    st.metric("Total Goals", f"{int(results['total_goals'].sum()):,}")
    st.metric("Teams Ever", int(pd.concat([results['home_team'], results['away_team']]).nunique()))
    st.metric("Model Accuracy", "49%  (+57% vs random)")
    st.divider()
    st.markdown(f'<div style="color:{MUTED};font-size:.73rem">🏆 Most titles: Al Hilal (6)<br>⚡ Longest win streak: Al Hilal (24 games)<br>🔥 Highest score: 10 goals (Al Fateh 5–5 Damac)</div>', unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────
champion = sim_standings.index[0]
st.markdown(f"""
<div class="hero">
  <h1>Saudi Pro League Analyzer</h1>
  <div class="sub">Machine learning · 15 seasons · 3,392 matches &nbsp;·&nbsp; Predicted next champion: <strong style="color:{GOLD}">{champion}</strong></div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────
tabs = st.tabs([
    "🔮 Match Predictor",
    "🏆 Next Season",
    "🗂 Knockout Bracket",
    "⚔️ Head to Head",
    "👤 Team Profile",
    "📜 All-Time Records",
    "📈 Analytics",
])
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = tabs


# ════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Pick a fixture</div>', unsafe_allow_html=True)
    c1, c_vs, c2 = st.columns([5,1,5])
    with c1:
        home = st.selectbox("🏠 Home Team", teams, key="ph")
    with c_vs:
        st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:900;color:#333;padding-top:1.7rem'>VS</div>", unsafe_allow_html=True)
    with c2:
        away = st.selectbox("✈️ Away Team", [t for t in teams if t != home], key="pa")

    predict_clicked = st.button("⚡ Predict Result", use_container_width=True)

    if predict_clicked:
        proba = predict_match(home, away)
        if proba is not None:
            hw, dr, aw = proba
            best = max(hw, dr, aw)

            p1, p2, p3 = st.columns(3)
            for col, pct, label, color in [
                (p1, hw, f"🏠 {home} Win", GREEN if hw==best else WHITE),
                (p2, dr, "🤝 Draw", GOLD if dr==best else WHITE),
                (p3, aw, f"✈️ {away} Win", BLUE if aw==best else WHITE),
            ]:
                with col:
                    st.markdown(f'<div class="predbox"><div class="predpct" style="color:{color}">{pct*100:.0f}%</div><div class="predlbl">{label}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gradient bar
            fig, ax = plt.subplots(figsize=(9, 0.5))
            ax.barh(0, hw, color=GREEN, height=1)
            ax.barh(0, dr, left=hw, color=GOLD, height=1)
            ax.barh(0, aw, left=hw+dr, color=BLUE, height=1)
            ax.set_xlim(0,1); ax.axis("off")
            mpl_dark(fig); fig.patch.set_facecolor(DARK)
            st.pyplot(fig, use_container_width=True)

            # Form + stats comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Team Comparison</div>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            for col, team in [(fc1, home), (fc2, away)]:
                with col:
                    dots = get_team_form_dots(team)
                    f = latest_form[latest_form["team"] == team]
                    ppg = f"{f.iloc[0]['form_points']:.2f}" if not f.empty else "—"
                    scored = f"{f.iloc[0]['form_scored']:.2f}" if not f.empty else "—"
                    conceded = f"{f.iloc[0]['form_conceded']:.2f}" if not f.empty else "—"
                    win_streak = win_streaks.get(team, 0)
                    title_count = titles.get(team, 0)
                    st.markdown(f"""
                    <div class="card">
                      <div style="font-weight:800;font-size:1rem;margin-bottom:.7rem">{team}</div>
                      <div style="margin-bottom:.5rem">{dots}</div>
                      <div style="display:flex;gap:1rem;margin-top:.8rem;flex-wrap:wrap">
                        <div><div style="color:{MUTED};font-size:.68rem;text-transform:uppercase">Pts/Game</div><div style="font-weight:700">{ppg}</div></div>
                        <div><div style="color:{MUTED};font-size:.68rem;text-transform:uppercase">Avg Scored</div><div style="font-weight:700">{scored}</div></div>
                        <div><div style="color:{MUTED};font-size:.68rem;text-transform:uppercase">Avg Conceded</div><div style="font-weight:700">{conceded}</div></div>
                        <div><div style="color:{MUTED};font-size:.68rem;text-transform:uppercase">Best Streak</div><div style="font-weight:700">{win_streak}W</div></div>
                        <div><div style="color:{MUTED};font-size:.68rem;text-transform:uppercase">Titles</div><div style="font-weight:700;color:{GOLD}">{title_count} 🏆</div></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # H2H quick preview
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Quick Head-to-Head</div>', unsafe_allow_html=True)
            h2h = results[((results["home_team"]==home)&(results["away_team"]==away))|
                          ((results["home_team"]==away)&(results["away_team"]==home))]
            if not h2h.empty:
                def team_r(row, t):
                    if row["home_team"]==t: return "W" if row["result"]=="H" else ("D" if row["result"]=="D" else "L")
                    return "W" if row["result"]=="A" else ("D" if row["result"]=="D" else "L")
                hw_wins = (h2h.apply(lambda r: team_r(r,home),axis=1)=="W").sum()
                aw_wins = (h2h.apply(lambda r: team_r(r,away),axis=1)=="W").sum()
                draws_h = (h2h.apply(lambda r: team_r(r,home),axis=1)=="D").sum()
                hc1,hc2,hc3,hc4 = st.columns(4)
                hc1.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{GREEN}">{hw_wins}</div><div class="h2hlbl">{home}</div></div>', unsafe_allow_html=True)
                hc2.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{GOLD}">{draws_h}</div><div class="h2hlbl">Draws</div></div>', unsafe_allow_html=True)
                hc3.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{BLUE}">{aw_wins}</div><div class="h2hlbl">{away}</div></div>', unsafe_allow_html=True)
                hc4.markdown(f'<div class="h2hbox"><div class="h2hnum">{len(h2h)}</div><div class="h2hlbl">Total</div></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 2 — NEXT SEASON
# ════════════════════════════════════════════════════
with tab2:
    champ_pts = int(sim_standings.iloc[0])
    st.markdown(f"""
    <div class="champ">
      <div class="ctrophy">🏆</div>
      <div class="cname">{champion}</div>
      <div class="csub">Predicted champion · {champ_pts} points · Season after {latest_season}</div>
    </div>
    """, unsafe_allow_html=True)

    col_t, col_c = st.columns([1,1])
    with col_t:
        st.markdown('<div class="section-label">Predicted Final Table</div>', unsafe_allow_html=True)
        n = len(sim_standings)
        for i, (team, pts) in enumerate(sim_standings.items(), 1):
            border = f"border-left:3px solid {GOLD}" if i==1 else \
                     f"border-left:3px solid {GREEN}" if i<=4 else \
                     f"border-left:3px solid {BLUE}" if i<=6 else \
                     (f"border-left:3px solid {RED}" if i>=n-2 else "")
            gclass = "g" if i<=3 else ""
            badge_bg = "#2a1f00" if i==1 else "#0f2e0f"
            badge_col = GOLD if i==1 else GREEN
            st.markdown(f"""
            <div class="srow" style="{border}">
              <div class="spos {gclass}">{i}</div>
              <div class="sname">{team}</div>
              <div class="spts" style="background:{badge_bg};color:{badge_col}">{int(pts)}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:.68rem;color:{MUTED};margin-top:.5rem">🥇 Champion &nbsp;|&nbsp; 🟢 Top 4 (CL zone) &nbsp;|&nbsp; 🔵 Top 6 &nbsp;|&nbsp; 🔴 Relegation</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-label">Points Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 7))
        top = sim_standings.head(10)
        colors = [GOLD if i==0 else GREEN if i<4 else BLUE if i<6 else "#333" for i in range(len(top))]
        bars = ax.barh(range(len(top)), top.values, color=colors, height=0.65, edgecolor="none")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("Predicted Points")
        for bar, val in zip(bars, top.values):
            ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2, str(int(val)), va="center", color=WHITE, fontsize=8, fontweight="bold")
        ax.set_xlim(0, max(top.values)*1.14)
        mpl_dark(fig); st.pyplot(fig, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Title Race Probability</div>', unsafe_allow_html=True)
        total_pts = sim_standings.sum()
        top5 = sim_standings.head(5)
        fig2, ax2 = plt.subplots(figsize=(5,2.5))
        probs = (top5.values / total_pts * 100)
        ax2.pie(probs, labels=top5.index, autopct="%1.0f%%", startangle=90,
                colors=[GOLD, GREEN, BLUE, "#e67e22", "#8e44ad"],
                textprops={"color":WHITE,"fontsize":8}, wedgeprops={"edgecolor":DARK,"linewidth":2})
        ax2.set_title("Share of predicted points (top 5)", color=WHITE, fontsize=9)
        mpl_dark(fig2); st.pyplot(fig2, use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 3 — KNOCKOUT BRACKET
# ════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Simulated Knockout Cup — Top 8 Teams</div>', unsafe_allow_html=True)
    st.caption("Top 8 from predicted standings enter a single-elimination bracket. Each match predicted by ML model.")

    top8 = list(sim_standings.head(8).index)

    def run_match(t1, t2):
        p = predict_match(t1, t2)
        if p is None: return t1, 0
        return (t1, p[0]) if p[0] >= p[2] else (t2, p[2])

    qf_pairs = [(top8[0],top8[7]),(top8[1],top8[6]),(top8[2],top8[5]),(top8[3],top8[4])]

    st.markdown('<div class="brlabel">Quarter Finals</div>', unsafe_allow_html=True)
    qf_cols = st.columns(4)
    sf_teams = []
    for i, (t1,t2) in enumerate(qf_pairs):
        winner, prob = run_match(t1, t2)
        loser = t2 if winner==t1 else t1
        sf_teams.append(winner)
        with qf_cols[i]:
            st.markdown(f'<div class="bm"><div class="bw">✓ {winner} <span style="font-size:.72rem;color:{MUTED}">{prob*100:.0f}%</span></div><div style="height:3px"></div><div class="bl">✗ {loser}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="brlabel">Semi Finals</div>', unsafe_allow_html=True)
    sf_pairs = [(sf_teams[0],sf_teams[3]),(sf_teams[1],sf_teams[2])]
    sf_cols = st.columns([1,2,2,1])
    final_teams = []
    for i,(t1,t2) in enumerate(sf_pairs):
        winner, prob = run_match(t1, t2)
        loser = t2 if winner==t1 else t1
        final_teams.append(winner)
        with sf_cols[i*2+(0 if i==0 else 1)]:
            st.markdown(f'<div class="bm"><div class="bw">✓ {winner} <span style="font-size:.72rem;color:{MUTED}">{prob*100:.0f}%</span></div><div style="height:3px"></div><div class="bl">✗ {loser}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="brlabel">🏆 Grand Final</div>', unsafe_allow_html=True)
    fw, fprob = run_match(final_teams[0], final_teams[1])
    fl = final_teams[1] if fw==final_teams[0] else final_teams[0]
    _, fc, _ = st.columns([2,2,2])
    with fc:
        st.markdown(f'<div class="bm" style="border-color:{GOLD};text-align:center"><div style="font-size:1.6rem">🏆</div><div class="bw" style="font-size:1.1rem;color:{GOLD}">{fw}</div><div style="color:{MUTED};font-size:.74rem;margin-top:.3rem">defeats {fl} · {fprob*100:.0f}% win probability</div></div>', unsafe_allow_html=True)

    # Third place
    third_loser_1 = sf_teams[3] if sf_teams[0]==final_teams[0] else sf_teams[0]
    third_loser_2 = sf_teams[2] if sf_teams[1]==final_teams[1] else sf_teams[1]
    tp_winner, tp_prob = run_match(third_loser_1, third_loser_2)
    st.markdown(f'<div style="text-align:center;margin-top:.5rem;color:{MUTED};font-size:.8rem">🥉 Third place: <strong style="color:{WHITE}">{tp_winner}</strong></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 4 — HEAD TO HEAD
# ════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">All-Time Head to Head</div>', unsafe_allow_html=True)
    all_teams_ever = sorted(pd.concat([results["home_team"], results["away_team"]]).unique())
    hc1, hc2 = st.columns(2)
    with hc1: h2h_a = st.selectbox("Team A", all_teams_ever, index=all_teams_ever.index("Al Hilal") if "Al Hilal" in all_teams_ever else 0)
    with hc2: h2h_b = st.selectbox("Team B", [t for t in all_teams_ever if t!=h2h_a], index=0)

    h2h_df = results[((results["home_team"]==h2h_a)&(results["away_team"]==h2h_b))|
                     ((results["home_team"]==h2h_b)&(results["away_team"]==h2h_a))].copy()

    if h2h_df.empty:
        st.warning("No historical matches found between these teams.")
    else:
        def tr(row, team):
            if row["home_team"]==team: return "W" if row["result"]=="H" else("D" if row["result"]=="D" else "L")
            return "W" if row["result"]=="A" else("D" if row["result"]=="D" else "L")

        a_wins = (h2h_df.apply(lambda r: tr(r,h2h_a),axis=1)=="W").sum()
        b_wins = (h2h_df.apply(lambda r: tr(r,h2h_b),axis=1)=="W").sum()
        draws  = (h2h_df.apply(lambda r: tr(r,h2h_a),axis=1)=="D").sum()
        total  = len(h2h_df)
        g_a = int(h2h_df.apply(lambda r: r["home_goals"] if r["home_team"]==h2h_a else r["away_goals"],axis=1).sum())
        g_b = int(h2h_df.apply(lambda r: r["home_goals"] if r["home_team"]==h2h_b else r["away_goals"],axis=1).sum())

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{GREEN}">{a_wins}</div><div class="h2hlbl">{h2h_a}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{GOLD}">{draws}</div><div class="h2hlbl">Draws</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="h2hbox"><div class="h2hnum" style="color:{BLUE}">{b_wins}</div><div class="h2hlbl">{h2h_b}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="h2hbox"><div class="h2hnum">{g_a}–{g_b}</div><div class="h2hlbl">Goals</div></div>', unsafe_allow_html=True)
        m5.markdown(f'<div class="h2hbox"><div class="h2hnum">{total}</div><div class="h2hlbl">Matches</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pc1,pc2 = st.columns(2)
        with pc1:
            fig,ax = plt.subplots(figsize=(4.5,3.5))
            ax.pie([a_wins,draws,b_wins],labels=[h2h_a,"Draw",h2h_b],
                   colors=[GREEN,GOLD,BLUE],autopct="%1.0f%%",startangle=90,
                   textprops={"color":WHITE,"fontsize":9},wedgeprops={"edgecolor":DARK,"linewidth":2})
            ax.set_title("Win share",color=WHITE,fontsize=10)
            mpl_dark(fig); st.pyplot(fig,use_container_width=True)

        with pc2:
            seasons_played = sorted(h2h_df["season"].unique())
            goals_by_s = [h2h_df[h2h_df["season"]==s]["total_goals"].sum() for s in seasons_played]
            fig2,ax2 = plt.subplots(figsize=(4.5,3.5))
            ax2.bar(range(len(seasons_played)), goals_by_s, color=GREEN, edgecolor="none", alpha=0.8)
            ax2.set_xticks(range(len(seasons_played)))
            ax2.set_xticklabels([s[:4] for s in seasons_played],rotation=45,fontsize=7)
            ax2.set_ylabel("Goals")
            ax2.set_title("Goals per season",color=WHITE,fontsize=10)
            mpl_dark(fig2); st.pyplot(fig2,use_container_width=True)

        # AI prediction for next match
        st.markdown('<div class="section-label" style="margin-top:1rem">Next Meeting Prediction</div>', unsafe_allow_html=True)
        pr_c1, pr_c2 = st.columns(2)
        for col, home_t, away_t in [(pr_c1,h2h_a,h2h_b),(pr_c2,h2h_b,h2h_a)]:
            p = predict_match(home_t, away_t)
            if p is not None:
                with col:
                    st.markdown(f"**{home_t} (H) vs {away_t} (A)**")
                    fig3,ax3 = plt.subplots(figsize=(3.5,0.4))
                    ax3.barh(0,p[0],color=GREEN,height=1)
                    ax3.barh(0,p[1],left=p[0],color=GOLD,height=1)
                    ax3.barh(0,p[2],left=p[0]+p[1],color=BLUE,height=1)
                    ax3.set_xlim(0,1); ax3.axis("off")
                    mpl_dark(fig3); fig3.patch.set_facecolor(DARK)
                    st.pyplot(fig3,use_container_width=True)
                    st.caption(f"🟢 {p[0]*100:.0f}%  🟡 {p[1]*100:.0f}%  🔵 {p[2]*100:.0f}%")

        st.markdown('<div class="section-label" style="margin-top:1rem">Match History</div>', unsafe_allow_html=True)
        recent = h2h_df.sort_values("season",ascending=False)[["season","home_team","score","away_team","city"]].reset_index(drop=True)
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# TAB 5 — TEAM PROFILE
# ════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">Team Deep Dive</div>', unsafe_allow_html=True)
    all_t = sorted(pd.concat([results["home_team"],results["away_team"]]).unique())
    sel_team = st.selectbox("Select Team", all_t, index=all_t.index("Al Hilal") if "Al Hilal" in all_t else 0)

    tm_all = results[(results["home_team"]==sel_team)|(results["away_team"]==sel_team)]
    def get_r(row,t):
        if row["home_team"]==t: return "W" if row["result"]=="H" else("D" if row["result"]=="D" else "L")
        return "W" if row["result"]=="A" else("D" if row["result"]=="D" else "L")
    tm_results = tm_all.apply(lambda r: get_r(r,sel_team),axis=1)
    total_played = len(tm_all)
    total_w = (tm_results=="W").sum()
    total_d = (tm_results=="D").sum()
    total_l = (tm_results=="L").sum()
    goals_for = int(tm_all.apply(lambda r: r["home_goals"] if r["home_team"]==sel_team else r["away_goals"],axis=1).sum())
    goals_ag  = int(tm_all.apply(lambda r: r["away_goals"] if r["home_team"]==sel_team else r["home_goals"],axis=1).sum())
    t_titles = titles.get(sel_team,0)
    t_streak = win_streaks.get(sel_team,0)
    seasons_played = sorted(tm_all["season"].unique())

    p1,p2,p3,p4,p5,p6 = st.columns(6)
    p1.metric("Matches", total_played)
    p2.metric("Wins", total_w)
    p3.metric("Draws", total_d)
    p4.metric("Losses", total_l)
    p5.metric("Goals For/Against", f"{goals_for}/{goals_ag}")
    p6.metric("Titles 🏆", t_titles)

    st.markdown(f'<div style="margin:.5rem 0"><span class="badge badge-gold">Best streak: {t_streak}W</span><span class="badge">Win rate: {total_w/total_played*100:.0f}%</span></div>', unsafe_allow_html=True)

    # Points over seasons
    st.markdown('<div class="section-label" style="margin-top:1rem">Points per Season</div>', unsafe_allow_html=True)
    pts_by_season = [team_season_pts(sel_team, s) for s in seasons_played]
    fig,ax = plt.subplots(figsize=(10,3))
    season_labels = [s[:4] for s in seasons_played]
    ax.fill_between(range(len(seasons_played)), pts_by_season, alpha=.25, color=GREEN)
    ax.plot(range(len(seasons_played)), pts_by_season, color=GREEN, linewidth=2.5, marker="o", markersize=5)
    for i,s in enumerate(seasons_played):
        if title_map.get(s)==sel_team:
            ax.annotate("🏆", (i, pts_by_season[i]), textcoords="offset points", xytext=(0,8), fontsize=12, ha="center")
    ax.set_xticks(range(len(seasons_played)))
    ax.set_xticklabels(season_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Points"); ax.set_title(f"{sel_team} — Points per Season", color=WHITE)
    mpl_dark(fig); st.pyplot(fig, use_container_width=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<div class="section-label">Season-by-Season W/D/L</div>', unsafe_allow_html=True)
        s_wins  = [len(results[(results["season"]==s)&((results["home_team"]==sel_team)|(results["away_team"]==sel_team))].apply(lambda r: get_r(r,sel_team),axis=1)[lambda x: x=="W"]) for s in seasons_played]
        s_draws = [len(results[(results["season"]==s)&((results["home_team"]==sel_team)|(results["away_team"]==sel_team))].apply(lambda r: get_r(r,sel_team),axis=1)[lambda x: x=="D"]) for s in seasons_played]
        s_loss  = [len(results[(results["season"]==s)&((results["home_team"]==sel_team)|(results["away_team"]==sel_team))].apply(lambda r: get_r(r,sel_team),axis=1)[lambda x: x=="L"]) for s in seasons_played]
        xs = range(len(seasons_played))
        fig2,ax2 = plt.subplots(figsize=(5,3))
        ax2.bar(xs, s_wins, color=GREEN, label="W", edgecolor="none")
        ax2.bar(xs, s_draws, bottom=s_wins, color=GOLD, label="D", edgecolor="none")
        ax2.bar(xs, s_loss, bottom=[w+d for w,d in zip(s_wins,s_draws)], color=RED, label="L", edgecolor="none")
        ax2.set_xticks(xs); ax2.set_xticklabels(season_labels,rotation=45,fontsize=7)
        ax2.legend(fontsize=8,facecolor=CARD,edgecolor="#333",labelcolor=WHITE)
        mpl_dark(fig2); st.pyplot(fig2,use_container_width=True)

    with rc2:
        st.markdown('<div class="section-label">Goals Scored vs Conceded by Season</div>', unsafe_allow_html=True)
        gf_by_s = [int(results[(results["season"]==s)&((results["home_team"]==sel_team)|(results["away_team"]==sel_team))].apply(lambda r: r["home_goals"] if r["home_team"]==sel_team else r["away_goals"],axis=1).sum()) for s in seasons_played]
        ga_by_s = [int(results[(results["season"]==s)&((results["home_team"]==sel_team)|(results["away_team"]==sel_team))].apply(lambda r: r["away_goals"] if r["home_team"]==sel_team else r["home_goals"],axis=1).sum()) for s in seasons_played]
        fig3,ax3 = plt.subplots(figsize=(5,3))
        ax3.plot(xs,gf_by_s,color=GREEN,linewidth=2,marker="o",markersize=4,label="Scored")
        ax3.plot(xs,ga_by_s,color=RED,linewidth=2,marker="o",markersize=4,label="Conceded")
        ax3.fill_between(xs,gf_by_s,ga_by_s,where=[g>c for g,c in zip(gf_by_s,ga_by_s)],alpha=.15,color=GREEN)
        ax3.fill_between(xs,gf_by_s,ga_by_s,where=[g<c for g,c in zip(gf_by_s,ga_by_s)],alpha=.15,color=RED)
        ax3.set_xticks(xs); ax3.set_xticklabels(season_labels,rotation=45,fontsize=7)
        ax3.legend(fontsize=8,facecolor=CARD,edgecolor="#333",labelcolor=WHITE)
        mpl_dark(fig3); st.pyplot(fig3,use_container_width=True)

    # Biggest wins & losses
    st.markdown('<div class="section-label" style="margin-top:1rem">Biggest Wins & Worst Defeats</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("**Biggest Wins**")
        bw = tm_all.copy()
        bw["gf"] = bw.apply(lambda r: r["home_goals"] if r["home_team"]==sel_team else r["away_goals"],axis=1)
        bw["ga"] = bw.apply(lambda r: r["away_goals"] if r["home_team"]==sel_team else r["home_goals"],axis=1)
        bw["gd"] = bw["gf"]-bw["ga"]
        top_wins = bw[bw["gd"]>0].nlargest(5,"gd")[["season","home_team","score","away_team"]].reset_index(drop=True)
        st.dataframe(top_wins, use_container_width=True, hide_index=True)
    with bc2:
        st.markdown("**Worst Defeats**")
        top_loss = bw[bw["gd"]<0].nsmallest(5,"gd")[["season","home_team","score","away_team"]].reset_index(drop=True)
        st.dataframe(top_loss, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# TAB 6 — ALL-TIME RECORDS
# ════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-label">All-Time Records & Milestones</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("**🏆 Title Winners**")
        for team, cnt in titles.most_common():
            bar = "█" * cnt
            st.markdown(f'<div class="reccard"><div class="rtitle">{team}</div><div class="rval" style="color:{GOLD if cnt>=4 else GREEN}">{bar} {cnt}</div></div>', unsafe_allow_html=True)

    with rc2:
        st.markdown("**⚡ Longest Win Streaks (All-Time)**")
        top_streaks = sorted(win_streaks.items(), key=lambda x: -x[1])[:8]
        for team, streak in top_streaks:
            st.markdown(f'<div class="reccard"><div class="rtitle">{team}</div><div class="rval">{streak} consecutive wins</div></div>', unsafe_allow_html=True)

    with rc3:
        st.markdown("**🎯 Most Goals Scored (All-Time)**")
        for i, (team, goals) in enumerate(total_goals_all.head(8).items()):
            st.markdown(f'<div class="reccard"><div class="rtitle">{team}</div><div class="rval" style="color:{GOLD if i==0 else WHITE}">{int(goals)} goals</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔥 Most Thrilling Matches (Highest Scoring)</div>', unsafe_allow_html=True)
    top_games = biggest_wins.copy()
    top_games = results.copy()
    top_games["gdiff"] = abs(top_games["home_goals"]-top_games["away_goals"])
    top10_games = top_games.nlargest(10,"total_goals")[["season","home_team","score","away_team","total_goals","city"]].reset_index(drop=True)
    top10_games.index += 1
    st.dataframe(top10_games, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">📅 Season-by-Season Champions</div>', unsafe_allow_html=True)
    champ_rows = []
    for season in sorted(results["season"].unique()):
        s = results[results["season"]==season]
        pts = {}
        for _, r in s.iterrows():
            h,a = r["home_team"],r["away_team"]
            pts.setdefault(h,0);pts.setdefault(a,0)
            if r["result"]=="H": pts[h]+=3
            elif r["result"]=="A": pts[a]+=3
            else: pts[h]+=1;pts[a]+=1
        winner = max(pts, key=pts.get)
        top2 = sorted(pts.items(),key=lambda x:-x[1])[:2]
        margin = top2[0][1]-top2[1][1] if len(top2)>1 else 0
        champ_rows.append({"Season":season,"Champion":winner,"Points":pts[winner],"Winning Margin":margin})
    champ_df = pd.DataFrame(champ_rows)
    st.dataframe(champ_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">🏟️ Peak Attendance Matches</div>', unsafe_allow_html=True)
    att_df = results.dropna(subset=["attendance"]).nlargest(10,"attendance")[["season","home_team","score","away_team","venue","attendance"]].reset_index(drop=True)
    att_df["attendance"] = att_df["attendance"].astype(int)
    st.dataframe(att_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# TAB 7 — ANALYTICS
# ════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-label">League-Wide Analytics</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Goals per match — 15 seasons**")
        avg_g = results.groupby("season")["total_goals"].mean()
        fig,ax = plt.subplots(figsize=(5.5,3))
        ax.fill_between(range(len(avg_g)),avg_g.values,alpha=.2,color=GREEN)
        ax.plot(range(len(avg_g)),avg_g.values,color=GREEN,linewidth=2.5,marker="o",markersize=5)
        ax.axhline(avg_g.mean(),color=GOLD,linestyle="--",linewidth=1.2,alpha=.7)
        ax.set_xticks(range(len(avg_g))); ax.set_xticklabels([s[:4] for s in avg_g.index],rotation=45,fontsize=7)
        ax.set_ylabel("Avg Goals/Match")
        ax.annotate(f"Avg {avg_g.mean():.2f}", xy=(len(avg_g)-1,avg_g.mean()), color=GOLD, fontsize=8, xytext=(0,6), textcoords="offset points")
        mpl_dark(fig); st.pyplot(fig,use_container_width=True)

    with a2:
        st.markdown("**Home / Away / Draw rates over time**")
        h_r = results.groupby("season").apply(lambda x:(x["result"]=="H").mean())
        a_r = results.groupby("season").apply(lambda x:(x["result"]=="A").mean())
        d_r = results.groupby("season").apply(lambda x:(x["result"]=="D").mean())
        xs=range(len(h_r))
        fig2,ax2 = plt.subplots(figsize=(5.5,3))
        ax2.plot(xs,h_r.values,color=GREEN,linewidth=2,label="Home Win")
        ax2.plot(xs,a_r.values,color=BLUE,linewidth=2,label="Away Win")
        ax2.plot(xs,d_r.values,color=GOLD,linewidth=2,label="Draw")
        ax2.set_xticks(xs); ax2.set_xticklabels([s[:4] for s in h_r.index],rotation=45,fontsize=7)
        ax2.set_ylabel("Rate"); ax2.legend(fontsize=8,facecolor=CARD,edgecolor="#333",labelcolor=WHITE)
        mpl_dark(fig2); st.pyplot(fig2,use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Goals by gameweek (avg across all seasons)**")
        gw_g = results.groupby("gameweek")["total_goals"].mean()
        fig3,ax3 = plt.subplots(figsize=(5.5,3))
        colors_gw = [GOLD if v==gw_g.max() else GREEN for v in gw_g.values]
        ax3.bar(gw_g.index, gw_g.values, color=colors_gw, edgecolor="none", width=0.8)
        ax3.set_xlabel("Gameweek"); ax3.set_ylabel("Avg Goals")
        ax3.axhline(gw_g.mean(),color=WHITE,linestyle="--",linewidth=1,alpha=.4)
        mpl_dark(fig3); st.pyplot(fig3,use_container_width=True)

    with b2:
        st.markdown("**Attendance trend (avg per season)**")
        att_by_s = results.dropna(subset=["attendance"]).groupby("season")["attendance"].mean()
        fig4,ax4 = plt.subplots(figsize=(5.5,3))
        ax4.bar(range(len(att_by_s)),att_by_s.values,color=BLUE,edgecolor="none",alpha=.8)
        ax4.set_xticks(range(len(att_by_s)))
        ax4.set_xticklabels([s[:4] for s in att_by_s.index],rotation=45,fontsize=7)
        ax4.set_ylabel("Avg Attendance")
        covid_idx = list(att_by_s.index).index("2020/21") if "2020/21" in att_by_s.index else None
        if covid_idx: ax4.annotate("COVID",xy=(covid_idx,att_by_s.iloc[covid_idx]),xytext=(0,8),textcoords="offset points",color=RED,fontsize=8,ha="center")
        mpl_dark(fig4); st.pyplot(fig4,use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">Season Drill-Down</div>', unsafe_allow_html=True)
    sel_s = st.select_slider("Choose Season", options=sorted(results["season"].unique()), value=latest_season)
    sd = results[results["season"]==sel_s]
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Matches", len(sd))
    d2.metric("Goals", int(sd["total_goals"].sum()))
    d3.metric("Avg Goals", f"{sd['total_goals'].mean():.2f}")
    d4.metric("Home Win %", f"{(sd['result']=='H').mean()*100:.0f}%")
    d5.metric("Champion", title_map.get(sel_s,"TBD"))

    tg = (sd.groupby("home_team")["home_goals"].sum().add(sd.groupby("away_team")["away_goals"].sum(),fill_value=0)).sort_values(ascending=False)
    fig5,ax5 = plt.subplots(figsize=(10,2.8))
    clrs=[GOLD if i==0 else GREEN for i in range(len(tg))]
    ax5.bar(range(len(tg)),tg.values,color=clrs,edgecolor="none")
    ax5.set_xticks(range(len(tg))); ax5.set_xticklabels(tg.index,rotation=45,ha="right",fontsize=8)
    ax5.set_title(f"Goals per team — {sel_s}",color=WHITE,fontsize=10)
    mpl_dark(fig5); st.pyplot(fig5,use_container_width=True)
