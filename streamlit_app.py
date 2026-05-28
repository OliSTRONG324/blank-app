import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
import time

try:
    from pybaseball import statcast_batter, statcast_pitcher
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HR Probability Scout",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ HR Probability Scout")
st.caption("Enter a batter and pitcher to get a data-driven home run probability score.")

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
SEASON_START = f"{date.today().year}-03-01"
SEASON_END   = f"{date.today().year}-11-01"

MLB_AVG_EV        = 88.6
MLB_AVG_BARREL    = 7.6
MLB_AVG_HARD_HIT  = 37.0
MLB_AVG_LA        = 12.5

PARK_FACTORS = {
    "Coors Field (COL)": 1.20,
    "Great American Ball Park (CIN)": 1.15,
    "Globe Life Field (TEX)": 1.10,
    "Yankee Stadium (NYY)": 1.08,
    "Fenway Park (BOS)": 1.06,
    "Truist Park (ATL)": 1.04,
    "Wrigley Field (CHC)": 1.03,
    "Neutral / Average Park": 1.00,
    "Oracle Park (SF)": 0.92,
    "Petco Park (SD)": 0.90,
    "Dodger Stadium (LAD)": 0.94,
    "Rogers Centre (TOR) — Dome": 1.00,
    "Tropicana Field (TB) — Dome": 0.97,
    "Minute Maid Park (HOU) — Dome": 1.02,
    "American Family Field (MIL) — Dome": 1.05,
    "T-Mobile Park (SEA)": 0.93,
}

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_player(name: str):
    """Search MLB Stats API for a player by name."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={name}&sportIds=1"
        res = requests.get(url, timeout=10).json()
        people = res.get("people", [])
        if not people:
            return None
        # Prefer active players
        active = [p for p in people if p.get("active", False)]
        return active[0] if active else people[0]
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_batter_stats(player_id: int):
    """Pull batter Statcast data and compute key metrics."""
    if not PYBASEBALL_AVAILABLE:
        return None
    try:
        data = statcast_batter(SEASON_START, SEASON_END, player_id)
        if data is None or len(data) == 0:
            return None

        bbe = data.dropna(subset=["launch_speed"])

        ev          = bbe["launch_speed"].mean()
        ev50        = bbe["launch_speed"].nlargest(int(len(bbe) * 0.5)).mean()
        hard_hit    = (bbe["launch_speed"] >= 95).mean() * 100
        barrel_pct  = bbe["barrel"].mean() * 100 if "barrel" in bbe.columns else 0

        la_data     = bbe.dropna(subset=["launch_angle"])
        la_avg      = la_data["launch_angle"].mean()
        sweet_spot  = ((la_data["launch_angle"] >= 8) & (la_data["launch_angle"] <= 32)).mean() * 100

        pull_pct    = 0.0
        if "hc_x" in bbe.columns and "stand" in data.columns:
            stand = data["stand"].iloc[0] if len(data) else "R"
            if stand == "R":
                pull_pct = (bbe["hc_x"] < 125).mean() * 100
            else:
                pull_pct = (bbe["hc_x"] > 175).mean() * 100

        hr_rate = (data["events"] == "home_run").sum() / max(len(data), 1) * 100

        return {
            "ev": round(ev, 1),
            "ev50": round(ev50, 1),
            "hard_hit": round(hard_hit, 1),
            "barrel_pct": round(barrel_pct, 2),
            "la_avg": round(la_avg, 1),
            "sweet_spot_pct": round(sweet_spot, 1),
            "pull_pct": round(pull_pct, 1),
            "hr_rate": round(hr_rate, 2),
            "pa": len(data),
        }
    except Exception as e:
        st.warning(f"Batter data error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_pitcher_stats(player_id: int):
    """Pull pitcher Statcast data and compute key metrics."""
    if not PYBASEBALL_AVAILABLE:
        return None
    try:
        data = statcast_pitcher(SEASON_START, SEASON_END, player_id)
        if data is None or len(data) == 0:
            return None

        bbe = data.dropna(subset=["launch_speed"])

        barrel_allowed  = bbe["barrel"].mean() * 100 if "barrel" in bbe.columns else 0
        hard_hit_allowed = (bbe["launch_speed"] >= 95).mean() * 100
        hr_rate         = (data["events"] == "home_run").sum() / max(len(data), 1) * 100

        fb_velo = None
        if "release_speed" in data.columns and "pitch_type" in data.columns:
            fastballs = data[data["pitch_type"].isin(["FF", "SI", "FC"])]
            if len(fastballs):
                fb_velo = round(fastballs["release_speed"].mean(), 1)

        fly_balls = data[data.get("bb_type", pd.Series()).isin(["fly_ball", "popup"])] if "bb_type" in data.columns else pd.DataFrame()
        hr_fb = 0.0
        if len(fly_balls) > 0:
            hr_fb = (fly_balls["events"] == "home_run").sum() / len(fly_balls) * 100

        return {
            "barrel_allowed": round(barrel_allowed, 2),
            "hard_hit_allowed": round(hard_hit_allowed, 1),
            "hr_rate": round(hr_rate, 2),
            "hr_fb": round(hr_fb, 1),
            "fb_velo": fb_velo,
            "bf": len(data),
        }
    except Exception as e:
        st.warning(f"Pitcher data error: {e}")
        return None

def score_matchup(bat, pit, park_factor, wind, platoon, count):
    """
    Score the matchup 0–100 using weighted factors.
    Returns total score and a breakdown dict.
    """
    breakdown = {}

    # ── Batter power (max ~42 pts) ──────────────────────────────
    ev_score = max(0, (bat["ev"] - MLB_AVG_EV) * 1.8)
    ev_score = min(ev_score, 12)
    breakdown["Exit velocity"] = round(ev_score, 1)

    barrel_score = max(0, (bat["barrel_pct"] - MLB_AVG_BARREL) * 1.2)
    barrel_score = min(barrel_score, 14)
    breakdown["Barrel %"] = round(barrel_score, 1)

    hard_score = max(0, (bat["hard_hit"] - MLB_AVG_HARD_HIT) * 0.3)
    hard_score = min(hard_score, 8)
    breakdown["Hard hit %"] = round(hard_score, 1)

    la = bat["la_avg"]
    if 20 <= la <= 32:
        la_score = 8.0
    elif 12 <= la < 20:
        la_score = 4.0
    elif 32 < la <= 38:
        la_score = 4.0
    else:
        la_score = 0.0
    breakdown["Launch angle"] = la_score

    ss_score = min(bat["sweet_spot_pct"] * 0.15, 6.0)
    breakdown["Sweet spot %"] = round(ss_score, 1)

    # ── Pitcher weakness (max ~30 pts) ─────────────────────────
    pit_barrel_score = min(pit["barrel_allowed"] * 1.0, 10.0)
    breakdown["Pitcher barrels allowed"] = round(pit_barrel_score, 1)

    hr_rate_score = min(pit["hr_rate"] * 8.0, 12.0)
    breakdown["Pitcher HR rate"] = round(hr_rate_score, 1)

    fb_score = 0.0
    if pit["fb_velo"] is not None:
        if pit["fb_velo"] < 91:
            fb_score = 6.0
        elif pit["fb_velo"] < 93:
            fb_score = 3.0
    breakdown["Pitcher velo"] = fb_score

    # ── Situation (max ~18 pts) ─────────────────────────────────
    park_score = min((park_factor - 1.0) * 30, 8.0)
    park_score = max(park_score, -4.0)
    breakdown["Park factor"] = round(park_score, 1)

    wind_scores = {"Blowing out (big boost)": 6, "Blowing out (light)": 3,
                   "Calm / Dome": 0, "Blowing in (light)": -2, "Blowing in (strong)": -4}
    wind_score = wind_scores.get(wind, 0)
    breakdown["Wind"] = wind_score

    platoon_score = 4.0 if platoon else 0.0
    breakdown["Platoon advantage"] = platoon_score

    count_scores = {"2-0 or 3-1 (hitter's count)": 4, "1-0 or 2-1": 2,
                    "Even (0-0, 1-1, 2-2)": 0, "0-1 or 0-2 (pitcher's count)": -2}
    count_score = count_scores.get(count, 0)
    breakdown["Count"] = count_score

    total = sum(breakdown.values())
    total = max(0, min(100, total))
    return round(total, 1), breakdown

# ─────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Batter")
    batter_name = st.text_input("Batter name", placeholder="e.g. Kyle Stowers")

with col2:
    st.subheader("Pitcher")
    pitcher_name = st.text_input("Pitcher name", placeholder="e.g. Kevin Gausman")

st.divider()
st.subheader("Game Conditions")

cond1, cond2, cond3, cond4 = st.columns(4)

with cond1:
    park = st.selectbox("Ballpark", list(PARK_FACTORS.keys()), index=7)
    park_factor = PARK_FACTORS[park]

with cond2:
    wind = st.selectbox("Wind", [
        "Calm / Dome",
        "Blowing out (light)",
        "Blowing out (big boost)",
        "Blowing in (light)",
        "Blowing in (strong)",
    ])

with cond3:
    platoon = st.checkbox("Platoon advantage?", value=True,
                          help="Lefty batter vs righty pitcher or vice versa")

with cond4:
    count = st.selectbox("At-bat count", [
        "Even (0-0, 1-1, 2-2)",
        "1-0 or 2-1",
        "2-0 or 3-1 (hitter's count)",
        "0-1 or 0-2 (pitcher's count)",
    ])

st.divider()

# ─────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────
if st.button("🔍 Analyze Matchup", type="primary", use_container_width=True):

    if not batter_name or not pitcher_name:
        st.error("Enter both a batter and pitcher name.")
        st.stop()

    if not PYBASEBALL_AVAILABLE:
        st.error("pybaseball is not installed. Run: pip install pybaseball")
        st.stop()

    with st.spinner("Looking up players..."):
        batter  = get_player(batter_name)
        pitcher = get_player(pitcher_name)

    if not batter:
        st.error(f"Could not find batter: '{batter_name}'. Try their full name.")
        st.stop()
    if not pitcher:
        st.error(f"Could not find pitcher: '{pitcher_name}'. Try their full name.")
        st.stop()

    with st.spinner(f"Pulling Statcast data for {batter['fullName']} and {pitcher['fullName']}..."):
        bat = get_batter_stats(batter["id"])
        pit = get_pitcher_stats(pitcher["id"])
        time.sleep(0.5)

    if not bat:
        st.error(f"No Statcast data found for {batter['fullName']} this season.")
        st.stop()
    if not pit:
        st.error(f"No Statcast data found for {pitcher['fullName']} this season.")
        st.stop()

    score, breakdown = score_matchup(bat, pit, park_factor, wind, platoon, count)

    # ── Verdict ────────────────────────────────────────────────
    if score >= 70:
        verdict, color = "🔥 HIGH HR RISK", "inverse"
    elif score >= 45:
        verdict, color = "⚡ MEDIUM HR RISK", "off"
    else:
        verdict, color = "🧊 LOW HR RISK", "normal"

    st.divider()
    st.subheader("Results")

    top1, top2, top3 = st.columns(3)
    top1.metric("HR Probability Score", f"{score} / 100")
    top2.metric("Verdict", verdict)
    top3.metric("Matchup", f"{batter['fullName']} vs {pitcher['fullName']}")

    st.divider()

    # ── Batter / Pitcher stats side by side ───────────────────
    b_col, p_col = st.columns(2)

    with b_col:
        st.markdown(f"### 🏏 {batter['fullName']}")
        bdf = pd.DataFrame({
            "Stat": ["Exit Velocity", "EV Top 50%", "Hard Hit %", "Barrel %",
                     "Avg Launch Angle", "Sweet Spot %", "Pull %", "HR Rate %", "Plate Appearances"],
            "Value": [f"{bat['ev']} mph", f"{bat['ev50']} mph", f"{bat['hard_hit']}%",
                      f"{bat['barrel_pct']}%", f"{bat['la_avg']}°", f"{bat['sweet_spot_pct']}%",
                      f"{bat['pull_pct']}%", f"{bat['hr_rate']}%", bat["pa"]],
            "MLB Avg": ["88.6 mph", "—", "37.0%", "7.6%", "12.5°", "—", "—", "—", "—"]
        })
        st.dataframe(bdf, hide_index=True, use_container_width=True)

    with p_col:
        st.markdown(f"### 🤾 {pitcher['fullName']}")
        prows = [
            ("Fastball Velocity", f"{pit['fb_velo']} mph" if pit['fb_velo'] else "N/A", "~93 mph avg"),
            ("Barrel % Allowed", f"{pit['barrel_allowed']}%", "7.6%"),
            ("Hard Hit % Allowed", f"{pit['hard_hit_allowed']}%", "37.0%"),
            ("HR Rate", f"{pit['hr_rate']}%", "—"),
            ("HR/FB %", f"{pit['hr_fb']}%", "~10-12%"),
            ("Batters Faced", str(pit["bf"]), "—"),
        ]
        pdf = pd.DataFrame(prows, columns=["Stat", "Value", "MLB Avg"])
        st.dataframe(pdf, hide_index=True, use_container_width=True)

    st.divider()

    # ── Score breakdown ────────────────────────────────────────
    st.markdown("### Score Breakdown")
    breakdown_df = pd.DataFrame([
        {"Factor": k, "Points": v, "Impact": "Positive" if v > 0 else ("Negative" if v < 0 else "Neutral")}
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1])
    ])

    def color_impact(val):
        if val == "Positive":
            return "color: green"
        elif val == "Negative":
            return "color: red"
        return ""

    st.dataframe(
        breakdown_df.style.applymap(color_impact, subset=["Impact"]),
        hide_index=True,
        use_container_width=True
    )

    # ── Tips ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💡 Key Observations")

    tips = []

    if bat["barrel_pct"] > 15:
        tips.append(f"✅ {batter['fullName']} has an elite barrel rate ({bat['barrel_pct']}% vs 7.6% avg) — high power potential.")
    if bat["ev"] < MLB_AVG_EV:
        tips.append(f"⚠️ {batter['fullName']}'s exit velocity ({bat['ev']} mph) is below MLB average — may struggle to carry the ball.")
    if bat["la_avg"] < 15:
        tips.append(f"⚠️ Low average launch angle ({bat['la_avg']}°) — hits the ball flat, limiting HR upside even on hard contact.")
    if pit["fb_velo"] and pit["fb_velo"] < 92:
        tips.append(f"✅ {pitcher['fullName']}'s fastball is down at {pit['fb_velo']} mph — easier to time and lift.")
    if pit["hr_rate"] > 3:
        tips.append(f"✅ {pitcher['fullName']} has a high HR rate ({pit['hr_rate']}%) — gives up home runs more than average.")
    if park_factor > 1.05:
        tips.append(f"✅ {park} is a hitter-friendly park (factor: {park_factor}) — ball carries well here.")
    if wind in ["Blowing out (big boost)", "Blowing out (light)"]:
        tips.append("✅ Wind blowing out — any well-hit fly ball gets extra carry today.")
    if platoon:
        tips.append("✅ Platoon advantage in play — batter has the favorable handedness matchup.")
    if count in ["2-0 or 3-1 (hitter's count)"]:
        tips.append("✅ Hitter's count — pitcher must throw a strike, batter can sit on the fastball.")

    if not tips:
        tips.append("No strong signals either way — this is a coin-flip matchup.")

    for tip in tips:
        st.write(tip)