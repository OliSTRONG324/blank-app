import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date
import concurrent.futures

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="Team HR Scout", page_icon="⚾", layout="wide")
st.title("⚾ Team vs. Pitcher HR Optimizer")
st.caption("Data source: Baseball Savant (MLB Statcast) & Official MLB API")

CURRENT_YEAR = date.today().year
SEASON_START = f"{CURRENT_YEAR}-03-01"
SEASON_END   = f"{CURRENT_YEAR}-11-01"

MLB_AVG = {
    "barrel_pct":    7.6,
    "hardhit_pct":  37.0,
    "fb_pct":       35.0,
    "iso":           0.170,
    "hr9":           1.30,
    "fb_allowed":   35.0,
    "hardhit_allowed": 37.0,
}

PARK_FACTORS = {
    "Neutral / Average Park":         1.00,
    "Coors Field (COL)":              1.20,
    "Great American Ball Park (CIN)": 1.15,
    "Globe Life Field (TEX)":         1.10,
    "Yankee Stadium (NYY)":           1.08,
    "Fenway Park (BOS)":              1.06,
    "Truist Park (ATL)":              1.04,
    "Wrigley Field (CHC)":            1.03,
    "Rogers Centre (TOR) — Dome":     1.00,
    "Tropicana Field (TB) — Dome":    0.97,
    "Minute Maid Park (HOU) — Dome":  1.02,
    "American Family Field (MIL)":    1.05,
    "Dodger Stadium (LAD)":           0.94,
    "Oracle Park (SF)":               0.92,
    "Petco Park (SD)":                0.90,
    "T-Mobile Park (SEA)":            0.93,
}

# ─────────────────────────────────────────
# MLB API Helpers
# ─────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_mlb_teams():
    """Fetches all active MLB teams dynamically."""
    try:
        res = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10).json()
        teams = {t["name"]: t["id"] for t in res.get("teams", [])}
        return dict(sorted(teams.items()))
    except Exception:
        return {"Chicago White Sox": 114, "New York Yankees": 147} # Fallback

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_hitters(team_id: int):
    """Gets the active roster and filters out pitchers."""
    try:
        res = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/Active", timeout=10).json()
        hitters = []
        for p in res.get("roster", []):
            if p["position"]["abbreviation"] not in ["P", "TWP"]:
                hitters.append({"id": p["person"]["id"], "name": p["person"]["fullName"]})
        return hitters
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def search_player(name: str):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={name}&sportIds=1"
        res = requests.get(url, timeout=10).json()
        people = res.get("people", [])
        active = [p for p in people if p.get("active", False)]
        return (active if active else people)
    except Exception:
        return []

def get_official_standard_stats(mlb_id: int, group: str):
    """Fetches accurate standard stats (HR, IP, ISO) from MLB API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}?hydrate=stats(group=[{group}],type=[season],season={CURRENT_YEAR})"
        res = requests.get(url, timeout=10).json()
        people = res.get("people", [])
        if not people: return {}
        stats_list = people[0].get("stats", [])
        if not stats_list: return {}
        splits = stats_list[0].get("splits", [])
        if not splits: return {}
        return splits[0].get("stat", {})
    except Exception:
        return {}

def is_barrel(ev, la):
    if pd.isna(ev) or pd.isna(la) or ev < 98: return False
    extra = ev - 98
    lo = max(26 - extra, 8)
    hi = min(30 + extra, 50)
    return lo <= la <= hi

# ─────────────────────────────────────────
# DATA FETCHING (BATTERS & PITCHER)
# ─────────────────────────────────────────
def get_batter_stats(mlb_id: int):
    try:
        api_stats = get_official_standard_stats(mlb_id, "hitting")
        iso = float(api_stats.get("slg", 0.0)) - float(api_stats.get("avg", 0.0))
        hr_count = int(api_stats.get("homeRuns", 0))

        from pybaseball import statcast_batter
        raw = statcast_batter(SEASON_START, SEASON_END, mlb_id)
        if raw is None or len(raw) == 0: return None

        if "game_date" in raw.columns:
            raw["game_date"] = pd.to_datetime(raw["game_date"])
            raw = raw[raw["game_date"].dt.year == CURRENT_YEAR].copy()

        bbe = raw[raw['description'] == 'hit_into_play'].dropna(subset=["launch_speed"]).copy()
        n_bbe = len(bbe)
        if n_bbe == 0: return None

        if "launch_angle" in bbe.columns:
            barrels = bbe.apply(lambda r: is_barrel(r["launch_speed"], r.get("launch_angle", np.nan)), axis=1).sum()
        else:
            barrels = 0
            
        barrel_pct = round((barrels / n_bbe) * 100, 1)
        hardhit_pct = round((bbe["launch_speed"] >= 95).sum() / n_bbe * 100, 1)
        fb_pct = round((bbe["bb_type"] == "fly_ball").sum() / n_bbe * 100, 1) if "bb_type" in bbe.columns else 0.0

        return {
            "barrel_pct": barrel_pct,
            "hardhit_pct": hardhit_pct,
            "fb_pct": fb_pct,
            "iso": round(iso, 3),
            "hr": hr_count,
            "n_bbe": n_bbe,
        }
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_pitcher_stats(mlb_id: int):
    try:
        api_stats = get_official_standard_stats(mlb_id, "pitching")
        hr9 = float(api_stats.get("homeRunsPer9", 0.0))
        ip = float(api_stats.get("inningsPitched", 0.0))
        hr_allowed = int(api_stats.get("homeRuns", 0))

        from pybaseball import statcast_pitcher
        raw = statcast_pitcher(SEASON_START, SEASON_END, mlb_id)
        if raw is None or len(raw) == 0: return None

        if "game_date" in raw.columns:
            raw["game_date"] = pd.to_datetime(raw["game_date"])
            raw = raw[raw["game_date"].dt.year == CURRENT_YEAR].copy()

        bbe = raw[raw['description'] == 'hit_into_play'].dropna(subset=["launch_speed"]).copy()
        n_bbe = len(bbe)

        hardhit_pct = round((bbe["launch_speed"] >= 95).sum() / n_bbe * 100, 1) if n_bbe > 0 else 0.0
        fb_pct = round((bbe["bb_type"] == "fly_ball").sum() / n_bbe * 100, 1) if "bb_type" in bbe.columns and n_bbe > 0 else 0.0

        return {
            "hr9": hr9, "fb_pct": fb_pct, "hardhit_pct": hardhit_pct,
            "hr_allowed": hr_allowed, "ip": ip, "n_bbe": n_bbe,
        }
    except Exception:
        return None

# ─────────────────────────────────────────
# SCORING MODEL
# ─────────────────────────────────────────
def score_matchup(bat, pit, park_factor, wind):
    breakdown = {}
    
    # Batter metrics
    breakdown["Barrel %"] = round(min(max((bat["barrel_pct"] - MLB_AVG["barrel_pct"]) * 2.0, 0), 25.0), 1)
    breakdown["ISO (Power)"] = round(min(max((bat["iso"] - MLB_AVG["iso"]) * 55, 0), 20.0), 1)
    breakdown["Hard Hit %"] = round(min(max((bat["hardhit_pct"] - MLB_AVG["hardhit_pct"]) * 0.4, 0), 10.0), 1)
    breakdown["Fly Ball %"] = round(min(max((bat["fb_pct"] - MLB_AVG["fb_pct"]) * 0.4, 0), 8.0), 1)
    breakdown["Batter HRs"] = round(min(bat["hr"] * 0.15, 5.0), 1)

    # Pitcher metrics
    breakdown["Pitcher HR/9"] = round(min(max((pit["hr9"] - MLB_AVG["hr9"]) * 7.0, 0), 15.0), 1)
    breakdown["Pitcher HRs allowed"] = round(min(pit["hr_allowed"] * 0.12, 8.0), 1)
    breakdown["Pitcher FB %"] = round(min(max((pit["fb_pct"] - MLB_AVG["fb_allowed"]) * 0.35, 0), 7.0), 1)
    breakdown["Pitcher HardHit %"] = round(min(max((pit["hardhit_pct"] - MLB_AVG["hardhit_allowed"]) * 0.35, 0), 5.0), 1)

    # Conditions
    breakdown["Park Factor"] = round(max(min((park_factor - 1.0) * 30, 8.0), -4.0), 1)
    wind_map = {"Blowing out (big boost)": 5, "Blowing out (light)": 2, "Calm / Dome": 0, "Blowing in (light)": -2, "Blowing in (strong)": -4}
    breakdown["Wind"] = wind_map.get(wind, 0)

    total = sum(breakdown.values())
    return round(max(0.0, min(100.0, total)), 1), breakdown

# ─────────────────────────────────────────
# UI & MAIN APP LOGIC
# ─────────────────────────────────────────
teams_dict = get_mlb_teams()

st.subheader("Matchup Selection")
col1, col2 = st.columns(2)
with col1:
    selected_team_name = st.selectbox("1. Select the Batting Team", list(teams_dict.keys()), index=list(teams_dict.keys()).index("Chicago White Sox") if "Chicago White Sox" in teams_dict else 0)
    team_id = teams_dict[selected_team_name]
with col2:
    pitcher_name = st.text_input("2. Enter Opposing Pitcher Name", placeholder="e.g. Kevin Gausman")

st.divider()
st.subheader("Game Conditions")
c1, c2 = st.columns(2)
with c1:
    park = st.selectbox("Ballpark", list(PARK_FACTORS.keys()))
    park_factor = PARK_FACTORS[park]
with c2:
    wind = st.selectbox("Wind", ["Calm / Dome", "Blowing out (light)", "Blowing out (big boost)", "Blowing in (light)", "Blowing in (strong)"])

st.divider()

if st.button("🔥 Generate Top 9 HR Lineup", type="primary", use_container_width=True):
    if not pitcher_name:
        st.error("Please enter an opposing pitcher.")
        st.stop()

    # 1. Fetch Pitcher
    with st.spinner(f"Looking up {pitcher_name}..."):
        pit_matches = search_player(pitcher_name)
        if not pit_matches:
            st.error(f"Could not find '{pitcher_name}'. Check spelling.")
            st.stop()
        pitcher = pit_matches[0]
        pit_stats = get_pitcher_stats(pitcher["id"])
        
        if not pit_stats:
            st.error(f"No {CURRENT_YEAR} Statcast data found for {pitcher['fullName']}.")
            st.stop()

    # 2. Fetch Team Hitters
    hitters = get_team_hitters(team_id)
    if not hitters:
        st.error(f"Could not load roster for {selected_team_name}.")
        st.stop()

    # 3. Parallel Processing for Team Statcast Data
    team_results = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.markdown(f"**Scouting {len(hitters)} {selected_team_name} hitters...** *(Running parallel Statcast queries)*")
    
    # We use ThreadPoolExecutor to fetch 5 players at a time so it doesn't take 5 minutes
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_hitter = {executor.submit(get_batter_stats, h["id"]): h for h in hitters}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_hitter)):
            h = future_to_hitter[future]
            try:
                stats = future.result()
                if stats and stats["n_bbe"] > 10: # Only count players with actual data/playing time
                    # Calculate their score against the pitcher
                    score, breakdown = score_matchup(stats, pit_stats, park_factor, wind)
                    team_results.append({
                        "name": h["name"],
                        "score": score,
                        "stats": stats,
                        "breakdown": breakdown
                    })
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(hitters))

    progress_text.empty()
    progress_bar.empty()

    if not team_results:
        st.error("Not enough data to generate a lineup.")
        st.stop()

    # 4. Sort Top 9
    top_9 = sorted(team_results, key=lambda x: x["score"], reverse=True)[:9]

    # 5. UI DISPLAY (3x3 Grid)
    st.markdown(f"## 🏆 Top 9 HR Threats: {selected_team_name} vs. {pitcher['fullName']}")
    st.caption("Ranked by probability score. Pitcher vulnerabilities and park factors applied to all.")
    st.divider()

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

    # Create 3 rows of 3 columns
    for row_idx in range(3):
        cols = st.columns(3)
        for col_idx in range(3):
            item_idx = (row_idx * 3) + col_idx
            if item_idx < len(top_9):
                player = top_9[item_idx]
                stats = player["stats"]
                
                # Assign threat level color
                if player['score'] >= 65:
                    threat = "🔴 HIGH RISK"
                elif player['score'] >= 35:
                    threat = "🟡 MED RISK"
                else:
                    threat = "🔵 LOW RISK"

                with cols[col_idx]:
                    st.markdown(f"#### {medals[item_idx]} {player['name']}")
                    st.metric("HR Score", f"{player['score']}", threat)
                    
                    st.markdown(f"""
                    * **Season HRs:** {stats['hr']}
                    * **Barrel %:** {stats['barrel_pct']}% *(Avg: 7.6%)*
                    * **ISO:** {stats['iso']:.3f}
                    * **HardHit %:** {stats['hardhit_pct']}%
                    """)
                    
                    with st.expander("Score Breakdown"):
                        for key, val in sorted(player["breakdown"].items(), key=lambda x: -x[1]):
                            if val > 0:
                                st.write(f"✅ {key}: +{val}")
                            elif val < 0:
                                st.write(f"❌ {key}: {val}")
        st.divider()