import streamlit as st
import requests

st.title("⚾ MLB Home Run Lookup")

player_name = st.text_input("Enter MLB player name")

if player_name:

    # Search for player
    search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={player_name}"

    search_response = requests.get(search_url)
    search_data = search_response.json()

    if search_data["people"]:

        player = search_data["people"][0]

        player_id = player["id"]
        full_name = player["fullName"]

        # Get player stats
        stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=hitting"

        stats_response = requests.get(stats_url)
        stats_data = stats_response.json()

        try:
            stats = stats_data["stats"][0]["splits"][0]["stat"]

            home_runs = stats.get("homeRuns", 0)
            avg = stats.get("avg", "N/A")

            st.subheader(full_name)
            st.write(f"Home Runs: {home_runs}")
            st.write(f"Batting Average: {avg}")

        except:
            st.write("No hitting stats found.")

    else:
        st.write("Player not found.")