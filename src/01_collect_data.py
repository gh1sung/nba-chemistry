##############################################################################
# NBA CHEMISTRY PROJECT — PHASE 1: DATA COLLECTION & RAW CHEMISTRY SCORES
# 
# HOW TO USE:
# 1. Open Google Colab (colab.research.google.com)
# 2. Create a new notebook
# 3. Paste this entire script into one cell (or split at the ### CELL markers)
# 4. Run it — takes ~10-15 minutes (NBA API has rate limits)
# 5. Download the 3 CSV files it produces
# 6. Upload them back to Claude for analysis
#
# Author: Jihwan Sung
# Project: NBA Player Chemistry Metric
# Date: March 2026
##############################################################################

### CELL 1: Install & Import ################################################

!pip install nba_api pandas numpy -q

import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguedashlineups, leaguedashplayerstats
from nba_api.stats.static import players, teams

print("✅ Packages loaded successfully")

### CELL 2: Pull 2-Man Lineup Data ##########################################
#
# This calls NBA.com's lineup stats endpoint with GroupQuantity=2.
# It returns every 2-player combination that shared the court,
# along with their combined plus-minus, minutes, offensive rating, etc.
#
# We pull two seasons: 2023-24 and 2024-25.
# Rate limit: we add a 1-second delay between API calls.

def get_two_man_lineups(season, season_type="Regular Season"):
    """
    Pull all 2-man lineup combinations for a given NBA season.
    Returns a DataFrame with columns like GROUP_NAME, MIN, PLUS_MINUS, etc.
    """
    print(f"  Pulling 2-man lineups for {season} ({season_type})...")
    
    lineups = leaguedashlineups.LeagueDashLineups(
        season=season,
        season_type_all_star=season_type,
        group_quantity=2,
        per_mode_detailed="Totals",  # raw totals, not per-game
        timeout=120
    )
    
    df = lineups.get_data_frames()[0]
    df["SEASON"] = season
    
    print(f"  → Got {len(df)} two-man combos")
    return df

# Pull both seasons
print("Pulling 2-man lineup data from NBA.com...")
print("(This takes a minute — NBA API is slow)\n")

lineups_2324 = get_two_man_lineups("2023-24")
time.sleep(2)  # respect rate limits

lineups_2425 = get_two_man_lineups("2024-25")
time.sleep(2)

# Combine
all_lineups = pd.concat([lineups_2324, lineups_2425], ignore_index=True)

print(f"\n✅ Total 2-man combos across both seasons: {len(all_lineups)}")
print(f"   Columns: {list(all_lineups.columns)}")

### CELL 3: Pull Individual Player Stats (BPM proxy) ########################
#
# BPM isn't directly available from nba_api, but we can get the components
# we need: individual plus-minus stats and advanced stats.
#
# For our Phase 1 baseline, we'll use E_NET_RATING from NBA.com's 
# player stats endpoint. This is the player's team's net rating 
# (pts/100 poss) while they're on court.
#
# Later in Phase 2, we can upgrade to actual BPM from Basketball Reference
# or compute our own RAPM.

def get_player_stats(season):
    """
    Pull individual player advanced stats for a given season.
    We use per-100-possession stats to normalize for pace.
    """
    print(f"  Pulling individual player stats for {season}...")
    
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Advanced",
        timeout=120
    )
    
    df = stats.get_data_frames()[0]
    df["SEASON"] = season
    
    print(f"  → Got stats for {len(df)} players")
    return df

print("Pulling individual player stats...")
print("(Another minute or so)\n")

players_2324 = get_player_stats("2023-24")
time.sleep(2)

players_2425 = get_player_stats("2024-25")
time.sleep(2)

all_players = pd.concat([players_2324, players_2425], ignore_index=True)

print(f"\n✅ Total player-season records: {len(all_players)}")

### CELL 4: Clean & Merge ####################################################
#
# The lineup data has a GROUP_NAME column like "LeBron James - Anthony Davis".
# We need to:
# 1. Parse out the two player names
# 2. Match them to individual stats
# 3. Compute chemistry = pair_net_rating - (player_A_net_rating + player_B_net_rating)

# --- Clean lineup data ---
lineup_df = all_lineups.copy()

# Parse the two player names from GROUP_NAME
# Format is "Player A - Player B"
lineup_df[["PLAYER_A", "PLAYER_B"]] = lineup_df["GROUP_NAME"].str.split(
    " - ", n=1, expand=True
)

# Key columns we care about from lineups
lineup_cols = [
    "GROUP_ID", "GROUP_NAME", "PLAYER_A", "PLAYER_B", "SEASON",
    "GP", "W", "L", "MIN", "PLUS_MINUS",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
    "PTS"
]
# Only keep columns that exist (API versions may vary)
lineup_cols = [c for c in lineup_cols if c in lineup_df.columns]
lineup_df = lineup_df[lineup_cols].copy()

# --- Clean player data ---
player_df = all_players.copy()

# Key individual columns
# NET_RATING = team's net rating while this player is on court (pts/100 poss)
# This is our individual baseline for Phase 1
player_cols = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON",
    "GP", "MIN", "NET_RATING", "OFF_RATING", "DEF_RATING",
    "PIE",  # Player Impact Estimate
    "USG_PCT", "PACE", "TS_PCT",
]
player_cols = [c for c in player_cols if c in player_df.columns]
player_df = player_df[player_cols].copy()

print(f"✅ Cleaned lineup data: {len(lineup_df)} rows")
print(f"✅ Cleaned player data: {len(player_df)} rows")

### CELL 5: Compute Raw Chemistry Scores #####################################

# Merge individual stats for Player A
chem_df = lineup_df.merge(
    player_df[["PLAYER_NAME", "SEASON", "NET_RATING", "MIN", "GP", "USG_PCT"]].rename(
        columns={
            "PLAYER_NAME": "PLAYER_A",
            "NET_RATING": "A_NET_RATING",
            "MIN": "A_TOTAL_MIN",
            "GP": "A_GP",
            "USG_PCT": "A_USG_PCT",
        }
    ),
    on=["PLAYER_A", "SEASON"],
    how="left"
)

# Merge individual stats for Player B
chem_df = chem_df.merge(
    player_df[["PLAYER_NAME", "SEASON", "NET_RATING", "MIN", "GP", "USG_PCT"]].rename(
        columns={
            "PLAYER_NAME": "PLAYER_B",
            "NET_RATING": "B_NET_RATING",
            "MIN": "B_TOTAL_MIN",
            "GP": "B_GP",
            "USG_PCT": "B_USG_PCT",
        }
    ),
    on=["PLAYER_B", "SEASON"],
    how="left"
)

# --- Compute pair net rating ---
# PLUS_MINUS is total, so we normalize to per-minute then approximate per-100-poss
# Rough conversion: 1 minute ≈ 2 possessions (NBA average ~100 poss per 48 min)
# So per-100-poss ≈ (PLUS_MINUS / MIN) * 50 ... but let's keep it simpler:
# pair_net_rating_approx = (PLUS_MINUS / MIN) * 48 * (100/PACE)
# For Phase 1, we'll use: pair_pm_per_min = PLUS_MINUS / MIN, then scale to per-48

chem_df["PAIR_PM_PER_MIN"] = chem_df["PLUS_MINUS"] / chem_df["MIN"]
chem_df["PAIR_PM_PER48"] = chem_df["PAIR_PM_PER_MIN"] * 48  # rough per-game scale

# --- Expected performance (additive assumption) ---
# A_NET_RATING and B_NET_RATING are already per-100-possessions from NBA.com
# But our pair metric is per-48-minutes, so units don't perfectly match.
# For Phase 1, we'll compute chemistry as a RANK-BASED comparison
# and also compute a simple difference metric.

# Simple approach: use the same scale for everyone
# Player individual net rating is pts/100 poss (from NBA.com advanced stats)
# Pair net rating: we'll convert PLUS_MINUS to a per-100-poss approximation
# Using ~100 possessions per 48 minutes as rough NBA average

chem_df["PAIR_NET_RATING_APPROX"] = (chem_df["PLUS_MINUS"] / chem_df["MIN"]) * 48

# Expected = sum of individual net ratings
chem_df["EXPECTED_NET_RATING"] = chem_df["A_NET_RATING"] + chem_df["B_NET_RATING"]

# CHEMISTRY = Actual pair performance - Expected from individuals
chem_df["CHEMISTRY"] = chem_df["PAIR_NET_RATING_APPROX"] - chem_df["EXPECTED_NET_RATING"]

# --- Filter for meaningful sample sizes ---
# Require at least 300 shared minutes to reduce noise
MIN_SHARED_MINUTES = 300

chem_filtered = chem_df[chem_df["MIN"] >= MIN_SHARED_MINUTES].copy()

print(f"\n✅ Chemistry scores computed!")
print(f"   Total pairs: {len(chem_df)}")
print(f"   Pairs with ≥{MIN_SHARED_MINUTES} shared minutes: {len(chem_filtered)}")

### CELL 6: Quick Sanity Checks ##############################################

print("\n" + "="*60)
print("TOP 20 CHEMISTRY PAIRS (≥300 shared minutes)")
print("="*60)

top = chem_filtered.nlargest(20, "CHEMISTRY")[
    ["PLAYER_A", "PLAYER_B", "SEASON", "MIN", "CHEMISTRY", 
     "PAIR_NET_RATING_APPROX", "EXPECTED_NET_RATING", "PLUS_MINUS"]
]
print(top.to_string(index=False))

print("\n" + "="*60)
print("BOTTOM 20 CHEMISTRY PAIRS (worst anti-chemistry)")
print("="*60)

bottom = chem_filtered.nsmallest(20, "CHEMISTRY")[
    ["PLAYER_A", "PLAYER_B", "SEASON", "MIN", "CHEMISTRY",
     "PAIR_NET_RATING_APPROX", "EXPECTED_NET_RATING", "PLUS_MINUS"]
]
print(bottom.to_string(index=False))

print("\n" + "="*60)
print("CHEMISTRY DISTRIBUTION STATS")
print("="*60)
print(chem_filtered["CHEMISTRY"].describe())
print(f"\nSkewness: {chem_filtered['CHEMISTRY'].skew():.3f}")
print(f"Kurtosis: {chem_filtered['CHEMISTRY'].kurtosis():.3f}")

### CELL 7: Case Studies — Pairs We Care About ###############################

target_pairs = [
    ("Nikola Jokic", "Jamal Murray"),
    ("Jayson Tatum", "Jaylen Brown"),
    ("Joel Embiid", "Tyrese Maxey"),
    ("LeBron James", "Anthony Davis"),
    ("Stephen Curry", "Draymond Green"),
    ("Shai Gilgeous-Alexander", "Chet Holmgren"),
    ("Luka Doncic", "Kyrie Irving"),
    ("Anthony Edwards", "Rudy Gobert"),
    ("Zach LaVine", "DeMar DeRozan"),  # anti-chemistry case
]

print("\n" + "="*60)
print("CASE STUDY PAIRS")
print("="*60)

for p_a, p_b in target_pairs:
    # Check both orderings (NBA API might list either way)
    mask = (
        ((chem_filtered["PLAYER_A"].str.contains(p_a.split()[-1], case=False, na=False)) & 
         (chem_filtered["PLAYER_B"].str.contains(p_b.split()[-1], case=False, na=False)))
        |
        ((chem_filtered["PLAYER_A"].str.contains(p_b.split()[-1], case=False, na=False)) & 
         (chem_filtered["PLAYER_B"].str.contains(p_a.split()[-1], case=False, na=False)))
    )
    result = chem_filtered[mask]
    
    if len(result) > 0:
        for _, row in result.iterrows():
            print(f"\n{row['PLAYER_A']} + {row['PLAYER_B']} ({row['SEASON']})")
            print(f"  Shared minutes: {row['MIN']:.0f}")
            print(f"  Pair net rating: {row['PAIR_NET_RATING_APPROX']:+.1f}")
            print(f"  Expected (A+B):  {row['EXPECTED_NET_RATING']:+.1f}")
            print(f"  → CHEMISTRY:     {row['CHEMISTRY']:+.1f}")
    else:
        print(f"\n{p_a} + {p_b}: NOT FOUND (may not meet minutes threshold)")

### CELL 8: Save CSVs ########################################################

# Save 3 files for upload back to Claude:

# 1. Full chemistry dataset (all pairs, all seasons)
chem_df.to_csv("chemistry_all_pairs.csv", index=False)

# 2. Filtered chemistry dataset (≥300 min only)
chem_filtered.to_csv("chemistry_filtered.csv", index=False)

# 3. Individual player stats
player_df.to_csv("player_stats_individual.csv", index=False)

print("\n✅ Saved 3 CSV files:")
print("   1. chemistry_all_pairs.csv — all pairs, both seasons")
print("   2. chemistry_filtered.csv — pairs with ≥300 shared minutes")
print("   3. player_stats_individual.csv — individual player stats")
print("\n📥 Download these and upload to Claude for analysis!")

# --- Download helper (for Colab) ---
try:
    from google.colab import files
    files.download("chemistry_all_pairs.csv")
    files.download("chemistry_filtered.csv")
    files.download("player_stats_individual.csv")
    print("\n📥 Files downloading to your device...")
except ImportError:
    print("\n(Not in Colab — files saved to current directory)")
