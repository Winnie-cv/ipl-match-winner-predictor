# ============================================================
# FEATURE ENGINEERING PIPELINE
# IPL Match Winner Prediction
# ============================================================

# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd


# ============================================================
# 2. Load Clean Dataset
# ============================================================

# Read the cleaned dataset
matches = pd.read_csv("../data/cleaned_data.csv")

# Convert date column into datetime format
matches["date"] = pd.to_datetime(matches["date"])

# Sort matches chronologically
# This prevents data leakage
matches = (
    matches
    .sort_values(by=["season", "date"])
    .reset_index(drop=True)
)

print("Dataset Loaded Successfully!")
print(f"Total Matches : {len(matches)}")
print()


# ============================================================
# 3. Feature Function
# Overall Team Win Percentage
# ============================================================

def calculate_team_win_rate(team, previous_matches):
    """
    Calculates the overall win percentage of a team
    before the current match.

    Parameters
    ----------
    team : str
        Team name

    previous_matches : DataFrame
        Matches played before current match

    Returns
    -------
    float
        Win percentage of the team
    """

    # Find every previous match played by this team
    team_matches = previous_matches[
        (previous_matches["team1"] == team) |
        (previous_matches["team2"] == team)
    ]

    total_matches = len(team_matches)

    # No history available
    if total_matches == 0:
        return 0.5

    # Count wins
    wins = (team_matches["winner"] == team).sum()

    # Win percentage
    win_rate = wins / total_matches

    return win_rate


# ============================================================
# 4. Feature Function
# Head-to-Head Record
# ============================================================

def calculate_head_to_head(team1, team2, previous_matches):
    """
    Calculates Team1's historical win percentage
    against Team2.
    """

    # Find previous matches between the two teams
    h2h_matches = previous_matches[
        (
            (previous_matches["team1"] == team1) &
            (previous_matches["team2"] == team2)
        )
        |
        (
            (previous_matches["team1"] == team2) &
            (previous_matches["team2"] == team1)
        )
    ]

    total_matches = len(h2h_matches)

    # No previous meetings
    if total_matches == 0:
        return 0.5

    # Team1 wins
    team1_wins = (h2h_matches["winner"] == team1).sum()

    # Return Team1 H2H Win %
    return team1_wins / total_matches
# ============================================================
# 5. Feature Function
# Venue Performance
# ============================================================

def calculate_venue_record(team, venue, previous_matches):
    """
    Calculates a team's historical win percentage
    at the current venue.
    """

    # Matches played at this venue
    venue_matches = previous_matches[
        previous_matches["venue"] == venue
    ]

    # Team's matches at this venue
    team_matches = venue_matches[
        (venue_matches["team1"] == team) |
        (venue_matches["team2"] == team)
    ]

    total_matches = len(team_matches)

    # No previous matches at this venue
    if total_matches == 0:
        return 0.5

    wins = (team_matches["winner"] == team).sum()

    return wins / total_matches


# ============================================================
# 6. Feature Function
# Recent Form (Last 5 Matches)
# ============================================================

def calculate_recent_form(team, previous_matches):
    """
    Calculates the team's win percentage
    in its last five matches.
    """

    # Find all previous matches played by the team
    team_matches = previous_matches[
        (previous_matches["team1"] == team) |
        (previous_matches["team2"] == team)
    ]

    # Keep only last 5 matches
    recent_matches = team_matches.tail(5)

    total_matches = len(recent_matches)

    # No previous matches
    if total_matches == 0:
        return 0.5

    wins = (recent_matches["winner"] == team).sum()

    return wins / total_matches


# ============================================================
# 7. Create Empty Feature Dataset
# ============================================================

feature_dataset = []


# ============================================================
# 8. Build Features For Every Match
# ============================================================

for index, match in matches.iterrows():

    # --------------------------------------------------------
    # IMPORTANT:
    # Only use matches BEFORE the current match
    # --------------------------------------------------------

    previous_matches = matches.iloc[:index]

    # --------------------------------------------------------
    # Current Match Details
    # --------------------------------------------------------

    team1 = match["team1"]
    team2 = match["team2"]
    venue = match["venue"]

    # --------------------------------------------------------
    # Feature 1
    # Overall Team Win Percentage
    # --------------------------------------------------------

    team1_win_rate = calculate_team_win_rate(
        team1,
        previous_matches
    )

    team2_win_rate = calculate_team_win_rate(
        team2,
        previous_matches
    )

    # --------------------------------------------------------
    # Feature 2
    # Head-to-Head Win Percentage
    # --------------------------------------------------------

    h2h = calculate_head_to_head(
        team1,
        team2,
        previous_matches
    )

    # --------------------------------------------------------
    # Feature 3
    # Venue Performance
    # --------------------------------------------------------

    team1_venue_win_rate = calculate_venue_record(
        team1,
        venue,
        previous_matches
    )

    team2_venue_win_rate = calculate_venue_record(
        team2,
        venue,
        previous_matches
    )

    # --------------------------------------------------------
    # Feature 4
    # Recent Form
    # --------------------------------------------------------

    team1_recent_form = calculate_recent_form(
        team1,
        previous_matches
    )

    team2_recent_form = calculate_recent_form(
        team2,
        previous_matches
    )

    # --------------------------------------------------------
    # Create One Feature Row
    # --------------------------------------------------------

    feature_row = {

        "team1": team1,

        "team2": team2,

        "team1_win_rate": team1_win_rate,

        "team2_win_rate": team2_win_rate,

        "head_to_head": h2h,

        "team1_venue_win_rate": team1_venue_win_rate,

        "team2_venue_win_rate": team2_venue_win_rate,

        "team1_recent_form": team1_recent_form,

        "team2_recent_form": team2_recent_form,

        # Target Variable
        "winner": match["winner"]

    }

    # Save one processed match
    feature_dataset.append(feature_row)
    # ============================================================
# 9. Convert Feature Dataset to DataFrame
# ============================================================

feature_dataset = pd.DataFrame(feature_dataset)


# ============================================================
# 10. Display Dataset Information
# ============================================================

print("\n========================================")
print("Feature Engineering Completed!")
print("========================================")

print("\nFirst 5 Rows:")
print(feature_dataset.head())

print("\nDataset Shape:")
print(feature_dataset.shape)

print("\nColumns:")
print(feature_dataset.columns.tolist())


# ============================================================
# 11. Check Missing Values
# ============================================================

print("\nMissing Values:")
print(feature_dataset.isnull().sum())


# ============================================================
# 12. Save Engineered Dataset
# ============================================================

output_path = "../data/matches_features.csv"

feature_dataset.to_csv(
    output_path,
    index=False
)

print(f"\nFeature dataset saved successfully!")
print(f"Location: {output_path}")


# ============================================================
# 13. Display Basic Statistics
# ============================================================

print("\nNumerical Feature Summary:")
print(
    feature_dataset[
        [
            "team1_win_rate",
            "team2_win_rate",
            "head_to_head",
            "team1_venue_win_rate",
            "team2_venue_win_rate",
            "team1_recent_form",
            "team2_recent_form"
        ]
    ].describe()
)


# ============================================================
# 14. Pipeline Finished
# ============================================================

print("\n========================================")
print("Feature Engineering Pipeline Completed Successfully!")
print("Your dataset is ready for Machine Learning.")
print("Next Step: Encoding + Train/Test Split + Model Training")
print("========================================")