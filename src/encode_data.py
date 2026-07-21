# src/encode_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ============================================================
# 1. Define Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. Load Feature Dataset
# ============================================================

df = pd.read_csv(DATA_DIR / "matches_features.csv")

# ============================================================
# 3. Collect All Unique Team Names
# ============================================================

all_teams = pd.concat(
    [df["team1"], df["team2"], df["winner"]]
).dropna().unique()

# ============================================================
# 4. Create and Fit Label Encoder
# ============================================================

team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

# ============================================================
# 5. Encode Team Columns
# ============================================================

df["team1_enc"] = team_encoder.transform(df["team1"])
df["team2_enc"] = team_encoder.transform(df["team2"])
df["winner_enc"] = team_encoder.transform(df["winner"])

# ============================================================
# 6. Save Encoder
# ============================================================

joblib.dump(team_encoder, MODEL_DIR / "label_encoder.pkl")

# ============================================================
# 7. Save Encoded Dataset
# ============================================================

df.to_csv(DATA_DIR / "matches_encoded.csv", index=False)

# ============================================================
# 8. Display Results
# ============================================================

print("Encoding completed successfully!\n")

print(f"Number of Teams: {len(team_encoder.classes_)}")

print("\nTeam Mapping:")

mapping = dict(
    zip(
        team_encoder.classes_,
        team_encoder.transform(team_encoder.classes_)
    )
)

for team, code in mapping.items():
    print(f"{team:30} -> {code}")

print("\nEncoded Dataset Preview:")

print(
    df[
        [
            "team1",
            "team1_enc",
            "team2",
            "team2_enc",
            "winner",
            "winner_enc"
        ]
    ].head()
)

print("\nEncoded dataset saved to:")
print(DATA_DIR / "matches_encoded.csv")

print("\nLabel encoder saved to:")
print(MODEL_DIR / "label_encoder.pkl")