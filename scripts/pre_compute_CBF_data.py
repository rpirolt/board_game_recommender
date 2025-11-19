import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import pickle
import warnings

warnings.filterwarnings('ignore')


# -----------------------------
# Helper function
# -----------------------------
def semicolon_to_list(value):
    """Convert a semicolon-delimited string into a list of strings.
       Handles NaN, empty strings, and already lists.
    """
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    value_str = str(value)
    if value_str.strip() == "":
        return []
    return [item.strip() for item in value_str.split(';') if item.strip()]


# -----------------------------
# Load CSV
# -----------------------------
games_file = "data/games_master_data.csv"

usecols = [
    'bgg_id', 'name', 'description', 'image', 'thumbnail', 'bgg_link',
    'avg_rating', 'bgg_rating', 'users_rated', 'game_weight',
    'players_min', 'players_max', 'players_best',
    'time_min', 'time_max', 'time_avg',
    'simple_game_mechanics', 'simple_game_categories', 'game_types',
    'year_published'
]

dtype_dict = {
    'bgg_id': 'int64', 'avg_rating': 'float64', 'bgg_rating': 'float64', 
    'users_rated': 'int64', 'game_weight': 'float64', 'players_best': 'float64',
    'players_min': 'int64', 'players_max': 'int64', 'time_min': 'int64',
    'time_max': 'int64', 'time_avg': 'int64'
}

games_df = pd.read_csv(
    games_file,
    usecols=usecols,
    converters={
        'simple_game_mechanics': semicolon_to_list,
        'simple_game_categories': semicolon_to_list,
        'game_types': semicolon_to_list
    },
    dtype=dtype_dict
)

# Rename columns for simplicity
games_df.rename(
    columns={
        'simple_game_categories': 'game_categories',
        'simple_game_mechanics': 'game_mechanics'
    },
    inplace=True
)

# -----------------------------
# Clean multi-label columns
# -----------------------------
for col in ['game_categories', 'game_mechanics', 'game_types']:
    games_df[col] = games_df[col].apply(semicolon_to_list)

# Fill numeric NaNs with zeros for safe scaling
for col in ['game_weight', 'players_best', 'time_avg']:
    games_df[col] = games_df[col].fillna(0)


# -----------------------------
# Create MultiLabelBinarizer encoders
# -----------------------------
mlb_game_categories = MultiLabelBinarizer()
mlb_game_mechanics = MultiLabelBinarizer()
mlb_game_types = MultiLabelBinarizer()

cat_features = mlb_game_categories.fit_transform(games_df['game_categories'])
mech_features = mlb_game_mechanics.fit_transform(games_df['game_mechanics'])
type_features = mlb_game_types.fit_transform(games_df['game_types'])

# -----------------------------
# Scale numeric features
# -----------------------------
scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(games_df[['game_weight', 'players_best', 'time_avg']])

# -----------------------------
# Combine features (weighted)
# -----------------------------
weighted_features = np.hstack([
    cat_features * 1.5,
    mech_features * 2.0,
    type_features * 1.0,
    numeric_features * 0.5
])

# -----------------------------
# Save precomputed data
# -----------------------------
precompute_data = {
    'games_df': games_df,
    'mlb_game_categories': mlb_game_categories,
    'mlb_game_mechanics': mlb_game_mechanics,
    'mlb_game_types': mlb_game_types,
    'scaler': scaler,
    'weighted_features': weighted_features
}

with open('precomputed_CBF.pkl', 'wb') as f:
    pickle.dump(precompute_data, f)

print("Precomputed CBF data saved to 'precomputed_CBF.pkl'.")
