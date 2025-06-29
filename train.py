# train.py (Final Version)
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')

# --- 1. Feature Engineering & Logic Definition ---
def process_sleep_times(sleeping_time_str, waking_up_time_str):
    """Engineers sleep duration and bedtime hour from time strings."""
    FMT = '%H:%M'
    try:
        t1 = datetime.strptime(sleeping_time_str, FMT)
        t2 = datetime.strptime(waking_up_time_str, FMT)
        if t2 < t1: t2 += timedelta(days=1)
        return (t2 - t1).total_seconds() / 3600, t1.hour
    except (ValueError, TypeError):
        return 7.5, 23 # Return a sensible default if times are invalid

def get_emotion_label_final(data):
    """Your proven V5 logic, adapted for the new sleep and music features."""
    # The 'sleep_factor' now uses the engineered 'sleep_duration' feature.
    sleep_optimal = 7.5
    sleep_factor = 1 - abs(data['sleep_duration'] - sleep_optimal) / sleep_optimal
    sleep_factor = max(0, sleep_factor) ** 1.2

    activity_factor = min(1, (data['stepsCount']/8000 + data['caloriesBurnt']/600)/2)
    hr_factor = max(0, 1 - abs(data['heartRate'] - 65) / 50)
    
    # The 'music_positivity' factor based on valence/energy has been REMOVED.
    # The 'music_factor' is now purely a penalty for restlessness.
    music_restlessness_penalty = (min(data['songsSkipped']/25, 1) * 0.15) # Renamed for clarity

    social_penalty = (data['socialTime'] / 300) ** 1.5

    # The composite score is re-balanced to account for the removed features.
    composite_score = (
        sleep_factor * 0.40 +               # Sleep is now more important
        activity_factor * 0.25 +
        hr_factor * 0.15 -
        music_restlessness_penalty -      # Music is now only a penalty
        social_penalty * 0.4
    )
    
    if composite_score > 0.5: return "Happy"
    elif composite_score > 0.25: return "Content"
    elif composite_score > 0.0: return "Neutral"
    elif composite_score > -0.2: return "Anxious"
    else: return "Stressed"

# --- 2. Generate the Final Synthetic Dataset ---
def generate_final_synthetic_data(num_samples=15000):
    np.random.seed(42); df = pd.DataFrame()
    print(f"📊 Generating {num_samples:,} data points for the final model...")
    
    # Generate sleep/wake times to create our new primary features
    bedtimes_hr = np.clip(np.random.normal(23, 1.5, num_samples), 21, 26) % 24
    sleep_durations_target = np.clip(np.random.normal(7.5, 1.5, num_samples), 4, 11)
    df['sleeping_time'] = [f"{int(hr):02d}:{random.randint(0,59):02d}" for hr in bedtimes_hr]
    df['waking_up_time'] = [(datetime.strptime(bt, '%H:%M') + timedelta(hours=sd)).strftime('%H:%M') for bt, sd in zip(df['sleeping_time'], sleep_durations_target)]
    
    # Engineer the features from the generated times
    sleep_features = df.apply(lambda row: process_sleep_times(row['sleeping_time'], row['waking_up_time']), axis=1)
    df['sleep_duration'] = [sf[0] for sf in sleep_features]
    df['bedtime_hour'] = [sf[1] for sf in sleep_features]
    
    # Generate other features
    df['stepsCount'] = np.random.poisson(7000, num_samples)
    df['caloriesBurnt'] = np.clip(np.random.normal(400 + df['stepsCount']*0.05, 100).astype(int), 100, 1500)
    df['heartRate'] = np.clip(np.random.normal(65, 10, num_samples), 45, 120).astype(int)
    df['socialTime'] = np.clip(np.random.exponential(120, num_samples).astype(int), 5, 600)
    df['songsSkipped'] = np.random.poisson(15, num_samples).astype(int)
    
    # Apply emotion labeling with the final logic
    df['emotion'] = df.apply(get_emotion_label_final, axis=1)
    print("✅ Synthetic data generation complete.")
    return df

# --- 3. Train the Final Model ---
print("🤖 Starting Final Model Training...")
os.makedirs('models', exist_ok=True)
dataset = generate_final_synthetic_data()

# The final, definitive feature list for the model.
features = [
    'sleep_duration',
    'bedtime_hour',
    'stepsCount',
    'caloriesBurnt',
    'heartRate',
    'socialTime',
    'songsSkipped'
]
X = dataset[features]; y = dataset['emotion']

label_encoder = LabelEncoder(); y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)
print(f"🎯 Final Model Accuracy: {accuracy_score(model.predict(X_test_scaled), y_test):.1%}")

# --- 4. Save Final Model Artifacts ---
# We save these as 'v6' to distinguish them from the previous version.
joblib.dump(model, 'models/emotion_model_v6.joblib')
joblib.dump(scaler, 'models/scaler_v6.joblib')
joblib.dump(label_encoder, 'models/label_encoder_v6.joblib')
joblib.dump(features, 'models/features_v6.joblib')
print("✅ Training complete. All V6 artifacts saved to the 'models' folder.")
