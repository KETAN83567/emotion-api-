# train.py (Final V6)
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

print("--- Starting Enhanced Model Training (V6) ---")

# --- 1. Define the Core Logic for Data Simulation ---

# This helper function is added to perform the new sleep time engineering.
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

def get_emotion_label_v6(data):
    """Your V5 logic, precisely adapted for the new features."""
    # The 'sleep_factor' now uses the engineered 'sleep_duration' feature instead of 'sleepHours'.
    sleep_optimal = 7.5
    sleep_factor = 1 - abs(data['sleep_duration'] - sleep_optimal) / sleep_optimal
    sleep_factor = max(0, sleep_factor) ** 1.2

    # Your original logic for activity and heart rate remains untouched.
    activity_factor = min(1, (data['stepsCount'] / 8000 + data['caloriesBurnt'] / 600) / 2)
    hr_factor = max(0, 1 - abs(data['heartRate'] - 65) / 50)

    # The 'music_positivity' factor is REMOVED as requested.
    # The music factor is now just the restlessness penalty.
    music_restlessness = min(data['songsSkipped'] / 25, 1)
    music_factor = 0 - (music_restlessness * 0.3) # Applying as a penalty

    # Your original social penalty logic remains untouched.
    social_penalty = (data['socialTime'] / 300) ** 1.5

    # Your original composite score logic, re-balanced for the removed features.
    composite_score = (
        sleep_factor * 0.35 +
        activity_factor * 0.25 +
        hr_factor * 0.15 +
        music_factor + # This is now a direct penalty
        social_penalty * -0.4 # Kept as a negative factor
    )

    # Your original emotion mapping logic remains untouched.
    if composite_score > 0.6: return "Euphoric"
    elif composite_score > 0.4: return "Happy"
    elif composite_score > 0.2: return "Content"
    elif composite_score > 0.0: return "Neutral"
    elif composite_score > -0.2: return "Restless"
    elif composite_score > -0.4: return "Anxious"
    else: return "Stressed"

def generate_enhanced_synthetic_data_v6(num_samples=25000):
    """Your V5 data generation, adapted for the new features."""
    np.random.seed(42)
    df = pd.DataFrame()
    print(f"📊 Generating {num_samples:,} synthetic data points...")

    # Generate realistic sleep/wake times to create our new primary features
    bedtimes_hr = np.clip(np.random.normal(23, 1.5, num_samples), 21, 26) % 24
    sleep_durations_target = np.clip(np.random.normal(7.5, 1.5, num_samples), 4, 11)
    df['sleeping_time'] = [f"{int(hr):02d}:{random.randint(0,59):02d}" for hr in bedtimes_hr]
    df['waking_up_time'] = [(datetime.strptime(bt, '%H:%M') + timedelta(hours=sd)).strftime('%H:%M') for bt, sd in zip(df['sleeping_time'], sleep_durations_target)]
    
    # Engineer the features for the training set from the generated times
    sleep_features = df.apply(lambda row: process_sleep_times(row['sleeping_time'], row['waking_up_time']), axis=1)
    df['sleep_duration'] = [sf[0] for sf in sleep_features]
    df['bedtime_hour'] = [sf[1] for sf in sleep_features]

    # Your original correlated feature generation for other variables.
    sleep_energy_factor = (df['sleep_duration'] - 4) / 8 # Uses new duration feature
    df['stepsCount'] = np.random.poisson(6000 + sleep_energy_factor * 6000)
    df['caloriesBurnt'] = np.clip(np.random.normal(400 + df['stepsCount']*0.05, 100).astype(int), 100, 1500)
    df['heartRate'] = np.clip(np.random.normal(65, 10, num_samples) + (df['stepsCount']/15000)*15, 45, 120).astype(int)
    df['socialTime'] = np.clip(np.random.exponential(120, num_samples).astype(int), 5, 600)
    
    # Simplified songsSkipped as it's no longer tied to valence.
    df['songsSkipped'] = np.random.poisson(15, num_samples).astype(int)

    # Apply the new emotion labeling logic
    df['emotion'] = df.apply(get_emotion_label_v6, axis=1)
    print("✅ Synthetic data generation complete.")
    return df

# --- 3. Train the Gradient Boosting Model ---
print("🤖 Starting Model Training...")
os.makedirs('models', exist_ok=True)
dataset = generate_enhanced_synthetic_data_v6()

# UPDATED: The final feature list for the V6 model.
features = [
    'sleep_duration',
    'bedtime_hour',
    'stepsCount',
    'caloriesBurnt',
    'heartRate',
    'socialTime',
    'songsSkipped'
]
X = dataset[features]
y = dataset['emotion']

# Your original code for encoding, splitting, and scaling remains untouched.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Your original model choice and training process remains untouched.
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Your original evaluation and saving logic remains untouched, just with new filenames.
y_pred = model.predict(X_test_scaled)
print("\nModel Performance:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(model, 'models/emotion_model_v6.joblib')
joblib.dump(scaler, 'models/scaler_v6.joblib')
joblib.dump(label_encoder, 'models/label_encoder_v6.joblib')
joblib.dump(features, 'models/features_v6.joblib')

print("\n✅ Enhanced training complete. All V6 artifacts are saved.")
