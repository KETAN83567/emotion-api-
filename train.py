# train.py (Corrected and Final)
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Define the Core Logic for Data Simulation ---
def get_emotion_label_v5(data):
    sleep_factor = max(0, 1 - abs(data['sleepHours'] - 7.5) / 7.5)**1.2
    activity_factor = min(1, (data['stepsCount']/8000 + data['caloriesBurnt']/600)/2)
    hr_factor = max(0, 1 - abs(data['heartRate'] - 65) / 50)
    music_positivity = (data['avg_valence']*0.7 + data['avg_energy']*0.3) - (min(data['songsSkipped']/25, 1)*0.3)
    social_penalty = (data['socialTime'] / 300) ** 1.5
    composite_score = (sleep_factor*0.35 + activity_factor*0.25 + hr_factor*0.15 + music_positivity*0.15 - social_penalty*0.4)
    
    # CORRECTED: Multi-line if/elif/else block with proper indentation
    if composite_score > 0.6:
        return "Euphoric"
    elif composite_score > 0.4:
        return "Happy"
    elif composite_score > 0.2:
        return "Content"
    elif composite_score > 0.0:
        return "Neutral"
    elif composite_score > -0.2:
        return "Restless"
    elif composite_score > -0.4:
        return "Anxious"
    else:
        return "Stressed"

# --- 2. Generate a Realistic, Correlated Dataset ---
def generate_enhanced_synthetic_data(num_samples=15000):
    np.random.seed(42); df = pd.DataFrame()
    df['sleepHours'] = np.clip(np.random.normal(7, 1.5, num_samples), 3, 12)
    sleep_energy_factor = (df['sleepHours']-4)/8
    df['stepsCount'] = np.random.poisson(6000 + sleep_energy_factor * 6000)
    df['caloriesBurnt'] = np.clip(np.random.normal(400 + df['stepsCount']*0.05, 100).astype(int), 100, 1500)
    df['heartRate'] = np.clip(np.random.normal(65, 10, num_samples) + (df['stepsCount']/15000)*15, 45, 120).astype(int)
    df['avg_valence']=np.random.beta(2,2,num_samples); df['avg_energy']=np.random.uniform(0,1,num_samples); df['avg_danceability']=np.random.uniform(0,1,num_samples)
    df['songsSkipped'] = np.random.poisson(10 + abs(df['avg_valence'] - 0.7)*20).astype(int)
    df['socialTime'] = np.clip(np.random.exponential(120, num_samples).astype(int), 5, 600)
    df['emotion'] = df.apply(get_emotion_label_v5, axis=1)
    return df

# --- 3. Train the Gradient Boosting Model ---
print("🤖 Starting Model Training...")
os.makedirs('models', exist_ok=True)
dataset = generate_enhanced_synthetic_data()
features = ['sleepHours','stepsCount','caloriesBurnt','heartRate','songsSkipped','avg_valence','avg_energy','avg_danceability','socialTime']
X = dataset[features]; y = dataset['emotion']

label_encoder = LabelEncoder(); y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)
print(f"🎯 Final Model Accuracy: {accuracy_score(model.predict(X_test_scaled), y_test):.1%}")

# --- 4. Save Final Model Artifacts ---
joblib.dump(model, 'models/emotion_model_v5.joblib')
joblib.dump(scaler, 'models/scaler_v5.joblib')
joblib.dump(label_encoder, 'models/label_encoder_v5.joblib')
joblib.dump(features, 'models/features_v5.joblib')
print("✅ Training complete. All V5 artifacts saved to the 'models' folder.")
