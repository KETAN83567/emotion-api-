# main.py (V6 - Final Production Version with Your Full Logic)
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
import random
from datetime import datetime, timedelta
import math

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Chrono-Wellness Emotional API",
    version="6.0",
    description="The definitive API providing emotion prediction and hyper-personalized recommendations based on sleep timing and other key wellness factors."
)

# --- 2. Pydantic Models (The Final API Contract) ---
class PredictionInput(BaseModel):
    # New primary inputs for sleep
    sleeping_time: str = Field(..., example="23:30", description="Time user went to sleep (HH:MM).")
    waking_up_time: str = Field(..., example="07:00", description="Time user woke up (HH:MM).")
    
    # All other essential variables from your V5 logic
    stepsCount: int = Field(..., ge=0)
    caloriesBurnt: int = Field(..., ge=0)
    heartRate: int = Field(..., ge=30, le=200)
    songsSkipped: int = Field(..., ge=0)
    socialTime: int = Field(...)
    instagramTime: Optional[int] = None
    xTime: Optional[int] = None
    redditTime: Optional[int] = None
    youtubeTime: Optional[int] = None
    musicListeningTime: Optional[int] = None
    currentHour: Optional[int] = Field(datetime.now().hour, ge=0, le=23)

class SmartRecommendation(BaseModel):
    category: str; text: str; priority: int; actionable: bool;
    impact_score: float; time_to_implement: str

class DetailedPrediction(BaseModel):
    predicted_emotion: str; confidence_score: float; wellbeing_score: int
    wellbeing_breakdown: Dict[str, float]; recommendations: List[SmartRecommendation]
    risk_factors: List[str]; positive_factors: List[str]; next_check_in: str

# --- 3. Load Trained V6 Model Artifacts ---
try:
    model=joblib.load('models/emotion_model_v6.joblib'); scaler=joblib.load('models/scaler_v6.joblib')
    label_encoder=joblib.load('models/label_encoder_v6.joblib'); model_features=joblib.load('models/features_v6.joblib')
except FileNotFoundError: raise RuntimeError("Model V6 artifacts not found. Please run the V6 train.py first.")

# --- 4. Your Brilliant V5 Logic Engines, Adapted for V6 Sleep Features ---
def process_sleep_times(sleeping_time_str, waking_up_time_str):
    FMT = '%H:%M';
    try:
        t1=datetime.strptime(sleeping_time_str, FMT); t2=datetime.strptime(waking_up_time_str, FMT)
        if t2 < t1: t2 += timedelta(days=1)
        return (t2 - t1).total_seconds()/3600, t1.hour
    except: return 7.5, 23

class SmartRecommendationEngine:
    def __init__(self):
        # This is YOUR original, detailed template from V5
        self.recommendation_templates = {
            'sleep': {'insufficient': ["Your sleep is below optimal. Try setting a consistent bedtime routine.","Consider reducing screen time 1 hour before bed."],'excessive': ["Excessive sleep may indicate underlying fatigue. Focus on sleep quality."]},
            'activity': {'low': ["Low physical activity detected. Start with a 10-minute walk.","Try the 2-minute rule: commit to just 2 minutes of exercise."]},
            'social_media': {'excessive': ["High social media usage detected. Try the 20-20-20 rule.","Consider using app timers to limit consumption."]},
            'music': {'restless': ["Frequent song skipping suggests restlessness. Try a 'focus' playlist."]},
            'physiological': {'high_hr': ["Elevated heart rate detected. Try 4-7-8 breathing."]}
        }
    def generate_smart_recommendations(self, data: PredictionInput, emotion: str, sleep_duration: float, bedtime_hour: int) -> List[SmartRecommendation]:
        recs = []; R = SmartRecommendation
        # This is YOUR original recommendation logic from V5, now using the new sleep features
        if sleep_duration < 6: recs.append(R(category="Sleep Optimization", text=random.choice(self.recommendation_templates['sleep']['insufficient']), priority=1, impact_score=0.8, time_to_implement="Tonight (30 min setup)"))
        elif sleep_duration > 9: recs.append(R(category="Sleep Regulation", text=random.choice(self.recommendation_templates['sleep']['excessive']), priority=3, impact_score=0.4, time_to_implement="1-2 weeks"))
        if bedtime_hour >= 1: recs.append(R(category="Sleep Timing", text=f"A late bedtime of {data.sleeping_time} can disrupt your circadian rhythm.", priority=2, impact_score=0.7, time_to_implement="Tonight"))
        if data.stepsCount < 5000: recs.append(R(category="Physical Activity", text=f"Start small: {random.choice(self.recommendation_templates['activity']['low'])}", priority=2, impact_score=0.6, time_to_implement="Immediate (10-15 min)"))
        if data.socialTime > 180:
            text = random.choice(self.recommendation_templates['social_media']['excessive'])
            if data.instagramTime and data.instagramTime > 90: text += f" Especially on Instagram ({data.instagramTime} min)."
            recs.append(R(category="Digital Wellness", text=text, priority=2, impact_score=0.7, time_to_implement="Immediate (app settings)"))
        if data.songsSkipped > 15: recs.append(R(category="Music & Mood", text=random.choice(self.recommendation_templates['music']['restless']), priority=4, impact_score=0.4, time_to_implement="Next listening session"))
        if data.heartRate > 80: recs.append(R(category="Stress Management", text=random.choice(self.recommendation_templates['physiological']['high_hr']), priority=1, impact_score=0.9, time_to_implement="Immediate (2-5 min)"))
        if emotion in ['Stressed', 'Anxious']: recs.append(R(category="Immediate Relief", text="Try the 5-4-3-2-1 grounding technique.", priority=1, impact_score=0.7, time_to_implement="Right now (2-3 min)"))
        if not recs:
             positive_habits = [];
             if 7 <= sleep_duration <= 8.5: positive_habits.append("excellent sleep schedule")
             if data.stepsCount >= 8000: positive_habits.append("great physical activity")
             if positive_habits: recs.append(R(category="Positive Reinforcement", text=f"You're maintaining {', '.join(positive_habits)}. Keep it up!", priority=5, impact_score=0.3, time_to_implement="Ongoing"))
        recs.sort(key=lambda x: (x.priority, -x.impact_score)); return recs[:min(4, len(recs))]

class EnhancedWellbeingScorer:
    def calculate_comprehensive_score(self, data: PredictionInput, sleep_duration: float, bedtime_hour: int) -> tuple:
        # This is YOUR original scoring logic, adapted for new features
        sleep_score = max(0, 100 - (abs(sleep_duration - 7.5) * 12) - (max(0, bedtime_hour - 23) * 5))
        activity_score = (min(100, (data.stepsCount / 10000) * 100) + min(100, (data.caloriesBurnt / 800) * 100)) / 2
        hr_score = 100 if 60 <= data.heartRate <= 75 else max(0, 100 - (min(abs(data.heartRate - 60), abs(data.heartRate - 75)) / 30) * 100)
        music_score = max(0, 80 - min((data.songsSkipped / 25) * 100, 80)) # Simplified score without valence
        social_score = 100 if data.socialTime <= 60 else max(0, 60 - ((data.socialTime - 180) / 240) * 60) if data.socialTime > 180 else 100 - ((data.socialTime - 60) / 120) * 40
        weights = {'sleep': 0.4, 'activity': 0.25, 'heart_rate': 0.15, 'music': 0.05, 'social': 0.15}
        composite_score = (sleep_score * weights['sleep'] + activity_score * weights['activity'] + hr_score * weights['heart_rate'] + music_score * weights['music'] + social_score * weights['social'])
        breakdown = {'sleep_circadian': round(sleep_score,1), 'physical_activity': round(activity_score,1), 'physiological_health': round(hr_score,1), 'music_engagement': round(music_score,1), 'digital_wellness': round(social_score,1)}
        return int(composite_score), breakdown

def identify_risk_and_positive_factors(data: PredictionInput, sleep_duration: float) -> tuple:
    risk_factors, positive_factors = [], []
    if sleep_duration < 6: risk_factors.append(f"Insufficient sleep ({sleep_duration:.1f} hours)")
    if data.stepsCount < 3000: risk_factors.append(f"Very low physical activity ({data.stepsCount:,} steps)")
    if data.socialTime > 300: risk_factors.append(f"Excessive social media usage ({data.socialTime} minutes)")
    if data.heartRate > 90: risk_factors.append(f"Elevated heart rate ({data.heartRate} bpm)")
    if 7 <= sleep_duration <= 8.5: positive_factors.append(f"Optimal sleep duration ({sleep_duration:.1f} hours)")
    if data.stepsCount >= 8000: positive_factors.append(f"Excellent physical activity ({data.stepsCount:,} steps)")
    return risk_factors, positive_factors

def determine_next_checkin(predicted_emotion: str, risk_factors: List[str]) -> str:
    if predicted_emotion in ['Stressed', 'Anxious'] or len(risk_factors) >= 2: return "Check back in 4-6 hours"
    else: return "Check back tomorrow"

rec_engine=SmartRecommendationEngine(); scorer=EnhancedWellbeingScorer()

# --- 5. Main API Endpoint ---
@app.post("/predict_v6", response_model=DetailedPrediction)
def predict_v6(input_data: PredictionInput):
    try:
        sleep_duration, bedtime_hour = process_sleep_times(input_data.sleeping_time, input_data.waking_up_time)
        model_input_dict = {
            'sleep_duration': sleep_duration, 'bedtime_hour': bedtime_hour,
            'stepsCount': input_data.stepsCount, 'caloriesBurnt': input_data.caloriesBurnt,
            'heartRate': input_data.heartRate, 'socialTime': input_data.socialTime,
            'songsSkipped': input_data.songsSkipped
        }
        input_df = pd.DataFrame([model_input_dict])[model_features]
        input_scaled = scaler.transform(input_df)
        
        prediction_proba = model.predict_proba(input_scaled)[0]
        predicted_class = np.argmax(prediction_proba)
        emotion = label_encoder.classes_[predicted_class]
        confidence = float(prediction_proba[predicted_class])
        
        score, breakdown = scorer.calculate_comprehensive_score(input_data, sleep_duration, bedtime_hour)
        recs = rec_engine.generate_smart_recommendations(input_data, emotion, sleep_duration, bedtime_hour)
        risks, positives = identify_risk_and_positive_factors(input_data, sleep_duration)
        next_checkin = determine_next_checkin(emotion, risks)
        
        return DetailedPrediction(
            predicted_emotion=emotion, confidence_score=confidence, wellbeing_score=score,
            wellbeing_breakdown=breakdown, recommendations=recs, risk_factors=risks,
            positive_factors=positives, next_check_in=next_checkin)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/", include_in_schema=False)
def root(): return {"message": "Chrono-Wellness API V6 is running. Go to /docs for documentation."}
