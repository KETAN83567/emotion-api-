# main.py (V6.1 - Bug Fix for Validation Error)
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
import random
from datetime import datetime, timedelta

# --- 1. Initialize FastAPI App ---
app = FastAPI(title="Chrono-Wellness Emotional API", version="6.1") # Version bump

# --- 2. Pydantic Models (API "Contract") ---
class PredictionInput(BaseModel):
    sleeping_time: str = Field(..., example="23:30"); waking_up_time: str = Field(..., example="07:00")
    stepsCount: int = Field(..., ge=0); caloriesBurnt: int = Field(..., ge=0); heartRate: int = Field(..., ge=30, le=200)
    songsSkipped: int = Field(..., ge=0); socialTime: int = Field(...)
    instagramTime: Optional[int]=None; xTime: Optional[int]=None; redditTime: Optional[int]=None
    youtubeTime: Optional[int]=None; musicListeningTime: Optional[int]=None
    currentHour: Optional[int] = Field(datetime.now().hour, ge=0, le=23)

class SmartRecommendation(BaseModel):
    category: str; text: str; priority: int; actionable: bool;
    impact_score: float; time_to_implement: str

class DetailedPrediction(BaseModel):
    predicted_emotion: str; confidence_score: float; wellbeing_score: int
    wellbeing_breakdown: Dict[str,float]; recommendations: List[SmartRecommendation]
    risk_factors: List[str]; positive_factors: List[str]; next_check_in: str

# --- 3. Load Trained V6 Model Artifacts ---
try:
    model=joblib.load('models/emotion_model_v6.joblib'); scaler=joblib.load('models/scaler_v6.joblib')
    label_encoder=joblib.load('models/label_encoder_v6.joblib'); model_features=joblib.load('models/features_v6.joblib')
except FileNotFoundError: raise RuntimeError("Model artifacts not found. Please run train.py first.")

# --- 4. Logic Engines (Scoring & Recommendations) ---
def process_sleep_times(sleeping_time_str, waking_up_time_str):
    FMT = '%H:%M';
    try:
        t1=datetime.strptime(sleeping_time_str,FMT); t2=datetime.strptime(waking_up_time_str,FMT)
        if t2 < t1: t2 += timedelta(days=1)
        return (t2 - t1).total_seconds()/3600, t1.hour
    except: return 7.5, 23

class SmartRecommendationEngine:
    def __init__(self):
        self.templates={'sleep':["...reducing screen time 1 hour before bed."], 'bedtime':["...shifting your bedtime 15 minutes earlier."], 'activity':["...a 10-minute walk to boost mood."],'social':["...using app timers to limit consumption."],'music':["...a 'focus' playlist."],'hr':["...4-7-8 breathing."],'relief':"Try the 5-4-3-2-1 grounding technique."}
    
    # THIS IS THE CORRECTED FUNCTION
    def generate(self, d: PredictionInput, e: str, sd: float, bh: int) -> List:
        recs=[]; R=SmartRecommendation; T=self.templates
        # Each recommendation now correctly includes the 'actionable' field.
        if sd < 6.5: recs.append(R(category="Sleep Duration",text=random.choice(T['sleep']),priority=1,actionable=True,impact_score=0.8,time_to_implement="Tonight"))
        if bh >= 1 and bh < 12: recs.append(R(category="Bedtime",text=random.choice(T['bedtime']),priority=1,actionable=True,impact_score=0.7,time_to_implement="Tonight"))
        if d.stepsCount < 5000: recs.append(R(category="Activity",text=random.choice(T['activity']),priority=2,actionable=True,impact_score=0.6,time_to_implement="10-15 min"))
        if d.socialTime > 180: recs.append(R(category="Digital Wellness",text=random.choice(T['social']),priority=2,actionable=True,impact_score=0.7,time_to_implement="Immediate"))
        if d.songsSkipped > 20: recs.append(R(category="Music Habits",text=random.choice(T['music']),priority=3,actionable=True,impact_score=0.4,time_to_implement="Next session"))
        if d.heartRate > 85: recs.append(R(category="Stress",text=random.choice(T['hr']),priority=1,actionable=True,impact_score=0.9,time_to_implement="2-5 min"))
        if e in ['Stressed', 'Anxious']: recs.append(R(category="Immediate Relief",text=T['relief'],priority=1,actionable=True,impact_score=0.7,time_to_implement="Right now"))
        if not recs: recs.append(R(category="Positive Habits",text="Your habits are well-balanced. Keep up the great work!",priority=5,actionable=False,impact_score=0.3,time_to_implement="Ongoing"))
        recs.sort(key=lambda x:x.priority); return recs[:3]

class EnhancedWellbeingScorer:
    def calculate(self, d: PredictionInput, sd: float, bh: int) -> tuple:
        s_score = max(0, 100 - (abs(sd - 7.5) * 12) - (max(0, bh - 23) * 5))
        a_score = (min(100, (d.stepsCount / 10000) * 100) + min(100, (d.caloriesBurnt / 800) * 100)) / 2
        so_score = 100 if d.socialTime <= 60 else max(0, 100 - ((d.socialTime - 60) / 300) * 100)
        hr_score = 100 if 60<=d.heartRate<=75 else max(0,100-(min(abs(d.heartRate-60),abs(d.heartRate-75))/30)*100)
        w={'sleep':0.50, 'activity':0.30, 'social':0.20}
        comp_score = (s_score * w['sleep'] + a_score * w['activity'] + so_score * w['social'])
        breakdown = {'sleep_circadian':round(s_score,1), 'physical_activity':round(a_score,1), 'digital_wellness':round(so_score,1)}
        return int(comp_score), breakdown

def identify_factors(d: PredictionInput, sd: float, bh: int) -> tuple:
    r,p=[],[]
    if sd < 6: r.append("Insufficient sleep");
    if bh >=1: r.append("Late bedtime")
    if 7 <= sd <= 8.5: p.append("Optimal sleep duration")
    if d.stepsCount < 3000: r.append("Very low activity")
    if d.stepsCount >= 8000: p.append("Excellent activity")
    return r, p

def determine_next_checkin(e:str,r:List[str])->str:
    if e in['Stressed','Anxious']or len(r)>=2: return"In 4-6 hours"
    else: return "Tomorrow"

rec_engine=SmartRecommendationEngine(); scorer=EnhancedWellbeingScorer()

@app.post("/predict_v6", response_model=DetailedPrediction)
def predict_v6(input_data: PredictionInput):
    try:
        sleep_duration, bedtime_hour = process_sleep_times(input_data.sleeping_time, input_data.waking_up_time)
        model_input_dict = {'sleep_duration':sleep_duration, 'bedtime_hour':bedtime_hour, 'stepsCount':input_data.stepsCount, 'caloriesBurnt':input_data.caloriesBurnt, 'heartRate':input_data.heartRate, 'socialTime':input_data.socialTime, 'songsSkipped':input_data.songsSkipped}
        input_df = pd.DataFrame([model_input_dict])[model_features]
        input_scaled = scaler.transform(input_df)
        prediction_proba = model.predict_proba(input_scaled)[0]
        predicted_class = np.argmax(prediction_proba)
        emotion = label_encoder.classes_[predicted_class]
        confidence = float(prediction_proba[predicted_class])
        score, breakdown = scorer.calculate(input_data, sleep_duration, bedtime_hour)
        recs = rec_engine.generate(input_data, emotion, sleep_duration, bedtime_hour)
        risks, positives = identify_factors(input_data, sleep_duration, bedtime_hour)
        next_checkin = determine_next_checkin(emotion, risks)
        return DetailedPrediction(predicted_emotion=emotion, confidence_score=confidence, wellbeing_score=score,
                                wellbeing_breakdown=breakdown, recommendations=recs, risk_factors=risks,
                                positive_factors=positives, next_check_in=next_checkin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/", include_in_schema=False)
def root(): return {"message": "Chrono-Wellness API V6 is running. Go to /docs for documentation."}
