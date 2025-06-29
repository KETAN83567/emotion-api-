# main.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
import random
from datetime import datetime

# --- 1. Initialize FastAPI App ---
app = FastAPI(title="Enhanced Emotional Wellbeing API", version="5.0")

# --- 2. Pydantic Models (API "Contract") ---
class PredictionInput(BaseModel):
    sleepHours: float=Field(...,ge=0,le=24); stepsCount: int=Field(...,ge=0); caloriesBurnt: int=Field(...,ge=0); heartRate: int=Field(...,ge=30,le=200); songsSkipped: int=Field(...,ge=0)
    avg_valence: float=Field(...,ge=0,le=1); avg_energy: float=Field(...,ge=0,le=1); avg_danceability: float=Field(...,ge=0,le=1); socialTime: int=Field(...)
    instagramTime: Optional[int]=None; xTime: Optional[int]=None; redditTime: Optional[int]=None; youtubeTime: Optional[int]=None; musicListeningTime: Optional[int]=None; currentHour: Optional[int]=Field(datetime.now().hour,ge=0,le=23)
class SmartRecommendation(BaseModel): category: str; text: str; priority: int; impact_score: float; time_to_implement: str
class DetailedPrediction(BaseModel):
    predicted_emotion: str; confidence_score: float; wellbeing_score: int; wellbeing_breakdown: Dict[str,float]
    recommendations: List[SmartRecommendation]; risk_factors: List[str]; positive_factors: List[str]; next_check_in: str

# --- 3. Load Trained Model Artifacts ---
try:
    model=joblib.load('models/emotion_model_v5.joblib'); scaler=joblib.load('models/scaler_v5.joblib')
    label_encoder=joblib.load('models/label_encoder_v5.joblib'); model_features=joblib.load('models/features_v5.joblib')
except FileNotFoundError: raise RuntimeError("Model artifacts not found. Please run train.py first to generate the 'models' folder.")

# --- 4. Logic Engines (Scoring & Recommendations) ---
class SmartRecommendationEngine:
    def __init__(self): self.templates={'sleep':['...reducing screen time 1 hour before bed...'],'activity':['...a 10-minute walk to boost mood.'],'social':['...using app timers to limit consumption.'],'music':["...a 'focus' playlist."],'hr':["...4-7-8 breathing."],'relief':"Try the 5-4-3-2-1 grounding technique."}
    def generate(self,d:PredictionInput,e:str)->List:
        recs=[]; R=SmartRecommendation; T=self.templates
        if d.sleepHours<6: recs.append(R(category="Sleep Optimization",text=random.choice(T['sleep']),priority=1,impact_score=0.8,time_to_implement="Tonight"))
        if d.stepsCount<5000: recs.append(R(category="Physical Activity",text=random.choice(T['activity']),priority=2,impact_score=0.6,time_to_implement="10-15 min"))
        if d.socialTime>180: recs.append(R(category="Digital Wellness",text=random.choice(T['social']),priority=2,impact_score=0.7,time_to_implement="Immediate"))
        if d.songsSkipped>15: recs.append(R(category="Music & Mood",text=random.choice(T['music']),priority=4,impact_score=0.4,time_to_implement="Next session"))
        if d.heartRate>80: recs.append(R(category="Stress",text=random.choice(T['hr']),priority=1,impact_score=0.9,time_to_implement="2-5 min"))
        if e in['Stressed','Anxious']: recs.append(R(category="Immediate Relief",text=T['relief'],priority=1,impact_score=0.7,time_to_implement="Right now"))
        if not recs: recs.append(R(category="Positive Reinforcement",text="Your habits are well-balanced. Keep it up!",priority=5,impact_score=0.3,time_to_implement="Ongoing"))
        recs.sort(key=lambda x:x.priority); return recs[:3]
class EnhancedWellbeingScorer:
    def calculate(self,d:PredictionInput)->tuple:
        s_score=max(0,100-(abs(d.sleepHours-7.5)/7.5)*60); a_score=(min(100,(d.stepsCount/10000)*100)+min(100,(d.caloriesBurnt/800)*100))/2; hr_score=100 if 60<=d.heartRate<=75 else max(0,100-(min(abs(d.heartRate-60),abs(d.heartRate-75))/30)*100); m_score=max(0,(d.avg_valence*100)-min((d.songsSkipped/30)*50,50)); so_score=100 if d.socialTime<=60 else max(0,100-((d.socialTime-60)/300)*100); w={'s':0.3,'a':0.25,'hr':0.15,'m':0.15,'so':0.15}
        comp_score=(s_score*w['s']+a_score*w['a']+hr_score*w['hr']+m_score*w['m']+so_score*w['so'])
        breakdown={'sleep':round(s_score,1),'activity':round(a_score,1),'phys_health':round(hr_score,1),'music_mood':round(m_score,1),'digital_wellness':round(so_score,1)}
        return int(comp_score),breakdown
def identify_factors(d:PredictionInput)->tuple:
    r,p=[],[]
    if d.sleepHours<6:r.append("Insufficient sleep")
    if 7<=d.sleepHours<=8.5:p.append("Optimal sleep")
    if d.stepsCount<3000:r.append("Very low activity")
    if d.stepsCount>=8000:p.append("Excellent activity")
    if d.socialTime>240:r.append("Excessive social media")
    if d.socialTime<=120:p.append("Balanced social media")
    if d.heartRate>85:r.append("Elevated heart rate")
    if 60<=d.heartRate<=75:p.append("Healthy heart rate")
    return r,p
def determine_next_checkin(e:str,r:List[str])->str:
    if e in['Stressed','Anxious']or len(r)>=3:return"In 4-6 hours"
    elif e in['Restless','Neutral']or len(r)>=1:return"Tomorrow"
    else:return"In 2-3 days"

rec_engine=SmartRecommendationEngine(); scorer=EnhancedWellbeingScorer()

# --- 5. Main API Endpoint ---
@app.post("/predict_enhanced", response_model=DetailedPrediction)
def predict_enhanced(input_data: PredictionInput):
    try:
        model_input=pd.DataFrame([input_data.dict()])[model_features]; input_scaled=scaler.transform(model_input)
        pred_proba=model.predict_proba(input_scaled)[0]; pred_class=np.argmax(pred_proba)
        emotion=label_encoder.classes_[pred_class]; confidence=float(pred_proba[pred_class])
        score,breakdown=scorer.calculate(input_data)
        recs=rec_engine.generate(input_data,emotion)
        risks,positives=identify_factors(input_data); next_checkin=determine_next_checkin(emotion,risks)
        return DetailedPrediction(predicted_emotion=emotion,confidence_score=confidence,wellbeing_score=score,
                                wellbeing_breakdown=breakdown,recommendations=recs,risk_factors=risks,
                                positive_factors=positives,next_check_in=next_checkin)
    except Exception as e: raise HTTPException(status_code=500,detail=f"Prediction failed: {e}")

@app.get("/", include_in_schema=False)
def root(): return {"message": "Wellbeing API is running. Go to /docs for documentation."}
