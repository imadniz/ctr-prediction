"""
CTR Prediction API - FastAPI Application
Simple deployment-ready API for CTR predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
import json
import numpy as np

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
with open('ctr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print(f"✓ Model loaded - AUC: {model_info['test_auc']:.4f}")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="CTR Prediction API",
    description="Real-time Click-Through Rate prediction for digital advertising",
    version="1.0.0"
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class PredictionRequest(BaseModel):
    """Ad impression request"""
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    C1: int = Field(..., description="Anonymized categorical variable")
    banner_pos: int = Field(..., ge=0, le=7, description="Banner position (0-7)")
    site_category: int = Field(..., ge=0, le=25, description="Site category")
    app_category: int = Field(..., ge=0, le=31, description="App category")
    device_type: int = Field(..., ge=0, le=4, description="Device type")
    device_conn_type: int = Field(..., ge=0, le=4, description="Connection type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hour": 14,
                "C1": 1005,
                "banner_pos": 0,
                "site_category": 10,
                "app_category": 15,
                "device_type": 1,
                "device_conn_type": 2
            }
        }

class PredictionResponse(BaseModel):
    """CTR prediction response"""
    click_probability: float = Field(..., description="Predicted CTR (0-1)")
    recommendation: str = Field(..., description="Targeting recommendation")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "CTR Prediction API",
        "version": "1.0.0",
        "model_auc": model_info['test_auc']
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict Click-Through Rate for an ad impression
    
    Returns predicted CTR and targeting recommendation
    """
    try:
        # Convert request to feature array
        features = np.array([[
            request.hour,
            request.C1,
            request.banner_pos,
            request.site_category,
            request.app_category,
            request.device_type,
            request.device_conn_type
        ]])
        
        # Get prediction
        ctr_probability = float(model.predict_proba(features)[0, 1])
        
        # Generate recommendation
        if ctr_probability >= 0.10:
            recommendation = "HIGH - Serve ad (top 10% predicted CTR)"
        elif ctr_probability >= 0.05:
            recommendation = "MEDIUM - Consider serving (top 25% predicted CTR)"
        else:
            recommendation = "LOW - Skip (below average predicted CTR)"
        
        return PredictionResponse(
            click_probability=round(ctr_probability, 4),
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
