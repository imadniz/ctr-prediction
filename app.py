from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Load model
with open('ctr_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="CTR Prediction API")

class AdRequest(BaseModel):
    hour: int
    banner_pos: int
    site_category: int
    device_type: int

@app.get("/")
def home():
    return {"message": "CTR Prediction API is running!", "status": "healthy"}

@app.post("/predict")
def predict(request: AdRequest):
    features = np.array([[
        request.hour,
        request.banner_pos,
        request.site_category,
        request.device_type
    ]])
    
    ctr = float(model.predict_proba(features)[0, 1])
    
    return {
        "click_probability": round(ctr, 4),
        "recommendation": "HIGH" if ctr > 0.1 else "LOW"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)