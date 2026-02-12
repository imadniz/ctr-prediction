# CTR Prediction API Deployment

**FastAPI + Docker deployment for CTR prediction model**

## 🚀 Quick Start - Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
```bash
python app.py
```

Visit: http://localhost:8000/docs

## 🐳 Docker Deployment

### Build the Image
```bash
docker build -t ctr-api .
```

### Run the Container
```bash
docker run -p 8000:8000 ctr-api
```

## ☁️ Deploy to Render

### Steps:
1. Push this folder to GitHub
2. Go to render.com → Sign up
3. New Web Service → Connect your GitHub repo
4. Settings:
   - **Environment:** Docker
   - **Health Check Path:** /health
5. Deploy!

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/predict` | POST | CTR prediction |
| `/docs` | GET | Interactive API docs |

## 🧪 Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 14,
    "C1": 1005,
    "banner_pos": 0,
    "site_category": 10,
    "app_category": 15,
    "device_type": 1,
    "device_conn_type": 2
  }'
```

## 📝 Resume Bullet

```
Deployed CTR prediction model via FastAPI + Docker, serving real-time ad 
scoring with sub-100ms latency; containerized for cloud deployment on Render
```

## 📁 Files

```
ctr-api-deployment/
├── app.py              # FastAPI application
├── ctr_model.pkl       # Trained model
├── model_info.json     # Model metadata
├── requirements.txt    # Dependencies
├── Dockerfile          # Container config
├── train_model.py      # Model training script
└── README.md           # This file
```
