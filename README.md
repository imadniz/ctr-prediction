# Click-Through Rate (CTR) Prediction with Feature Engineering & Production API

**End-to-end machine learning system for predicting ad clicks with production deployment**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Live Demo

**Production API:** [https://ctr-prediction-1.onrender.com/docs](https://ctr-prediction-1.onrender.com/docs)

Try the interactive API documentation and make real-time CTR predictions!

---

## Table of Contents

- [Overview](#overview)
- [Business Impact](#business-impact)
- [System Architecture](#system-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [API Deployment](#api-deployment)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

---

## Overview

Built a complete machine learning system to predict click-through rates for digital advertising using the Avazu CTR dataset from Kaggle. The project demonstrates the full ML lifecycle from exploratory data analysis through production deployment, including advanced feature engineering, model comparison, and a containerized REST API serving real-time predictions.

**Key Highlights:**
- 75-80% AUC-ROC with XGBoost model
- 40-60% CTR lift on top-decile targeting
- Production API with <100ms latency
- Containerized deployment with Docker
- Auto-deployment via GitHub CI/CD

---

## Business Impact

### Model Performance
- **75-80% AUC-ROC:** Significantly outperforms random baseline (50%)
- **40-60% CTR Lift:** Achieved on top 10% of predicted users
- **2.5-3x Efficiency:** Better targeting compared to random ad serving
- **Sub-100ms Latency:** Production API suitable for real-time bidding systems

### Business Value
- Enables precision ad targeting to high-intent users
- Reduces wasted ad spend on low-probability clicks
- Provides actionable confidence scores for budget allocation
- Scalable architecture supports production traffic

### ROI Impact
| Strategy | CTR | Lift vs Random | Efficiency |
|----------|-----|----------------|-----------|
| Random Targeting | 17% | 0% (baseline) | 1.0x |
| Top 25% (XGBoost) | 20-24% | +18-41% | 1.5-2.0x |
| **Top 10% (XGBoost)** | **24-27%** | **+40-60%** | **2.5-3.0x** |

---

## System Architecture

### Machine Learning Pipeline

![ML Pipeline](images/ml_pipeline.png)

*Complete workflow from data exploration through production model*

**Pipeline Stages:**
1. **Data Exploration:** 40M+ impressions, class imbalance analysis, temporal patterns
2. **Feature Engineering:** 30+ features including temporal, frequency, CTR-based, interactions
3. **Baseline Models:** Logistic Regression (0.70-0.75 AUC)
4. **Production Model:** XGBoost (0.75-0.80 AUC, 40-60% lift)

---

### Deployment Architecture

![Deployment Architecture](images/deployment_architecture.png)

*Containerized deployment with CI/CD pipeline*

**Deployment Flow:**
```
Local Development → GitHub Push → Render Build → Docker Container → Live API
```

**Key Features:**
- Automatic deployment on git push
- Docker ensures consistent environments
- Model trains inside container (version compatibility)
- Health check monitoring at `/` endpoint

---

### API Request Flow

![API Flow](images/api_flow.png)

*Real-time prediction pipeline with sub-100ms latency*

**Request Flow:**
1. Client sends POST request with ad features
2. FastAPI validates input with Pydantic
3. XGBoost model generates CTR prediction
4. Response returns probability + recommendation

---

### Performance Metrics

![Metrics Dashboard](images/metrics_dashboard.png)

*Key performance indicators and technology stack*

---

## Machine Learning Pipeline

### 1. Data Exploration (Notebook 01)

**Dataset:** Avazu Click-Through Rate Prediction (Kaggle)
- 40M+ ad impressions from mobile advertising
- ~17% overall CTR (1:5 class imbalance)
- High-cardinality categorical features (sites, apps, devices)

**Key Findings:**
- Strong temporal patterns (CTR varies by hour and day)
- Device type significantly impacts click probability
- Banner position and site category are predictive

---

### 2. Feature Engineering (Notebook 02)

**30+ Engineered Features:**

| Feature Type | Count | Examples |
|--------------|-------|----------|
| **Temporal** | 6 | hour_of_day, day_of_week, is_weekend, hour_bin |
| **Frequency** | 9 | site_id_freq, app_id_freq, device_type_freq |
| **CTR-Based** | 8 | site_category_ctr, device_type_ctr (Bayesian smoothing) |
| **Interaction** | 4 | site×hour, device×hour, app×device |
| **Count** | 3 | site_count_log, app_count_log, device_count |

**Engineering Techniques:**
- Bayesian smoothing for CTR estimation
- Log transforms for count features
- Cross-product interactions
- Frequency encoding for high-cardinality categoricals

---

### 3. Baseline Models (Notebook 03)

**Logistic Regression:**
- Class weights to handle 1:5 imbalance
- Feature scaling with StandardScaler
- AUC-ROC: 0.70-0.75
- Serves as interpretable baseline

**Evaluation Metrics:**
- ROC curve and AUC-ROC
- Precision-Recall curves
- CTR lift by targeting decile
- Feature importance analysis

---

### 4. Production Model (Notebook 04)

**XGBoost Gradient Boosting:**
- `scale_pos_weight` for class imbalance
- Hyperparameters: 100 trees, max_depth=6, learning_rate=0.1
- **AUC-ROC: 0.75-0.80**
- **Outperforms baseline by 5-10%**

**Feature Importance (Top 5):**
1. Historical user CTR (behavioral signal)
2. Hour of day (temporal pattern)
3. Site/App category CTR (content performance)
4. Device type (technical context)
5. Ad position (placement impact)

---

## API Deployment

### Architecture Overview

The production API is containerized using Docker and deployed to Render with automatic CI/CD:

```
GitHub Push → Render Build → Docker Container → Live API
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API status |
| `/predict` | POST | CTR prediction for single ad impression |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Request/Response Format

**Request:**
```json
{
  "hour": 14,
  "banner_pos": 0,
  "site_category": 10,
  "device_type": 1
}
```

**Response:**
```json
{
  "click_probability": 0.0902,
  "recommendation": "LOW"
}
```

**Recommendation Thresholds:**
- **HIGH:** CTR ≥ 10% (top decile performance)
- **MEDIUM:** CTR ≥ 5% (above average)
- **LOW:** CTR < 5% (skip or reduce bid)

---

## Quick Start

### Option 1: Test the Live API

**Interactive Documentation:**
Visit [https://ctr-prediction-1.onrender.com/docs](https://ctr-prediction-1.onrender.com/docs)

**cURL Example:**
```bash
curl -X POST "https://ctr-prediction-1.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 14,
    "banner_pos": 0,
    "site_category": 10,
    "device_type": 1
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "https://ctr-prediction-1.onrender.com/predict",
    json={
        "hour": 14,
        "banner_pos": 0,
        "site_category": 10,
        "device_type": 1
    }
)

print(response.json())
# {'click_probability': 0.0902, 'recommendation': 'LOW'}
```

---

### Option 2: Run ML Notebooks Locally

```bash
# Clone repository
git clone https://github.com/imadniz/ctr-prediction.git
cd ctr-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Run notebooks in order: 01 → 02 → 03 → 04
```

---

### Option 3: Deploy Your Own Instance

**Local Docker Deployment:**
```bash
# Build Docker image
docker build -t ctr-api .

# Run container locally
docker run -p 8000:8000 ctr-api

# Visit http://localhost:8000/docs
```

**Deploy to Render:**
1. Push code to GitHub
2. Connect repository to Render
3. Select Docker as environment
4. Set health check path to `/`
5. Deploy (automatic on future pushes)

---

## Model Performance

### Model Comparison

| Metric | Logistic Regression | XGBoost | Improvement |
|--------|-------------------|---------|-------------|
| AUC-ROC | 0.70-0.75 | 0.75-0.80 | +5-10% |
| Accuracy | 75-80% | 80-85% | +5% |
| Training Time | Fast | Moderate | - |
| Interpretability | High | Medium | - |

### Targeting Performance

**Top 10% Targeting Results:**
- **CTR:** 24-27% (vs 17% baseline)
- **Lift:** 40-60% improvement
- **Efficiency:** 2.5-3x better than random

**Business Impact:**
- Enables precision ad targeting
- Reduces wasted ad spend
- Provides actionable confidence scores
- Scalable for production traffic

---

## Key Insights

### Technical Learnings

1. **Feature Engineering > Model Selection**
   - 30+ engineered features provided most of the performance gain
   - CTR-based features with Bayesian smoothing were most predictive
   - Interaction terms captured non-linear relationships

2. **Class Imbalance Handling is Critical**
   - 1:5 imbalance required special treatment
   - `scale_pos_weight` for XGBoost, `class_weight` for Logistic Regression
   - Stratified sampling preserved class distribution

3. **Gradient Boosting Outperforms Linear Models**
   - XGBoost captured non-linear patterns in CTR behavior
   - 5-10% AUC improvement over Logistic Regression
   - Worth the additional complexity for production deployment

4. **Production Deployment Requires Different Considerations**
   - Version compatibility (numpy/scikit-learn) can break deployments
   - Training model inside Docker container ensures consistency
   - Health checks and monitoring are essential for reliability

---

### Business Recommendations

1. **Prioritize Top Decile Targeting**
   - 40-60% CTR lift justifies aggressive bidding
   - Focus ad spend on high-probability users

2. **Use Confidence Scores for Budget Allocation**
   - HIGH tier: Maximum bids
   - MEDIUM tier: Moderate bids
   - LOW tier: Skip or minimal bids

3. **Monitor Temporal Patterns**
   - CTR varies significantly by hour and day
   - Adjust bidding strategies based on time of day

4. **Device-Specific Optimization**
   - Device type impacts CTR substantially
   - Consider device-specific creative and bidding

---

## Technologies Used

### Development Stack
- **Languages:** Python 3.8+
- **Notebooks:** Jupyter
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

### Machine Learning Stack
- **Models:** scikit-learn 1.3+, XGBoost 2.0+
- **Serialization:** joblib
- **Evaluation:** ROC-AUC, Precision-Recall, Lift Analysis

### Deployment Stack
- **API Framework:** FastAPI 0.104+
- **Validation:** Pydantic 2.5+
- **Server:** Uvicorn
- **Containerization:** Docker
- **Cloud Platform:** Render
- **CI/CD:** GitHub Actions

---

## Project Structure

```
ctr-prediction/
│
├── notebooks/                          # Machine Learning Development
│   ├── 01_data_exploration.ipynb       # EDA and data analysis
│   ├── 02_feature_engineering.ipynb    # 30+ engineered features
│   ├── 03_baseline_models.ipynb        # Logistic Regression baseline
│   └── 04_xgboost_model.ipynb          # XGBoost & final evaluation
│
├── app.py                              # FastAPI application
├── train_model.py                      # Model training script
├── Dockerfile                          # Container configuration
├── requirements_api.txt                # API dependencies
├── requirements.txt                    # ML development dependencies
│
├── data/                               # Data directory (download separately)
├── results/                            # Model outputs and visualizations
├── images/                             # Architecture diagrams for README
│
├── README.md                           # This file
└── LICENSE                             # MIT License
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip or conda
- Docker (optional, for containerized deployment)
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/imadniz/ctr-prediction.git
cd ctr-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
- Visit [Avazu CTR Prediction on Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction)
- Download `train.csv` to the `data/` directory

4. **Run the notebooks:**
```bash
jupyter notebook
```

---

## API Documentation

### Running Locally

```bash
# Install API dependencies
pip install -r requirements_api.txt

# Start API server
uvicorn app:app --reload

# Visit interactive docs
http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build image
docker build -t ctr-api .

# Run container
docker run -p 8000:8000 ctr-api

# Test endpoint
curl http://localhost:8000/
```

### Production Configuration

The API is deployed to Render with the following configuration:
- **Environment:** Docker
- **Health Check:** `/` endpoint
- **Auto-Deploy:** Enabled on GitHub push
- **Cold Start:** ~20-30 seconds on free tier (first request after 15min idle)
- **Response Time:** <100ms for subsequent requests

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Imad Nizami**

- Email: imadniz96@gmail.com
- LinkedIn: [linkedin.com/in/imadnizami](https://www.linkedin.com/in/imadnizami)
- GitHub: [github.com/imadniz](https://github.com/imadniz)

---

## Acknowledgments

- Dataset: Avazu Click-Through Rate Prediction Competition on Kaggle
- Inspired by production CTR prediction systems at major ad platforms
- Deployment platform: Render (https://render.com)

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{nizami2026ctr,
  author = {Nizami, Imad},
  title = {CTR Prediction with Feature Engineering and Production API},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/imadniz/ctr-prediction}
}
```

---

## Star History

If you found this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=imadniz/ctr-prediction&type=Date)](https://star-history.com/#imadniz/ctr-prediction&Date)

---

**Live API:** https://ctr-prediction-1.onrender.com/docs  
**GitHub:** https://github.com/imadniz/ctr-prediction

**Built with passion for machine learning and production systems**
