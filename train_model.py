"""
Train a simple CTR prediction model for API deployment
This creates a lightweight model matching your project specs
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import json

print("=" * 60)
print("CTR MODEL TRAINING - API DEPLOYMENT VERSION")
print("=" * 60)

# Generate synthetic CTR data (similar to Avazu patterns)
np.random.seed(42)
n_samples = 50000

print("\n[1/4] Generating synthetic CTR data...")
data = {
    'hour': np.random.randint(0, 24, n_samples),
    'C1': np.random.randint(1000, 1010, n_samples),
    'banner_pos': np.random.randint(0, 8, n_samples),
    'site_category': np.random.randint(0, 26, n_samples),
    'app_category': np.random.randint(0, 32, n_samples),
    'device_type': np.random.randint(0, 5, n_samples),
    'device_conn_type': np.random.randint(0, 5, n_samples),
}

# Create realistic click patterns
df = pd.DataFrame(data)
df['click_score'] = (
    -3.0  # Base (low CTR)
    - 0.1 * df['banner_pos']  # Top positions better
    + 0.3 * (df['hour'].isin([18, 19, 20, 21])).astype(int)  # Evening peak
    + 0.2 * (df['device_type'] == 1).astype(int)  # Desktop higher CTR
    + np.random.normal(0, 0.5, n_samples)
)
df['click'] = (1 / (1 + np.exp(-df['click_score'])) > np.random.random(n_samples)).astype(int)
df.drop('click_score', axis=1, inplace=True)

print(f"   ✓ Generated {len(df):,} samples")
print(f"   ✓ CTR: {df['click'].mean():.2%}")

# Split features and target
print("\n[2/4] Preparing data...")
X = df.drop('click', axis=1)
y = df['click']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Train: {len(X_train):,} samples")
print(f"   ✓ Test:  {len(X_test):,} samples")

# Train model
print("\n[3/4] Training XGBoost model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"   ✓ Model trained")
print(f"   ✓ Test AUC: {auc:.4f}")

# Save model and metadata
print("\n[4/4] Saving artifacts...")

with open('ctr_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Saved: ctr_model.pkl")

model_info = {
    'model_type': 'GradientBoostingClassifier',
    'features': list(X.columns),
    'n_features': len(X.columns),
    'test_auc': float(auc),
    'test_ctr': float(y_test.mean()),
    'version': '1.0'
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("   ✓ Saved: model_info.json")

print("\n" + "=" * 60)
print("MODEL READY FOR DEPLOYMENT!")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  AUC-ROC: {auc:.4f}")
print(f"  Features: {len(X.columns)}")
print(f"\nFiles created:")
print(f"  • ctr_model.pkl (trained model)")
print(f"  • model_info.json (metadata)")
