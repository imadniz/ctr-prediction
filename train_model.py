import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Training model...")

# Generate data
np.random.seed(42)
n = 10000
X = [[i % 24, i % 8, i % 26, i % 5] for i in range(n)]
y = [1 if i % 7 == 0 else 0 for i in range(n)]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save with joblib
joblib.dump(model, 'ctr_model.pkl')
print("Model saved!")