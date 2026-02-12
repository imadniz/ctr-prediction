import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Simple sample data (no numpy needed)
X = [[i % 24, i % 8, i % 26, i % 5] for i in range(10000)]
y = [1 if i % 7 == 0 else 0 for i in range(10000)]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save
with open('ctr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved!")