"""
Create a simple model for testing.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Create dummy data
n_samples = 1000
np.random.seed(42)

# Generate features
data = {
    'text_length': np.random.randint(10, 280, n_samples),
    'word_count': np.random.randint(5, 50, n_samples),
    'hashtag_count': np.random.randint(0, 5, n_samples),
    'mention_count': np.random.randint(0, 3, n_samples),
    'emoji_count': np.random.randint(0, 4, n_samples),
    'flesch_reading_ease': np.random.uniform(0, 100, n_samples),
    'sentiment': np.random.uniform(-1, 1, n_samples),
    'subjectivity': np.random.uniform(0, 1, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target (engagement) based on features
engagement = (
    0.3 * df['text_length'] +
    2.0 * df['word_count'] +
    5.0 * df['hashtag_count'] +
    3.0 * df['mention_count'] +
    4.0 * df['emoji_count'] +
    0.1 * df['flesch_reading_ease'] +
    10.0 * df['sentiment'] +
    5.0 * df['subjectivity']
)
df['engagement'] = engagement + np.random.normal(0, 10, n_samples)

# Split features and target
X = df.drop('engagement', axis=1)
y = df['engagement']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_scaled, y)

# Save model and scaler
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Model, scaler, and feature names saved successfully.") 