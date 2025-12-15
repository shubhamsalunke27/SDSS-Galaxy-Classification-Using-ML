#!/usr/bin/env python
"""
Script to create a sample Random Forest model for SDSS galaxy classification
This generates a mock model for testing the Flask application
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a simple Random Forest model with sample data
# Feature names matching app.py
feature_names = [
    'specobjid','u','modelFlux_i','modelFlux_z',
    'petroRad_u','petroRad_g','petroRad_i',
    'petroRad_r','petroRad_z','redshift'
]

# Generate synthetic training data
np.random.seed(42)
X_train = np.random.rand(1000, len(feature_names))
y_train = np.random.randint(0, 2, 1000)

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save the model as rf.pkl (lowercase as expected by app.py)
with open('rf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Model created and saved as 'rf.pkl'")
print(f"✓ Model features: {len(feature_names)}")
print(f"✓ Number of trees: {model.n_estimators}")
print(f"✓ Model can now be used by Flask app")
