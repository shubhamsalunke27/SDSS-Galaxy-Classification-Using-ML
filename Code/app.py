from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "rf.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("rf.pkl not found in project folder.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Feature names (same order as training)
# ---------------------------
feature_names = [
    'specobjid','u','modelFlux_i','modelFlux_z',
    'petroRad_u','petroRad_g','petroRad_i',
    'petroRad_r','petroRad_z','redshift'
]

# ---------------------------
# Mapping 0 → STARFORMING, 1 → STARBURST
# ---------------------------
manual_map = {0: "STARBURST", 1: "STARFORMING"}

# ---------------------------
# Home Page
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------------------------
# Prediction Page
# ---------------------------
@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html", feature_names=feature_names)

# ---------------------------
# About Page
# ---------------------------
@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

# ---------------------------
# Submit / Prediction Route
# ---------------------------
@app.route("/submit", methods=["POST"])
def submit():

    try:
        input_values = []

        # read values in same order as training
        for fname in feature_names:
            value = request.form.get(fname)
            if value is None or value.strip() == "":
                return render_template(
                    "output.html",
                    error=True,
                    message=f"Missing value for {fname}",
                    prediction=None
                )
            input_values.append(float(value))

        # Convert to DataFrame (single row)
        data = pd.DataFrame([input_values], columns=feature_names)

        # Predict class
        pred_raw = model.predict(data)[0]

        # Map 0/1 to class label
        pred_label = manual_map[pred_raw]

        # Probability (if available)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = round(model.predict_proba(data)[0][pred_raw], 4)

        return render_template(
            "output.html",
            error=False,
            prediction=pred_label,
            probability=prob
        )

    except Exception as e:
        return render_template(
            "output.html",
            error=True,
            message=str(e),
            prediction=None
        )

# ---------------------------------------------------
# Extra function (ONLY what you requested)
# Predict multiple samples programmatically
# ---------------------------------------------------
def predict_multiple(test_cases):
    df = pd.DataFrame(test_cases, columns=feature_names)
    pred_indices = model.predict(df)

    print("Predicted subclasses:")
    for i, idx in enumerate(pred_indices):
        print(f"Test Case {i+1}: {manual_map[idx]}")

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=2222)
