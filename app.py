from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"

print("Loading model from:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

# Load trained pipeline
model = joblib.load(MODEL_PATH)

# Load feature schema
feature_columns = joblib.load(FEATURES_PATH)
print("Number of expected features:", len(feature_columns))


# -----------------------------
# Risk bucketing logic
# -----------------------------
def assign_risk_bucket(prob):
    if prob < 0.20:
        return "Low Risk"
    elif prob < 0.50:
        return "Medium Risk"
    else:
        return "High Risk"


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        # -----------------------------
        # 1️⃣ User inputs (only key fields)
        # -----------------------------
        user_data = {
            "loan_amnt": float(request.form["loan_amnt"]),
            "int_rate": float(request.form["int_rate"]),
            "annual_inc": float(request.form["annual_inc"]),
            "dti": float(request.form["dti"]),
            "term": request.form["term"],
            "grade": request.form["grade"],
            "home_ownership": request.form["home_ownership"],
        }

        # -----------------------------
        # 2️⃣ Create full feature vector
        # -----------------------------
        # Fill all missing training features with NaN
        full_data = {col: None for col in feature_columns}
        full_data.update(user_data)

        df = pd.DataFrame([full_data])

        # -----------------------------
        # 3️⃣ Predict
        # -----------------------------
        prob = model.predict_proba(df)[0, 1]
        bucket = assign_risk_bucket(prob)

        result = {
            "probability": round(prob, 4),
            "bucket": bucket,
        }

    return render_template("index.html", result=result)


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
