import joblib
import pandas as pd


model = joblib.load("model/fraud_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")


def predict_fraud(transaction):

    data = pd.DataFrame([transaction])

    data = preprocessor.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "risk": risk
    }