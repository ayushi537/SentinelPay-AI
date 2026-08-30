from pydantic import BaseModel
import ollama

# ============================================================
# TRANSACTION MODEL
# ============================================================

class Transaction(BaseModel):
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    amount: float
    merchant_category: str
    mcc_code: int
    merchant_country: str
    card_present: int
    device_type: str
    device_known: int
    ip_risk_score: float
    is_foreign_txn: int
    time_since_last_s: int
    velocity_1h: int
    amount_vs_avg_ratio: float
    account_age_days: int
    has_2fa: int
    credit_limit: float


# ============================================================
# AI FRAUD CHECK
# ============================================================

def fraud_check(
    transaction: Transaction,
    ml_result: dict
):
    """
    Explain the ML result using Ollama.

    IMPORTANT:
    This function DOES NOT run the ML model again.
    The ML result comes directly from app.py.
    """

    # ========================================================
    # GET EXISTING ML RESULT FROM APP.PY
    # ========================================================

    prediction = int(ml_result["prediction"])
    probability = float(ml_result["probability"])
    risk = str(ml_result["risk"])

    # Convert decimal probability to percentage
    probability_percent = probability * 100

    # Pydantic v1 & v2 cross-compatibility for dictionary extraction
    txn_dict = (
        transaction.model_dump() 
        if hasattr(transaction, "model_dump") 
        else transaction.dict()
    )

    # ========================================================
    # AI PROMPT
    # ========================================================

    prompt = f"""
You are an AI fraud risk analyst for a payment company.

The machine learning model has ALREADY evaluated this transaction.

Your ONLY job is to explain the existing ML result.

============================================================
IMPORTANT RULES
============================================================

1. The ML result is AUTHORITATIVE.
2. DO NOT perform another ML prediction.
3. DO NOT calculate a new fraud probability.
4. DO NOT change the fraud probability.
5. DO NOT change the risk level.
6. The exact ML probability is:
   {probability_percent:.6f}%
7. The exact ML risk level is:
   {risk}
8. You MUST use these exact values in your response.
9. Give exactly 3 factual reasons.
10. Use ONLY the transaction information provided.
11. Do NOT invent customer history, statistics, averages, previous transactions, or other information.
12. Recommended action MUST match the ML risk level.

============================================================
TRANSACTION
============================================================

{txn_dict}

============================================================
ML MODEL RESULT
============================================================

Prediction: {prediction}
Fraud Probability: {probability_percent:.6f}%
Risk Level: {risk}

============================================================
REQUIRED RESPONSE FORMAT
============================================================

Fraud Probability: {probability_percent:.6f}%
Risk Level: {risk}

Reasons:
- reason 1
- reason 2
- reason 3

Recommended Action:
<recommended action>
"""

    # ========================================================
    # OLLAMA INFERENCE
    # ========================================================

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract text cleanly from dictionary or object response
    if isinstance(response, dict):
        ai_content = response["message"]["content"]
    else:
        ai_content = response.message.content

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "ml_prediction": {
            "prediction": prediction,
            "fraud_probability": probability,
            "risk": risk
        },
        "ai_analysis": ai_content
    }