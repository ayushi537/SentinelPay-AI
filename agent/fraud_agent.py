from agent.tools import predict_fraud
from pydantic import BaseModel
import ollama


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


def fraud_check(transaction: Transaction):

    # ML model prediction
    prediction = predict_fraud(transaction.model_dump())

    # Convert probability to percentage
    probability = prediction["fraud_probability"] * 100
    risk = prediction["risk"]

    # Prompt for Llama
    prompt = f"""
You are an AI fraud risk analyst for a payment company.

Analyze the transaction using ONLY the information provided below.

IMPORTANT RULES:
- The ML model result is authoritative.
- Do NOT calculate or change the fraud probability.
- Do NOT invent averages, statistics, customer history, or other information.
- Use only the transaction values provided.
- Give exactly 3 factual reasons.
- Keep the reasons concise.
- The recommended action must match the risk level.

Transaction:
{transaction.model_dump()}

ML Model Result:
{prediction}

Fraud probability to display:
{probability:.6f}%

Risk level to display:
{risk}

Return exactly this format:

Fraud Probability: {probability:.6f}%
Risk Level: {risk}

Reasons:
- reason 1
- reason 2
- reason 3

Recommended Action:
<recommended action>
"""

    # Ask Llama
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "ml_prediction": prediction,
        "ai_analysis": response.message.content
    }