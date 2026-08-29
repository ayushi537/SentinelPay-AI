from agent.tools import predict_fraud


transaction = {
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "amount": 5000,
    "merchant_category": "electronics",
    "mcc_code": 5732,
    "merchant_country": "India",
    "card_present": 1,
    "device_type": "mobile",
    "device_known": 1,
    "ip_risk_score": 20,
    "is_foreign_txn": 0,
    "time_since_last_s": 3600,
    "velocity_1h": 1,
    "amount_vs_avg_ratio": 1.2,
    "account_age_days": 500,
    "has_2fa": 1,
    "credit_limit": 100000
}


result = predict_fraud(transaction)

print(result)