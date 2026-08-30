import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SentinelPay AI",
    page_icon="assets/1.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    /* MAIN APP (Deep Blue Canvas) */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC; 
    }

    /* SIDEBAR (Dark Blue-Gray) */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
    }

    /* METRICS */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #2563EB !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500;
    }

    /* DASHBOARD CARD */
    .card {
        background-color: #1E293B; 
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
    }

    .card-title {
        font-size: 12px;
        color: #38BDF8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .card-value {
        font-size: 27px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 5px;
    }

    /* AI REPORT */
    .ai-report {
        background-color: #1E293B;
        border: 1px solid #2563EB;
        border-radius: 10px;
        padding: 24px;
        margin-top: 10px;
        line-height: 1.8;
        font-size: 16px;
        color: #E2E8F0;
        white-space: pre-wrap;
    }

    /* HEADER */
    .main-header {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .sub-header {
        color: #38BDF8;
        font-size: 15px;
        margin-bottom: 20px;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

@st.cache_resource
def load_pipeline():
    model_path = "model/fraud_model.pkl"
    preprocessor_path = "model/preprocessor.pkl"

    model = None
    preprocessor = None

    if os.path.exists(model_path):
        model = joblib.load(model_path)

    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


model, preprocessor = load_pipeline()

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "agent_notes" not in st.session_state:
    st.session_state.agent_notes = None

if "agent_ml_result" not in st.session_state:
    st.session_state.agent_ml_result = None

# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "amount",
    "merchant_category",
    "mcc_code",
    "merchant_country",
    "card_present",
    "device_type",
    "device_known",
    "ip_risk_score",
    "is_foreign_txn",
    "time_since_last_s",
    "velocity_1h",
    "amount_vs_avg_ratio",
    "account_age_days",
    "has_2fa",
    "credit_limit",
]

# ============================================================
# RISK FUNCTION
# ============================================================

def assess_risk_level(probability):
    if probability >= 0.15:
        return "HIGH"
    elif probability >= 0.05:
        return "MEDIUM"
    return "LOW"

# ============================================================
# ML INFERENCE
# ============================================================

def run_inference(transaction_data):
    df_raw = pd.DataFrame([transaction_data])[FEATURE_COLUMNS]

    if model is None or preprocessor is None:
        st.error("ML model or preprocessor was not found.")
        return 0, 0.0, "LOW"

    try:
        X_transformed = preprocessor.transform(df_raw)
        probability = float(model.predict_proba(X_transformed)[0][1])

        # Optional anomaly overrides
        if transaction_data["ip_risk_score"] > 0.60 or (
            transaction_data["is_foreign_txn"] == 1
            and transaction_data["has_2fa"] == 0
        ):
            probability = max(probability, 0.85)

        elif (
            transaction_data["amount_vs_avg_ratio"] > 3.0
            or transaction_data["amount"] > 50000
        ):
            probability = max(probability, 0.45)

        prediction = 1 if probability >= 0.15 else 0
        risk = assess_risk_level(probability)

        return prediction, probability, risk

    except Exception as e:
        st.error(f"ML inference error: {e}")
        return 0, 0.0, "LOW"

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    if os.path.exists("assets/1.png"):
        st.image("assets/1.png", width=60)

    st.title("SentinelPay AI")
    st.caption("AI + ML Fraud Detection Platform")
    st.divider()

    nav = st.radio(
        "Navigation",
        ["Overview", "Evaluate Risk", "Agent Report", "Metrics & Logs"],
    )

    st.divider()

    if model is not None:
        st.caption("🟢 ML Engine: Connected")
    else:
        st.caption("🔴 ML Engine: Not Found")

    col_icon, col_text = st.columns([1, 12], vertical_alignment="center")
    with col_icon:
        if os.path.exists("assets/2.png"):
            st.image("assets/2.png", width=18)
    with col_text:
        st.caption("AI Agent: Ollama / Llama 3.2")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if nav == "Overview":
    col_logo, col_title = st.columns([1, 8])

    with col_logo:
        if os.path.exists("assets/1.png"):
            st.image("assets/1.png", width=65)

    with col_title:
        st.markdown(
            '<div class="main-header">SentinelPay AI</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sub-header">'
        "AI + ML powered transaction fraud detection and risk analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    total_evaluated = len(st.session_state.history)
    fraud_count = sum(
        1 for item in st.session_state.history if item["prediction"] == 1
    )
    safe_count = total_evaluated - fraud_count
    average_risk = (
        np.mean([item["probability"] for item in st.session_state.history]) * 100
        if total_evaluated > 0
        else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Total Evaluated</div>
                <div class="card-value">{total_evaluated}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Fraud Detected</div>
                <div class="card-value">{fraud_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Safe Transactions</div>
                <div class="card-value">{safe_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Average Risk Score</div>
                <div class="card-value">{average_risk:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔄 Detection Pipeline")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**1. Transaction Data**\n\n"
            "Collect payment, merchant, device, "
            "network and account information."
        )

    with c2:
        st.info(
            "**2. ML Fraud Detection**\n\n"
            "The trained ML model processes the transaction "
            "and calculates fraud probability."
        )

    with c3:
        st.info(
            "**3. AI Risk Analysis**\n\n"
            "Ollama Llama 3.2 explains the ML result, "
            "identifies risk factors and recommends an action."
        )

    st.divider()

    st.markdown("### 🧩 System Architecture")
    st.code(
        """
Transaction Input -> Streamlit Interface -> Data Preprocessor -> ML Fraud Model
                                                                       │
                                                                       ├── Probability / Risk Level
                                                                       ▼
AI Fraud Agent (Ollama Llama 3.2) -> Explanations & Actions -> Final AI Report
        """,
        language="text",
    )

# ============================================================
# PAGE 2 — EVALUATE RISK
# ============================================================

elif nav == "Evaluate Risk":
    st.title("💳 Transaction Risk Evaluator")
    st.caption("Enter transaction information to perform an ML fraud assessment.")
    st.divider()

    st.subheader("Transaction Context")
    c1, c2, c3 = st.columns(3)

    with c1:
        amount = st.number_input(
            "Amount (₹)", min_value=0.0, value=2500.0, step=500.0
        )
        merchant_category = st.selectbox(
            "Merchant Category",
            [
                "grocery",
                "electronics",
                "travel",
                "food",
                "fashion",
                "healthcare",
                "other",
            ],
        )
        merchant_country = st.selectbox(
            "Merchant Country",
            ["India", "USA", "UK", "Singapore", "UAE", "Other"],
        )
        mcc_code = st.number_input("MCC Code", min_value=0, value=5411)
        card_present = st.selectbox(
            "Card Present", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c2:
        hour_of_day = st.slider("Hour of Day", 0, 23, 15)
        day_of_week = st.selectbox(
            "Day of Week",
            [0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][x],
        )
        is_weekend = st.selectbox(
            "Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        device_type = st.selectbox(
            "Device Type", ["mobile", "desktop", "tablet"]
        )
        device_known = st.selectbox(
            "Known Device", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c3:
        ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.15, 0.01)
        is_foreign_txn = st.selectbox(
            "Foreign Transaction",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        time_since_last_s = st.number_input(
            "Time Since Last Transaction (seconds)", min_value=0, value=120
        )
        velocity_1h = st.number_input(
            "Transactions in Last Hour", min_value=0, value=1
        )
        amount_vs_avg_ratio = st.number_input(
            "Amount vs Average Ratio", min_value=0.0, value=1.0, step=0.1
        )

    st.subheader("👤 Account Baseline")
    a1, a2, a3 = st.columns(3)

    with a1:
        account_age_days = st.number_input(
            "Account Age (Days)", min_value=0, value=365
        )

    with a2:
        has_2fa = st.selectbox(
            "2FA Status",
            [1, 0],
            format_func=lambda x: "Enabled" if x == 1 else "Disabled",
        )

    with a3:
        credit_limit = st.number_input(
            "Credit Limit (₹)", min_value=0.0, value=100000.0, step=5000.0
        )

    st.write("")

    if st.button(
        "🔍 Run Security Assessment", type="primary", use_container_width=True
    ):
        payload = {
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "amount": amount,
            "merchant_category": merchant_category,
            "mcc_code": mcc_code,
            "merchant_country": merchant_country,
            "card_present": card_present,
            "device_type": device_type,
            "device_known": device_known,
            "ip_risk_score": ip_risk_score,
            "is_foreign_txn": is_foreign_txn,
            "time_since_last_s": time_since_last_s,
            "velocity_1h": velocity_1h,
            "amount_vs_avg_ratio": amount_vs_avg_ratio,
            "account_age_days": account_age_days,
            "has_2fa": has_2fa,
            "credit_limit": credit_limit,
        }

        with st.spinner("ML model analyzing transaction..."):
            prediction, probability, risk = run_inference(payload)

        st.session_state.last_result = {
            "prediction": prediction,
            "probability": probability,
            "risk": risk,
            "payload": payload,
        }

        st.session_state.agent_notes = None
        st.session_state.agent_ml_result = None

        st.session_state.history.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "amount": amount,
                "prediction": prediction,
                "probability": probability,
                "risk": risk,
            }
        )

        st.success("Transaction assessment completed successfully.")

    if st.session_state.last_result:
        result = st.session_state.last_result

        st.divider()
        st.subheader("📊 Assessment Findings")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Decision",
                "🚨 FRAUD DETECTED"
                if result["prediction"] == 1
                else "✅ PASSED SAFE",
            )

        with r2:
            st.metric(
                "Fraud Probability", f"{result['probability'] * 100:.2f}%"
            )

        with r3:
            st.metric("Risk Level", result["risk"])

        if result["risk"] == "HIGH":
            st.error(
                "🚨 HIGH RISK — Block the transaction or route it for manual verification."
            )
        elif result["risk"] == "MEDIUM":
            st.warning(
                "⚠️ MEDIUM RISK — Request additional authentication or verification."
            )
        else:
            st.success(
                "✅ LOW RISK — Transaction can proceed through standard payment flow."
            )

# ============================================================
# PAGE 3 — AI AGENT REPORT
# ============================================================

elif nav == "Agent Report":
    st.title("🤖 AI Fraud Risk Analysis")
    st.caption("ML prediction explained by the Ollama-powered AI fraud agent.")
    st.divider()

    if not st.session_state.last_result:
        st.info("No transaction has been evaluated yet.")
        st.write(
            "Go to **Evaluate Risk**, enter transaction details, and click **Run Security Assessment**."
        )

    else:
        current = st.session_state.last_result
        txn_data = current["payload"]

        st.subheader("Transaction Summary")
        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric("Transaction Amount", f"₹{txn_data['amount']:,.2f}")

        with s2:
            st.metric(
                "ML Fraud Probability", f"{current['probability'] * 100:.2f}%"
            )

        with s3:
            st.metric("ML Risk Level", current["risk"])

        st.divider()

        if st.button(
            "🧠 Generate AI Diagnostic Report",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.agent_notes = None
            st.session_state.agent_ml_result = None

            with st.spinner("AI Agent is analyzing the transaction..."):
                try:
                    from agent.fraud_agent import Transaction, fraud_check

                    transaction = Transaction(**txn_data)
                    agent_result = fraud_check(transaction, current)

                    st.session_state.agent_notes = agent_result.get(
                        "ai_analysis", "No AI analysis returned."
                    )
                    st.session_state.agent_ml_result = agent_result.get(
                        "ml_prediction", {}
                    )

                    st.success("AI diagnostic report generated successfully.")

                except Exception as e:
                    st.session_state.agent_notes = None
                    st.session_state.agent_ml_result = None
                    st.error(f"AI Agent Error: {str(e)}")

        if st.session_state.agent_notes:
            st.divider()
            st.subheader("🧠 AI Diagnostic Report")
            st.markdown(
                f"""
                <div class="ai-report">
                {st.session_state.agent_notes}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.get("agent_ml_result"):
                st.divider()
                st.subheader("🔍 ML Model Output")
                ml_result = st.session_state.agent_ml_result

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Prediction",
                        "🚨 FRAUD"
                        if ml_result.get("prediction") == 1
                        else "✅ SAFE",
                    )
                with m2:
                    st.metric(
                        "Fraud Probability",
                        f"{ml_result.get('fraud_probability', 0.0) * 100:.2f}%",
                    )
                with m3:
                    st.metric("Risk Level", ml_result.get("risk", "N/A"))

# ============================================================
# PAGE 4 — METRICS & LOGS
# ============================================================

elif nav == "Metrics & Logs":
    st.title("📈 Metrics & Audit Logs")
    st.caption("Session-level analytics for evaluated transactions.")
    st.divider()

    if not st.session_state.history:
        st.info("No transactions have been evaluated yet.")

    else:
        df_log = pd.DataFrame(st.session_state.history)

        total_transactions = len(df_log)
        fraud_transactions = (df_log["prediction"] == 1).sum()
        safe_transactions = (df_log["prediction"] == 0).sum()
        average_probability = df_log["probability"].mean() * 100

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Transactions", total_transactions)

        with m2:
            st.metric("Fraud Detected", fraud_transactions)

        with m3:
            st.metric("Safe Transactions", safe_transactions)

        with m4:
            st.metric("Average Fraud Score", f"{average_probability:.2f}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Categorization")
            counts = df_log["risk"].value_counts().reset_index()
            counts.columns = ["Level", "Count"]

            fig_pie = px.pie(
                counts,
                values="Count",
                names="Level",
                hole=0.4,
                color="Level",
                color_discrete_map={
                    "LOW": "#10b981",
                    "MEDIUM": "#f59e0b",
                    "HIGH": "#ef4444",
                },
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Risk Score Trajectory")
            chart_data = df_log.copy()
            chart_data["probability_pct"] = chart_data["probability"] * 100

            fig_line = px.line(
                chart_data,
                x="timestamp",
                y="probability_pct",
                markers=True,
                labels={
                    "timestamp": "Time",
                    "probability_pct": "Fraud Prob (%)",
                },
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        st.subheader("📜 Detailed Audit Trail")
        st.dataframe(df_log, use_container_width=True)