import streamlit as st
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, CustomerData

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn risk.")
st.divider()

# ── Input Form ────────────────────────────────────────────────
with st.form("prediction_form"):

    st.subheader("👤 Customer Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        Partner = st.selectbox("Has Partner", ["Yes", "No"])

    col4, col5 = st.columns(2)
    with col4:
        Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    with col5:
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

    st.divider()
    st.subheader("📞 Phone & Internet Services")
    col6, col7, col8 = st.columns(3)
    with col6:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    with col7:
        MultipleLines = st.selectbox("Multiple Lines",
                                      ["Yes", "No", "No phone service"])
    with col8:
        InternetService = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])

    st.divider()
    st.subheader("🔒 Online Services")
    col9, col10, col11 = st.columns(3)
    with col9:
        OnlineSecurity = st.selectbox("Online Security",
                                       ["Yes", "No", "No internet service"])
    with col10:
        OnlineBackup = st.selectbox("Online Backup",
                                     ["Yes", "No", "No internet service"])
    with col11:
        DeviceProtection = st.selectbox("Device Protection",
                                         ["Yes", "No", "No internet service"])

    col12, col13, col14 = st.columns(3)
    with col12:
        TechSupport = st.selectbox("Tech Support",
                                    ["Yes", "No", "No internet service"])
    with col13:
        StreamingTV = st.selectbox("Streaming TV",
                                    ["Yes", "No", "No internet service"])
    with col14:
        StreamingMovies = st.selectbox("Streaming Movies",
                                        ["Yes", "No", "No internet service"])

    st.divider()
    st.subheader("💳 Account & Billing")
    col15, col16, col17 = st.columns(3)
    with col15:
        Contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
    with col16:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col17:
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    col18, col19 = st.columns(2)
    with col18:
        MonthlyCharges = st.number_input("Monthly Charges ($)",
                                          min_value=0.0, max_value=200.0,
                                          value=65.0, step=0.5)
    with col19:
        TotalCharges = st.number_input("Total Charges ($)",
                                        min_value=0.0, max_value=10000.0,
                                        value=MonthlyCharges * tenure, step=1.0)

    st.divider()
    submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

# ── Prediction Output ─────────────────────────────────────────
if submitted:
    try:
        # Build input dataframe
        customer = CustomerData(
            gender=gender,
            SeniorCitizen=SeniorCitizen,
            Partner=Partner,
            Dependents=Dependents,
            tenure=tenure,
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges
        )

        input_df = customer.get_data_as_dataframe()
        pipeline = PredictPipeline()
        churn_proba, churn_label = pipeline.predict(input_df)

        # ── Results ───────────────────────────────────────────
        st.divider()
        st.subheader("📊 Prediction Results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                label="Churn Probability",
                value=f"{churn_proba * 100:.1f}%"
            )

        with col_b:
            risk = (
                "🔴 Critical" if churn_proba > 0.75 else
                "🟠 High"     if churn_proba > 0.50 else
                "🟡 Medium"   if churn_proba > 0.25 else
                "🟢 Low"
            )
            st.metric(label="Risk Level", value=risk)

        with col_c:
            contract_months = {
                "Month-to-month": 8,
                "One year": 18,
                "Two year": 30
            }
            remaining = contract_months.get(Contract, 8)
            revenue_at_risk = MonthlyCharges * remaining * churn_label
            st.metric(
                label="Revenue at Risk",
                value=f"${revenue_at_risk:,.0f}"
            )

        # ── Recommendation ────────────────────────────────────
        st.divider()
        st.subheader("💡 Recommended Action")

        if churn_proba > 0.75:
            st.error(
                "**Immediate intervention required.** "
                "Assign a dedicated CSM, offer contract upgrade with loyalty discount, "
                "and schedule a call within 24 hours."
            )
        elif churn_proba > 0.50:
            st.warning(
                "**Proactive outreach recommended.** "
                "Offer a tech support bundle or plan optimisation. "
                "Monitor usage over the next 30 days."
            )
        elif churn_proba > 0.25:
            st.info(
                "**Keep an eye on this customer.** "
                "Send a satisfaction survey and highlight underused features."
            )
        else:
            st.success(
                "**Customer looks healthy.** "
                "No action needed. Consider upselling an add-on service."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure you have run the training pipeline first.")