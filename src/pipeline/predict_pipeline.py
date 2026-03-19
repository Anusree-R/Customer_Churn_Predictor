import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Loads saved artifacts and predicts churn for new customer data.
    """
    def __init__(self):
        self.model_path       = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, input_df):
        """
        Takes a DataFrame of one customer's details,
        applies preprocessing, returns churn probability and label.
        """
        try:
            logging.info("Prediction pipeline started")

            # ── 1. Load saved artifacts ───────────────────────────
            model       = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            label_encoders = preprocessor["label_encoders"]
            scaler         = preprocessor["scaler"]
            feature_names  = preprocessor["feature_names"]

            logging.info("Model and preprocessor loaded")

            # ── 2. Clean input data ───────────────────────────────
            input_df["TotalCharges"] = pd.to_numeric(
                input_df["TotalCharges"], errors="coerce"
            )
            input_df["TotalCharges"] = input_df["TotalCharges"].fillna(
                input_df["MonthlyCharges"] * input_df["tenure"]
            )

            # Feature engineering — same as training
            addon_cols = [
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            input_df["num_addons"] = (input_df[addon_cols] == "Yes").sum(axis=1)
            input_df["avg_monthly_spend"] = (
                input_df["TotalCharges"] / input_df["tenure"].replace(0, 1)
            )

            # ── 3. Encode categorical columns ─────────────────────
            cat_columns = list(label_encoders.keys())
            for col in cat_columns:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))

            # ── 4. Drop columns not used in training ──────────────
            input_df = input_df.drop(
                columns=["customerID", "Churn"],
                errors="ignore"   # Don't crash if column doesn't exist
            )

            # ── 5. Reorder columns to match training order ─────────
            input_df = input_df[feature_names]

            # ── 6. Scale ──────────────────────────────────────────
            input_scaled = scaler.transform(input_df)

            # ── 7. Predict ────────────────────────────────────────
            churn_proba = model.predict_proba(input_scaled)[:, 1][0]
            churn_label = int(churn_proba > 0.50)

            logging.info(f"Prediction complete — Probability: {churn_proba:.4f}")

            return churn_proba, churn_label

        except Exception as e:
            raise CustomException(e, sys)


class CustomerData:
    """
    Collects raw input from the Streamlit form and
    converts it into a DataFrame the model can use.
    """
    def __init__(
        self,
        gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    ):
        self.gender           = gender
        self.SeniorCitizen    = SeniorCitizen
        self.Partner          = Partner
        self.Dependents       = Dependents
        self.tenure           = tenure
        self.PhoneService     = PhoneService
        self.MultipleLines    = MultipleLines
        self.InternetService  = InternetService
        self.OnlineSecurity   = OnlineSecurity
        self.OnlineBackup     = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport      = TechSupport
        self.StreamingTV      = StreamingTV
        self.StreamingMovies  = StreamingMovies
        self.Contract         = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod    = PaymentMethod
        self.MonthlyCharges   = MonthlyCharges
        self.TotalCharges     = TotalCharges

    def get_data_as_dataframe(self):
        """
        Converts all the input fields into a single-row DataFrame.
        This is what gets passed to PredictPipeline.predict()
        """
        try:
            data = {
                "gender":           [self.gender],
                "SeniorCitizen":    [self.SeniorCitizen],
                "Partner":          [self.Partner],
                "Dependents":       [self.Dependents],
                "tenure":           [self.tenure],
                "PhoneService":     [self.PhoneService],
                "MultipleLines":    [self.MultipleLines],
                "InternetService":  [self.InternetService],
                "OnlineSecurity":   [self.OnlineSecurity],
                "OnlineBackup":     [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport":      [self.TechSupport],
                "StreamingTV":      [self.StreamingTV],
                "StreamingMovies":  [self.StreamingMovies],
                "Contract":         [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod":    [self.PaymentMethod],
                "MonthlyCharges":   [self.MonthlyCharges],
                "TotalCharges":     [self.TotalCharges],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)