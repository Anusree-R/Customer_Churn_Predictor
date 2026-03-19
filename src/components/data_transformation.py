import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

        # These are the columns we'll encode and scale
        self.cat_columns = [
            "gender", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        self.num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.drop_columns = ["customerID", "Churn"]

    def _clean_data(self, df):
        """
        Fixes TotalCharges and creates engineered features.
        Underscore prefix means this is an internal helper method.
        """
        # Fix TotalCharges — blank strings to NaN then fill
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"]
        )

        # Feature engineering
        addon_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        df["num_addons"] = (df[addon_cols] == "Yes").sum(axis=1)
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

        return df

    def initiate_data_transformation(self, train_path, test_path):
        """
        Main method — reads train/test CSVs, cleans, encodes,
        scales, balances, and returns arrays ready for model training.
        """
        logging.info("Data transformation started")

        try:
            # ── 1. Load train and test data ──────────────────────
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded")

            # ── 2. Clean both sets ───────────────────────────────
            train_df = self._clean_data(train_df)
            test_df  = self._clean_data(test_df)
            logging.info("Data cleaning completed")

            # ── 3. Encode target column ──────────────────────────
            train_df["Churn"] = (train_df["Churn"] == "Yes").astype(int)
            test_df["Churn"]  = (test_df["Churn"]  == "Yes").astype(int)

            # ── 4. Label encode categorical columns ─────────────
            # We save the encoders so we can reuse them on new data
            label_encoders = {}
            for col in self.cat_columns:
                le = LabelEncoder()
                # Fit on train, transform both train and test
                train_df[col] = le.fit_transform(train_df[col].astype(str))
                test_df[col]  = le.transform(test_df[col].astype(str))
                label_encoders[col] = le
            logging.info("Categorical columns encoded")

            # ── 5. Separate features and target ─────────────────
            X_train = train_df.drop(columns=self.drop_columns)
            y_train = train_df["Churn"]
            X_test  = test_df.drop(columns=self.drop_columns)
            y_test  = test_df["Churn"]

            # ── 6. Scale numeric columns ─────────────────────────
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            logging.info("Feature scaling completed")

            # ── 7. Handle class imbalance (oversample minority) ──
            X_train_df = pd.DataFrame(X_train_scaled)
            y_train_s  = pd.Series(y_train.values)

            X_majority = X_train_df[y_train_s == 0]
            X_minority = X_train_df[y_train_s == 1]
            y_majority = y_train_s[y_train_s == 0]
            y_minority = y_train_s[y_train_s == 1]

            X_minority_up, y_minority_up = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=len(X_majority),
                random_state=42
            )

            X_train_bal = pd.concat([X_majority, X_minority_up]).values
            y_train_bal = pd.concat([y_majority, y_minority_up]).values

            # Shuffle so classes are mixed
            shuffle_idx = np.random.RandomState(42).permutation(len(y_train_bal))
            X_train_bal = X_train_bal[shuffle_idx]
            y_train_bal = y_train_bal[shuffle_idx]
            logging.info(f"Class balancing done — train size: {len(y_train_bal)}")

            # ── 8. Combine X and y back into single arrays ───────
            # We stack them so we can pass one object to model_trainer
            train_arr = np.c_[X_train_bal, y_train_bal]
            test_arr  = np.c_[X_test_scaled, y_test.values]

            # ── 9. Save the preprocessor ─────────────────────────
            # preprocessor holds both encoders and scaler
            preprocessor = {
                "label_encoders": label_encoders,
                "scaler":         scaler,
                "feature_names":  list(X_train.columns)
            }
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            logging.info("Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)