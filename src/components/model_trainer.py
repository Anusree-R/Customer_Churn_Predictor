import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Path where the best model will be saved.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    # Minimum AUC-ROC we consider acceptable
    min_auc_threshold: float = 0.75


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains multiple models, picks the best one by AUC-ROC,
        saves it to artifacts/model.pkl and returns performance metrics.
        """
        logging.info("Model training started")

        try:
            # ── 1. Split arrays back into X and y ────────────────
            # Last column is the target (we stacked them in data_transformation)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]
            logging.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

            # ── 2. Define models to compare ──────────────────────
            models = {
                "Logistic Regression": LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                ),
            }

            # ── 3. Train and evaluate all models ─────────────────
            # evaluate_models() is from utils.py
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # ── 4. Print comparison table ─────────────────────────
            logging.info("Model comparison:")
            print("\nModel Performance Summary:")
            print("-" * 45)
            print(f"{'Model':<25} {'AUC-ROC':>8}  {'F1':>8}")
            print("-" * 45)
            for name, metrics in model_report.items():
                print(f"{name:<25} {metrics['auc_roc']:>8}  {metrics['f1']:>8}")
            print("-" * 45)

            # ── 5. Pick the best model by AUC-ROC ────────────────
            best_name = max(
                model_report,
                key=lambda k: model_report[k]["auc_roc"]
            )
            best_auc   = model_report[best_name]["auc_roc"]
            best_model = model_report[best_name]["model"]

            logging.info(f"Best model: {best_name} with AUC-ROC: {best_auc}")
            print(f"\nBest Model: {best_name}  (AUC-ROC: {best_auc})")

            # ── 6. Check if model meets minimum threshold ─────────
            if best_auc < self.model_trainer_config.min_auc_threshold:
                raise CustomException(
                    f"No model met the minimum AUC threshold of "
                    f"{self.model_trainer_config.min_auc_threshold}. "
                    f"Best was {best_auc}",
                    sys
                )

            # ── 7. Full evaluation report for best model ──────────
            y_pred = best_model.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=["Retained", "Churned"]
            ))

            # ── 8. Save the best model ────────────────────────────
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved to artifacts/model.pkl")

            logging.info("Model training completed")

            return best_auc, best_name

        except Exception as e:
            raise CustomException(e, sys)