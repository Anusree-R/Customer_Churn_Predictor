import os
import sys
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Saves any Python object (model, preprocessor) as a .pkl file.
    Used by data_transformation.py and model_trainer.py
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create artifacts/ if not exists

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a saved .pkl file back into Python.
    Used by predict_pipeline.py
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains each model and returns a report of AUC-ROC scores.
    Used by model_trainer.py to compare models and pick the best one.

    models: dict of {"Model Name": model_object}
    returns: dict of {"Model Name": auc_roc_score}
    """
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred       = model.predict(X_test)

            auc  = roc_auc_score(y_test, y_pred_proba)
            f1   = f1_score(y_test, y_pred)

            report[name] = {
                "auc_roc": round(auc, 4),
                "f1":      round(f1, 4),
                "model":   model
            }

            logging.info(f"{name} — AUC: {auc:.4f}  F1: {f1:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)