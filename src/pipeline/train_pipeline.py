import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """
    Runs the full training pipeline in sequence:
    1. Data Ingestion    → loads CSV, saves train.csv and test.csv
    2. Data Transformation → cleans, encodes, scales, saves preprocessor.pkl
    3. Model Trainer    → trains models, saves best as model.pkl
    """
    try:
        # ── Step 1: Data Ingestion ────────────────────────────────
        logging.info("Starting training pipeline")
        logging.info("Step 1: Data Ingestion")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        print(f" Data Ingestion complete")
        print(f"   train.csv → {train_path}")
        print(f"   test.csv  → {test_path}")

        # ── Step 2: Data Transformation ───────────────────────────
        logging.info("Step 2: Data Transformation")

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = \
            transformation.initiate_data_transformation(train_path, test_path)

        print(f"\n Data Transformation complete")
        print(f"   preprocessor.pkl → {preprocessor_path}")
        print(f"   Train array shape: {train_arr.shape}")
        print(f"   Test array shape : {test_arr.shape}")

        # ── Step 3: Model Training ────────────────────────────────
        logging.info("Step 3: Model Training")

        trainer = ModelTrainer()
        best_auc, best_name = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"\n Model Training complete")
        print(f"   Best model : {best_name}")
        print(f"   AUC-ROC    : {best_auc}")
        print(f"   model.pkl  → artifacts/model.pkl")

        logging.info("Training pipeline completed successfully")
        print("\n🎉 Training pipeline finished! All artifacts saved.")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()