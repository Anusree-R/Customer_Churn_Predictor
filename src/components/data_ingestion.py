import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Holds the file paths for where we want to save things.
    @dataclass automatically creates __init__ for us.
    """
    raw_data_path: str  = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str  = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        # When DataIngestion is created, it gets a config with all file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw CSV, splits into train/test, saves both to artifacts/.
        Returns the paths of train.csv and test.csv for the next step.
        """
        logging.info("Data ingestion started")

        try:
            # ── 1. Load the raw dataset ──────────────────────────
            df = pd.read_csv("notebooks/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            logging.info("Dataset loaded successfully")

            # ── 2. Create artifacts/ folder if it doesn't exist ──
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            # ── 3. Save a copy of the raw data ───────────────────
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to artifacts/")

            # ── 4. Train / test split ────────────────────────────
            train_df, test_df = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["Churn"]   # Keep same churn ratio in both sets
            )
            logging.info(f"Train size: {len(train_df)}  Test size: {len(test_df)}")

            # ── 5. Save train.csv and test.csv ───────────────────
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path,  index=False)
            logging.info("train.csv and test.csv saved to artifacts/")

            logging.info("Data ingestion completed")

            # Return paths so the next step knows where to find the data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
