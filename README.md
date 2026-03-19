# Telecom Customer Churn Prediction

End-to-end ML project predicting customer churn for a telecom company
using the IBM Telco dataset.

## Project Structure
- `src/components/` — data ingestion, transformation, model training
- `src/pipeline/`   — training and prediction pipelines
- `artifacts/`      — saved model, preprocessor, train/test splits
- `notebooks/`      — exploratory analysis
- `app.py`          — Streamlit web app

## Setup
pip install -r requirements.txt

## Run Training Pipeline
python src/pipeline/train_pipeline.py

## Run Streamlit App
streamlit run app.py
```

---

Once all files are created, your full folder structure should look like this:
```
telecom-churn/
├── artifacts/              ← auto-created when pipeline runs
├── logs/                   ← auto-created when pipeline runs
├── notebooks/
│   └── data/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       └── predict_pipeline.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md