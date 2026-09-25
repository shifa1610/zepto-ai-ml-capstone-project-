# Zepto Data & AI Platform

This repository contains three capstone modules: a book-data pipeline, a Titanic analytics and modeling pipeline, and a Zepto policy support assistant.

## Setup

The project uses one consolidated `requirements.txt` in the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. The first Analytics run needs internet access to load Seaborn's Titanic dataset. The Support Assistant downloads its local embedding model on first startup.

## Run the modules

### Data pipeline

```powershell
python .\data_pipeline\pipeline.py
```

This scrapes the first five catalogue pages, cleans the data, creates the SQLite database, runs SQL queries, and saves their outputs.

### Analytics

Run EDA first:

```powershell
python .\analytics\01_eda.py
```

This saves the raw offline fallback as `analytics/titanic.csv` and cleaned data as `analytics/titanic_cleaned.csv`.

Then run modeling:

```powershell
python .\analytics\02_modeling.py
```

This trains and evaluates classifiers, compares imbalance strategies, tunes the Random Forest, predicts fare with linear regression, and saves the fitted classifier pipeline.

### Support assistant

```powershell
python -m uvicorn support_assistant.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the `POST /ask` endpoint. Mock mode is the default and requires no LLM API key.

## Design decisions

### Data pipeline
- Converts prices using the fixed project rate of **1 GBP = 105.50 INR**.
- Uses median imputation for numeric parsing failures.
- Stores books and categories in related SQLite tables.
- Saves SQL query outputs and confirms the JOIN matches the pandas merge.

### Analytics
- Saves a raw Titanic CSV fallback before cleaning.
- Applies the assignment's missing-value thresholds.
- Fits preprocessing steps only on training data using scikit-learn pipelines.
- Saves the complete tuned Random Forest pipeline with preprocessing.

### Support assistant
- Writes the eight policy documents locally and embeds them with Sentence Transformers.
- Stores embeddings in a persistent Chroma collection.
- Uses a LangGraph workflow for intent classification and routing.
- Serves structured JSON responses through FastAPI.

## Generated artifacts

- SQLite database and query output under `data_pipeline/`
- Titanic CSV files, model metrics, fitted pipeline, and charts under `analytics/`
- Policy documents and Chroma database under `support_assistant/`

See `analytics/EDA_NOTES.md` for the EDA interpretations and model conclusions.