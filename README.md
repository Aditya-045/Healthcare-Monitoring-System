# Healthcare AI Monitoring System (CKD + Liver Disease)

This project contains a AI assisted healthcare monitoring system containing a structured lab report input with disease-specific ML classifier for Chronic Kidney Disease and Liver disease supported by a  RAG-induced chatbot (stored locally through Ollama) for follow-up questions about the result. It was built as a single Streamlit app on top of two independently trained scikit-learn pipelines.

## Project structure
```
healthcare_ai/
├── requirements.txt
├── app.py                      Streamlit app (run this to use the system)
├── data/
│   ├── kidney_disease.csv          <- place here (not included)
│   ├── indian_liver_patient.csv    <- place here (not included)
│   └── reference/                  curated RAG reference text (included)
├── models/                     trained pipelines + metadata land here
├── vector_store/                RAG vector DB lands here (created by a script)
└── src/
    ├── common.py                shared preprocessing / CV / tuning / saving helpers
    ├── preprocess_kidney.py     CKD data cleaning
    ├── preprocess_liver.py      ILPD data cleaning
    ├── train_kidney.py          baseline ensemble training (Week 1)
    ├── train_liver.py           baseline ensemble training (Week 1)
    ├── tune_kidney.py           hyperparameter tuning (Week 2)
    ├── tune_liver.py            hyperparameter tuning (Week 2)
    ├── feature_importance.py   feature importance extraction (Week 2)
    ├── build_vector_store.py   builds the RAG vector store (Week 3)
    ├── rag.py                   retrieval helper (Week 3)
    └── chatbot.py                RAG-grounded Ollama chat wrapper (Week 3)
```

## Prerequisites
- Python 3.10+ with a virtual environment (`.venv`) activated.
- [Ollama](https://ollama.com) installed and running locally, with the
  `llama3.2` model pulled (`ollama pull llama3.2`).
- The two source CSVs downloaded and placed under `data/`:
  - `kidney_disease.csv` (UCI Chronic Kidney Disease dataset)
  - `indian_liver_patient.csv` (Indian Liver Patient Dataset / ILPD)
- Internet access the first time you run `build_vector_store.py` (it
  downloads the small `all-MiniLM-L6-v2` embedding model once, then it's
  cached locally).

## For training the model

```bash
# Activate your venv, then install everything
pip install -r requirements.txt

# Check the raw data cleans up as expected
python src/preprocess_kidney.py data/kidney_disease.csv
python src/preprocess_liver.py data/indian_liver_patient.csv

# Train baseline models (compares RF / ExtraTrees / GradientBoosting for each disease)
python src/train_kidney.py --data data/kidney_disease.csv --out models
python src/train_liver.py  --data data/indian_liver_patient.csv --out models

# Tune the best model per disease
python src/tune_kidney.py --data data/kidney_disease.csv --models models
python src/tune_liver.py  --data data/indian_liver_patient.csv --models models

# Extract feature importance
python src/feature_importance.py --disease kidney --models models
python src/feature_importance.py --disease liver  --models models

# Build the RAG vector store from the reference material
python src/build_vector_store.py

# Launch the app
streamlit run app.py
```

Steps 1–5 only need to be re-run if the data, tuning, or reference text
changes. But once built and `models/` and `vector_store/` are populated, no need to repeat again.

## Using the app
1. Pick a disease in the sidebar (Chronic Kidney Disease / Liver Disease).
   The sidebar also shows which model won and its test F1.
2. Choose **Fill form** (enter one patient's values by hand) or
   **Upload CSV** (batch-predict a file matching the expected schema).
3. Run the prediction — you'll see the predicted label, confidence, and a
   global feature-importance chart.
4. Scroll to **Ask about your results** and chat with the assistant about
   your report. It's grounded by the reference material in
   `data/reference/` via RAG — if it can't find relevant material it says
   so rather than guessing, and it never gives a diagnosis.

Every prediction and chat response carries a disclaimer: this is a
decision-support demo, not a medical diagnosis.

## Troubleshooting
- **"No trained model found for ..."** — run steps 2–3 for that disease first.
- **Chat errors / can't reach Ollama** — make sure `ollama serve` is running
  and `llama3.2` is pulled (`ollama list` to check).
- **Chat says nothing relevant was found** — run step 5 (`build_vector_store.py`)
  if you haven't yet, or re-run it after editing the `.md` files in
  `data/reference/`.
- **CSV upload rejected** — the app tells you exactly which expected
  columns are missing; check against `models/<disease>_pipeline_metadata.json`
  → `numeric_features` / `categorical_features`.

