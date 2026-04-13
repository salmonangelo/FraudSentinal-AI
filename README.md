# FraudSentinel

## Setup

1. **Install dependencies**  
   `pip install -r requirements.txt`  
   *Expected output:* Successfully installed fastapi-0.104.1 uvicorn-0.24.0 xgboost-2.0.2 scikit-learn-1.3.2 imbalanced-learn-0.11.0 shap-0.44.1 chromadb-0.4.18 ollama-0.2.1 pandas-2.1.4 numpy-1.26.2 matplotlib-3.8.2 joblib-1.3.2 python-multipart-0.0.6 httpx-0.25.2 (and dependencies).

2. **Download dataset**  
   Download `creditcard.csv` from Kaggle and place it in the `data/` directory.  
   *Expected output:* No output (manual step).

3. **Train models**  
   `python train.py`  
   *Expected output:* Training pipeline completes with model evaluation metrics printed, SHAP explanations generated, and artifacts saved to `models/`. Ends with "All pipeline steps complete!".

4. **Setup RAG database**  
   `python rag_setup.py`  
   *Expected output:* "RAG setup complete: 50 cases indexed" followed by sample query results showing top 3 similar fraud cases.

## How to Run

5. **Start Ollama**  
   `ollama run mistral`  
   *Expected output:* Ollama downloads and starts the Mistral model, ready for API calls.

6. **Start the web server**  
   `uvicorn main:app --reload --port 8000`  
   *Expected output:* Server starts listening on http://127.0.0.1:8000 or http://localhost:8000.

7. **Open the application**  
   Open http://localhost:8000 in your browser.  
   *Expected output:* Fraud detection web interface loads with upload and analysis features.