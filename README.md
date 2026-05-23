# 🚨 FraudSentinel AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.2-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent fraud detection system powered by machine learning and AI, featuring real-time analysis, SHAP explanations, and a Retrieval-Augmented Generation (RAG) system for case similarity matching.

## ✨ Features

- 🔍 **Real-time Fraud Detection**: Upload transaction data and get instant fraud predictions
- 🤖 **AI-Powered Analysis**: Uses XGBoost and ensemble methods for high accuracy
- 📊 **Explainable AI**: SHAP-based feature importance explanations
- 🔄 **RAG System**: Semantic search for similar fraud cases using ChromaDB and Google Gemini
- 🌐 **Web Interface**: Modern, responsive UI built with HTML/CSS/JavaScript
- 📈 **Model Training Pipeline**: Automated training with evaluation metrics
- 🛡️ **Imbalanced Data Handling**: Techniques for fraud detection's class imbalance

## 📸 Screenshots

![Fraud Detection Interface](dashboard.png)
![Analysis Results](analysis.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key (for AI features)
- Kaggle account (for dataset)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/salmonangelo/FraudSentinal-AI.git
   cd FraudSentinal-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place it in the `data/` directory

5. **Train the model**
   ```bash
   python train.py
   ```

6. **Setup RAG database**
   ```bash
   python rag_setup.py
   ```

### Running the Application

1. **Create a `.env` file**
   ```bash
   echo GEMINI_API_KEY=your_key_here > .env
   ```

2. **Start the web server**
   ```bash
   uvicorn main:create_app --factory --reload --port 8000
   ```

3. **Add API key authentication**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Then add the generated key to `.env`:
   ```bash
   echo FRAUD_API_KEY=your_generated_key >> .env
   ```

   The backend requires `X-API-Key` for all protected endpoints except `/`, `/analyst`, and `/health`.

4. **Open your browser**
   - Navigate to `http://localhost:8000`
   - Upload transaction data and analyze for fraud!

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload CSV for analysis |
| `/analyze` | POST | Analyze transactions |
| `/similar-cases` | GET | Find similar fraud cases |

## 🔐 Authentication

All protected API endpoints requires an `X-API-Key` header.
Generate a key with:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```
Add it to `.env` as:
```bash
FRAUD_API_KEY=<generated_key>
```
The frontend uses this key automatically for demo purposes.

## ⏱️ Rate Limiting

API endpoints are protected with rate limiting to prevent abuse:
- **Scoring endpoints** (`POST /score`, `POST /explain`, `POST /api/investigate/stream`): **10 requests/minute**
- **Streaming endpoint** (`GET /stream`): **60 requests/minute** (higher limit for real-time updates)
- **Public endpoints** (`GET /`, `GET /analyst`, `GET /health`): **Unlimited**

When a rate limit is exceeded, the server returns a **429 (Too Many Requests)** status with the message: "Rate limit reached. Please wait 60 seconds."

## 🏗️ Project Structure

```
fraudsentinel/
├── main.py              # FastAPI application
├── train.py             # Model training script
├── rag_setup.py         # RAG database setup
├── requirements.txt     # Python dependencies
├── static/              # Web interface files
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/              # Trained models and metrics
├── data/                # Dataset (not included)
└── chroma_db/           # Vector database (generated)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Powered by [Google Gemini](https://developers.generativeai.google/) and [ChromaDB](https://www.trychroma.com/)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [XGBoost](https://xgboost.readthedocs.io/)