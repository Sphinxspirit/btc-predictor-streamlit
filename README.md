# 🔮 Bitcoin Price Movement Predictor (Streamlit App)

This Streamlit app predicts the next-day Bitcoin price movement using:
- Sentiment scores from CryptoNews and CryptoPotato (via FinBERT)
- Macroeconomic indicators (from Yahoo Finance and FRED)
- PCA-transformed features and a trained machine learning model

## 📦 Included Files
- `app.py`: Main Streamlit script
- `histgb_pca_model_clean.pkl`: Trained classifier
- `scaler.pkl`: Standard scaler
- `pca.pkl`: PCA transformer
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes sensitive files like `creds.json` and `predictions_log.csv`

## 🚀 How to Run (Locally)
```bash
pip install -r requirements.txt
streamlit run app.py

