# app_final.py (final debugged version)
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
import re
import time
import cloudpickle

# ---------------------------- CONFIG ----------------------------
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
CRYPTO_NEWS_API_KEY = st.secrets["CRYPTO_NEWS_API_KEY"]
FRED_API_KEY = st.secrets["FRED_API_KEY"]

FINBERT_API = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

TICKERS = {
    "bitcoin": "BTC-USD",
    "gold": "GC=F",
    "sp500": "^GSPC",
    "dxy": "DX-Y.NYB"
}

FRED_CODES = {
    "interest_rate": "FEDFUNDS",
    "inflation": "CPIAUCSL"
}
# Load model using cloudpickle
with open("histgb_pca_model_clean.pkl", "rb") as f:
    model = cloudpickle.load(f)

pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------- FUNCTIONS ----------------------------
def fetch_news(source):
    url = f"https://cryptonews-api.com/api/v1/category"
    params = {
        "section": "general",
        "items": 10,
        "page": 1,
        "source": source,
        "token": CRYPTO_NEWS_API_KEY
    }
    r = requests.get(url, params=params)
    articles = r.json().get("data", [])
    texts = []
    for art in articles:
        summary = art.get("text") or art.get("content", "").split(".")[0]
        texts.append(summary)
    return texts

def call_finbert(news_list):
    results_df = []
    news_list = news_list[:5]
    for idx, news in enumerate(news_list):
        if not isinstance(news, str) or not news.strip():
            results_df.append({"positive": 0.0, "neutral": 0.0, "negative": 0.0})
            continue
        payload = {"inputs": news}
        for attempt in range(5):
            try:
                response = requests.post(FINBERT_API, headers=HEADERS, json=payload, timeout=30)
                response.raise_for_status()
                output = response.json()
                
                # Get raw scores
                scores_raw = {item["label"].lower(): item["score"] for item in output[0]}
                
                # Ensure fixed column order
                aligned_scores = {
                    "positive": scores_raw.get("positive", 0.0),
                    "neutral": scores_raw.get("neutral", 0.0),
                    "negative": scores_raw.get("negative", 0.0)
                }
                
                results_df.append(aligned_scores)
                break
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ FinBERT error on article {idx+1}, attempt {attempt+1}/5: {e}")
                time.sleep(2)
            except Exception as ex:
                st.warning(f"❌ Failed to analyze article {idx+1}: {ex}")
                results_df.append({"positive": 0.0, "neutral": 0.0, "negative": 0.0})
                break
    return pd.DataFrame(results_df)

def aggregate_sentiments(sentiment_df):
    scaled = sentiment_df.copy()
    for col in scaled.columns:
        scaled[col] = (scaled[col] - scaled[col].min()) / (scaled[col].max() - scaled[col].min() + 1e-8)
    weighted = scaled.copy()
    for col in ["positive", "negative"]:
        weighted[col] = np.where(scaled[col] > 0.75, scaled[col] * 1.5, scaled[col])
        weighted[col] = np.clip(weighted[col], 0, 1)
    weighted["neutral"] = scaled["neutral"]
    return weighted.mean().to_dict(), (scaled > 0.75).sum().to_dict()

def fetch_yahoo_data(ticker, date):
    data = yf.Ticker(ticker).history(start=date, end=date + timedelta(days=1))
    if not data.empty:
        return {
            "open": round(data["Open"].iloc[0], 2),
            "high": round(data["High"].iloc[0], 2),
            "low": round(data["Low"].iloc[0], 2),
            "close": round(data["Close"].iloc[0], 2),
            "volume": int(data["Volume"].iloc[0]) if ticker != TICKERS["dxy"] else None,
            "change_pct": round(((data["Close"].iloc[0] - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100, 2)
        }
    else:
        st.warning(f"⚠️ No trading data for {ticker} on {date.strftime('%Y-%m-%d')}, using previous available data.")
        return fetch_yahoo_data(ticker, date - timedelta(days=1))

def fetch_fred(code, month):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": code,
        "observation_start": f"{month}-01",
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    res = requests.get(url, params=params).json()
    try:
        return float(res["observations"][0]["value"])
    except:
        prev_month = (datetime.strptime(month, "%Y-%m") - timedelta(days=30)).strftime("%Y-%m")
        return fetch_fred(code, prev_month)

def make_prediction(input_data):
    expected_cols = list(scaler.feature_names_in_)

    # SAFETY CHECK
    if len(input_data) != len(expected_cols):
        raise ValueError(f"❌ Input length mismatch! Got {len(input_data)}, expected {len(expected_cols)}")

    # Align input values to expected column order
    input_dict = dict(zip(expected_cols, input_data))
    input_df = pd.DataFrame([input_dict])[expected_cols]

    # DEBUG VIEW
    st.write("📄 Aligned Input DataFrame:")
    st.dataframe(input_df)

    # Transform
    x_scaled = scaler.transform(input_df)
    x_pca = pca.transform(x_scaled)
    proba = model.predict_proba(x_pca)[0][1]
    prediction = "Increase" if proba >= 0.72 else "Decrease"
    return prediction, round(proba, 4)


import gspread
from oauth2client.service_account import ServiceAccountCredentials

def log_prediction(record):
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
        client = gspread.authorize(creds)

        sheet = client.open("BTC Predictions Log").sheet1  # Must match your actual Google Sheet name
        sheet.append_row(list(record.values()))
        st.success("✅ Logged to Google Sheet successfully.")
    except Exception as e:
        st.warning(f"⚠️ Logging to Google Sheets failed: {e}")

# ---------------------------- STREAMLIT UI ----------------------------
st.set_page_config(page_title="Next Day Bitcoin Price Movement", layout="wide")
st.title("🔮 Next Day Bitcoin Price Movement Predictor")

date = st.date_input("Select a date", datetime.today() - timedelta(days=1))
month = date.strftime("%Y-%m")

if "news_loaded" not in st.session_state:
    st.session_state.news_loaded = False

sentiment_features = []
aggregated_display = {}
news_by_source = {"CryptoNews": [], "CryptoPotato": []}
edited_news_by_source = {}

# ------------------------------------
# STEP 1: FETCH NEWS + ENABLE EDITING
# ------------------------------------
if not st.session_state.news_loaded:
    if st.button("📥 Fetch News"):
        for src in ["CryptoNews", "CryptoPotato"]:
            try:
                news = fetch_news(src)
                news_by_source[src] = news
                st.session_state[src] = "\n\n".join(news)  # store for text_area default
            except Exception as e:
                st.warning(f"⚠️ Could not fetch {src}: {e}")
                st.session_state[src] = ""
        st.session_state.news_loaded = True
        st.rerun()

# ------------------------------------
# STEP 2: SHOW TEXT BOXES + RUN PREDICTION
# ------------------------------------
if st.session_state.news_loaded:
    st.subheader("📝 Edit News Articles")
    for src in ["CryptoNews", "CryptoPotato"]:
        default_text = st.session_state.get(src, "")
        user_input = st.text_area(f"{src} Articles (5 max, one per paragraph)", default_text, height=300)
        edited_news_by_source[src] = [para.strip() for para in user_input.split("\n\n") if para.strip()]

    if st.button("🔮 Make Prediction"):
        for src in ["CryptoNews", "CryptoPotato"]:
            try:
                news_by_source[src] = edited_news_by_source[src]
                scores_df = call_finbert(news_by_source[src])
                st.write(f"📊 FinBERT Scores for {src}:", scores_df)

                weighted_avg, extreme_count = aggregate_sentiments(scores_df)
                total_articles = len(scores_df)

                pct_scores = {
                    "positive_pct": extreme_count.get("positive", 0) / total_articles,
                    "neutral_pct": extreme_count.get("neutral", 0) / total_articles,
                    "negative_pct": extreme_count.get("negative", 0) / total_articles
                }

                sentiment_features.extend([
                    weighted_avg["positive"],
                    weighted_avg["neutral"],
                    weighted_avg["negative"],
                    pct_scores["positive_pct"],
                    pct_scores["neutral_pct"],
                    pct_scores["negative_pct"]
                ])
            except Exception as e:
                st.warning(f"⚠️ Failed for {src}: {e}")
                sentiment_features.extend([0.0] * 6)
                news_by_source[src] = []

        st.markdown("**Aggregated Sentiment**")
        st.write("🔎 News by Source:", news_by_source)
        sentiment_feature_labels = {
	    "cryptonews_positive_weighted": sentiment_features[0],
            "cryptonews_neutral_weighted": sentiment_features[1],
            "cryptonews_negative_weighted": sentiment_features[2],
            "cryptonews_positive_pct": sentiment_features[3],
            "cryptonews_neutral_pct": sentiment_features[4],
            "cryptonews_negative_pct": sentiment_features[5],
            "cryptopotato_positive_weighted": sentiment_features[6],
            "cryptopotato_neutral_weighted": sentiment_features[7],
            "cryptopotato_negative_weighted": sentiment_features[8],
            "cryptopotato_positive_pct": sentiment_features[9],
            "cryptopotato_neutral_pct": sentiment_features[10],
            "cryptopotato_negative_pct": sentiment_features[11],
        }
        st.markdown("### 🧠 Sentiment Features by Source")
        st.json(sentiment_feature_labels)	

        # Average across both sources
        if len(sentiment_features) == 12:
            aggregated_sentiments = [
                (sentiment_features[0] + sentiment_features[6]) / 2,
                (sentiment_features[1] + sentiment_features[7]) / 2,
                (sentiment_features[2] + sentiment_features[8]) / 2,
                (sentiment_features[3] + sentiment_features[9]) / 2,
                (sentiment_features[4] + sentiment_features[10]) / 2,
                (sentiment_features[5] + sentiment_features[11]) / 2
            ]
        elif len(sentiment_features) == 6:
            aggregated_sentiments = sentiment_features
        else:
            st.warning("⚠️ Sentiment features incomplete. Defaulting to 0s.")
            aggregated_sentiments = [0.0] * 6

        # Fetch BTC + macro data
        st.subheader("📈 Bitcoin Price Data")
        btc = fetch_yahoo_data(TICKERS["bitcoin"], date)
        st.json(btc)

        st.subheader("📊 Macroeconomic Indicators")
        macro = {}
        for k, t in TICKERS.items():
            if k != "bitcoin":
                try:
                    macro[k] = fetch_yahoo_data(t, date)
                except Exception as e:
                    st.warning(f"⚠️ Failed to fetch {k.upper()} data: {e}")
                    macro[k] = {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0, "change_pct": 0}
        st.json(macro)

        st.subheader("🏩 Fed Indicators")
        fed = {
            "interest_rate": fetch_fred(FRED_CODES["interest_rate"], month),
            "inflation": fetch_fred(FRED_CODES["inflation"], month)
        }
        st.json(fed)

        # ========== BUILD FINAL INPUT DICT SAFELY ==========
        final_input_dict = {
            "S&P_500_Open": macro["sp500"].get("open", 0),
            "S&P_500_High": macro["sp500"].get("high", 0),
            "S&P_500_Low": macro["sp500"].get("low", 0),
            "S&P_500_Close": macro["sp500"].get("close", 0),
            "S&P_500_Volume": macro["sp500"].get("volume", 0),
            "S&P_500_%_Change": macro["sp500"].get("change_pct", 0),

            "Gold_Prices_Open": macro["gold"].get("open", 0),
            "Gold_Prices_High": macro["gold"].get("high", 0),
            "Gold_Prices_Low": macro["gold"].get("low", 0),
            "Gold_Prices_Close": macro["gold"].get("close", 0),
            "Gold_Prices_Volume": macro["gold"].get("volume", 0),
            "Gold_Prices_%_Change": macro["gold"].get("change_pct", 0),

            "US_Dollar_Index_DXY_Open": macro["dxy"].get("open", 0),
            "US_Dollar_Index_DXY_High": macro["dxy"].get("high", 0),
            "US_Dollar_Index_DXY_Low": macro["dxy"].get("low", 0),
            "US_Dollar_Index_DXY_Close": macro["dxy"].get("close", 0),
            "US_Dollar_Index_DXY_%_Change": macro["dxy"].get("change_pct", 0),

            "Federal_Reserve_Interest_Rates_FEDFUNDS": fed.get("interest_rate", 0),
            "Inflation_CPIAUCNS": fed.get("inflation", 0),

            "Open": btc.get("open", 0),
            "High": btc.get("high", 0),
            "Low": btc.get("low", 0),
            "Close": btc.get("close", 0),
            "Volume": btc.get("volume", 0),
            "Change %": btc.get("change_pct", 0),

            "positive_weighted": aggregated_sentiments[0],
            "neutral_weighted": aggregated_sentiments[1],
            "negative_weighted": aggregated_sentiments[2],
            "negative_pct": aggregated_sentiments[5],
            "neutral_pct": aggregated_sentiments[4],
            "positive_pct": aggregated_sentiments[3],
        }

        # ========== PREPARE & PREDICT ==========
        expected_cols = list(scaler.feature_names_in_)
        final_input = [final_input_dict[col] for col in expected_cols]

        if any(pd.isna(x) for x in final_input):
            st.error("❌ Missing or invalid input data. Please check news, market, or macro feeds.")
        else:
            # Prepare aligned input
            input_df = pd.DataFrame([final_input_dict])[expected_cols]
            x_scaled = scaler.transform(input_df)
            x_pca = pca.transform(x_scaled)

            # Model prediction
            proba = model.predict_proba(x_pca)[0][1]
            prediction = "Increase" if proba >= 0.62 else "Decrease"

            # PCA features table
            pca_df = pd.DataFrame(x_pca, columns=[f"PC{i+1}" for i in range(x_pca.shape[1])])
            st.markdown("### 🧬 PCA-Transformed Features")
            st.dataframe(pca_df.style.format("{:.4f}"))

            # Prediction display
            st.subheader("🔮 Prediction")
            if prediction == "Decrease":
                st.markdown(
                    f"<div style='background-color:#fbeaea;color:#9e1c1c;padding:10px;border-radius:8px;'>"
                    f"<b>Next Day BTC Price:</b> {prediction} (Prob: {proba:.2f})</div>",
                    unsafe_allow_html=True
                )
            else:
                st.success(f"Next Day BTC Price: **{prediction}** (Prob: {proba:.2f})")

            # Log prediction
            log = {
                "fetch_date": datetime.today().strftime("%Y-%m-%d"),
                "btc_open": btc["open"],
                "btc_close": btc["close"],
                "sent_pos": aggregated_sentiments[0],
                "sent_neu": aggregated_sentiments[1],
                "sent_neg": aggregated_sentiments[2],
                "sent_pos_pct": aggregated_sentiments[3],
                "sent_neu_pct": aggregated_sentiments[4],
                "sent_neg_pct": aggregated_sentiments[5],
                "macro_gold": macro["gold"]["close"],
                "macro_sp500": macro["sp500"]["close"],
                "macro_dxy": macro["dxy"]["close"],
                "interest_rate": fed["interest_rate"],
                "inflation": fed["inflation"],
                "prediction": prediction,
                "prob": proba,
                "news_cryptonews": " || ".join(news_by_source["CryptoNews"]),
                "news_cryptopotato": " || ".join(news_by_source["CryptoPotato"])
            }

            log_prediction(log)
            st.success("✅ Logged to predictions_log.csv")




