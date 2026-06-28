import streamlit as st
import requests


# ==========================================
# NEWS API CONFIG
# ==========================================

try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except Exception:
    NEWS_API_KEY = None


# ==========================================
# VERIFY NEWS
# ==========================================

def verify_news(query):

    if not NEWS_API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        return data.get("articles", [])

    except Exception:

        return []