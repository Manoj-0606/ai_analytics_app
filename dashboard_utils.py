# dashboard_utils.py
from dotenv import load_dotenv
import os
import requests
import streamlit as st

load_dotenv()
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

try:
    import plotly.express as px  # noqa: F401
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

@st.cache_data(ttl=30)
def fetch_json(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
