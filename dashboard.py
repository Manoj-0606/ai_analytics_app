# dashboard.py (final version with warnings + robust parsing)
import calendar
from pathlib import Path
import pandas as pd
import streamlit as st

# import helpers and config
from dashboard_utils import BASE_URL, fetch_json, PLOTLY_AVAILABLE

# Page config
st.set_page_config(page_title="Cloud Spend Analytics", layout="centered", initial_sidebar_state="expanded")

# Small, unobtrusive CSS for nicer look
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #071226, #0b1220); color: #e6eef8; }
      .block-container { max-width: 1100px; margin-left:auto; margin-right:auto; padding-top:1rem; }
      .metric-card { background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; }
      div[data-testid="stMetricValue"] { color:#7ee787; font-weight:700; }
      .stButton>button { background-color:#0ea5a0; color:white; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("☁️ Cloud Spend Analytics Dashboard")
st.caption(f"Backend: {BASE_URL}")

# Load local CSV (optional) for sidebar filters
DATA_PATH = Path("data/cloud_spend.csv")
if DATA_PATH.exists():
    df_local = pd.read_csv(DATA_PATH)
else:
    df_local = pd.DataFrame(columns=["month", "service", "cost"])

# Sidebar filters
st.sidebar.header("Filters")
months_list = sorted(df_local["month"].dropna().unique().tolist())
services_list = sorted(df_local["service"].dropna().unique().tolist())

selected_month = st.sidebar.selectbox("Month (optional)", options=[""] + months_list, index=0)
selected_service = st.sidebar.selectbox("Service (optional)", options=[""] + services_list, index=0)

# Helper to pass filters when calling backend endpoints
filter_params = {}
if selected_month:
    filter_params["month"] = selected_month
if selected_service:
    filter_params["service"] = selected_service

# --- Single call to /kpi (for warnings + KPIs) ---
kpi_resp = fetch_json("/kpi")
kpi_warnings = []
if isinstance(kpi_resp, dict):
    kpi_warnings = kpi_resp.get("warnings", [])

# show warnings in sidebar
if kpi_warnings:
    st.sidebar.markdown("**⚠️ Data Warnings**")
    for w in kpi_warnings:
        st.sidebar.warning(w)

# KPIs
st.header("📊 Key Performance Indicators (KPIs)")
if not isinstance(kpi_resp, dict) or "error" in kpi_resp:
    st.error("Failed to load KPIs: " + (kpi_resp.get("error", "unknown") if isinstance(kpi_resp, dict) else str(kpi_resp)))
else:
    total = kpi_resp.get("total_spend", 0)
    high = kpi_resp.get("highest_service", "N/A")
    low = kpi_resp.get("lowest_service", "N/A")
    trend = kpi_resp.get("monthly_trend", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💰 Total Spend", f"${int(total):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Highest Service", high)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📉 Lowest Service", low)
        st.markdown('</div>', unsafe_allow_html=True)

    # inline warnings under KPIs
    if kpi_warnings:
        for w in kpi_warnings:
            st.warning(w)

    # small trend chart
    if trend:
        months = [calendar.month_abbr[i] for i in range(1, len(trend) + 1)]
        df_trend = pd.DataFrame({"Month": months, "Cost": [int(x) for x in trend]})
        st.subheader("Monthly trend")
        if PLOTLY_AVAILABLE and not df_trend.empty:
            import plotly.express as px
            fig = px.line(df_trend, x="Month", y="Cost", markers=True, template="plotly_dark")
            fig.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df_trend.set_index("Month"))

# Charts & service breakdown
st.markdown("---")
st.subheader("Service-wise spend")
svc_resp = fetch_json("/services")
if "error" in svc_resp:
    st.info("Services data not available: " + svc_resp.get("error", ""))
else:
    # handle both {"serviceA": costA,...} and {"services": {...}, "warnings":[...]}
    if isinstance(svc_resp, dict) and "services" in svc_resp and isinstance(svc_resp["services"], dict):
        svc_map = svc_resp["services"]
    else:
        svc_map = svc_resp

    df_svc = pd.DataFrame(list(svc_map.items()), columns=["Service", "Cost"])
    if not df_svc.empty:
        df_svc["Cost"] = df_svc["Cost"].astype(int)
        if PLOTLY_AVAILABLE:
            import plotly.express as px
            fig = px.bar(df_svc, x="Service", y="Cost", text="Cost", template="plotly_dark")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df_svc.set_index("Service"))

# Recommendations
st.markdown("---")
st.subheader("Recommendations")
recs = fetch_json("/recommendations", params=filter_params)
if "error" in recs:
    st.info("Recommendations not available")
else:
    recs_list = recs.get("recommendations", recs if isinstance(recs, dict) else [])
    for r in recs_list:
        if "message" in r:
            st.success(r["message"])
        else:
            st.warning(f"{r['service']} ↑ {r['pct_increase']}% — {r['action']}")

# Smart Q&A (calls backend /ask)
st.markdown("---")
st.subheader("Smart Q&A (LLM-backed)")
q = st.text_input("Ask a question (e.g., 'Top service in 2025-02?')")

if st.button("Ask LLM"):
    if not q.strip():
        st.info("Enter a question first.")
    else:
        params = {"question": q}
        if selected_month:
            params["month"] = selected_month
        if selected_service:
            params["service"] = selected_service

        with st.spinner("Calling LLM..."):
            resp = fetch_json("/ask", params=params)
            if "error" in resp:
                st.error("LLM error: " + resp.get("error", "unknown"))
            else:
                st.markdown("**Answer:**")
                st.write(resp.get("answer", "No answer returned."))
                sources = resp.get("sources", [])
                if sources:
                    st.markdown("**Sources (rows used):**")
                    st.table(pd.DataFrame(sources))

st.caption("LLM runs on the backend. Keep API keys on the server.")
