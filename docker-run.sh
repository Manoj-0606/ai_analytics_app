#!/bin/sh
# Start FastAPI backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard in foreground
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

