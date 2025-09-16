# AI Analytics App

## Overview
This project is a FinOps analytics assistant.  
It ingests cloud spend data, calculates KPIs, detects anomalies, enables Q&A with LLMs, and shows insights on a Streamlit dashboard.

## Features
- Data ingestion & schema normalization.
- KPIs: total spend, monthly trend, highest/lowest service.
- Data quality checks (nulls, negatives, duplicates).
- Recommendations: idle resource detection.
- Smart Q&A: retrieval-augmented generation with OpenAI.
- Dashboard: KPIs, charts, recommendations, chat Q&A.
- Dockerized: API + Dashboard.

## How to Run

### Local (without Docker)
```bash
# backend
uvicorn main:app --reload

# dashboard
streamlit run dashboard.py

#With Docker
docker-compose up --build

#API Endpoints
/kpi → key metrics
/services → service-wise costs
/recommendations → idle resource suggestions
/ask → natural language Q&A
/build_index → build embeddings index

#Tests
pytest -v

#Evaluation
Retriever tested on 12 Q&A pairs:
Recall@1 = 0.92
Recall@3 = 0.92
Answer quality: scored subjectively 4/5.

#Notes
Requires Python 3.11+
API key configured via .env (see .env.example)
Structured logs with latency included in every response
Basic prompt-injection guard implemented for LLM input