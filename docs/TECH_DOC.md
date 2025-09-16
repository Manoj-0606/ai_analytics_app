# Technical Design Document

## Architecture

## Components
- **Data layer**: CSV in `data/cloud_spend.csv`. Loader normalizes schema (month, service, cost, account_id, subscription, resource_id, region, tags).
- **ETL & KPIs**: `services.py` handles loading, quality checks, and KPI calculations.
- **Recommendations**: Idle resource detection heuristic (zero spend in last N months).
- **Retriever (RAG)**: `retriever.py` builds embeddings using OpenAI, stores in NumPy, queries top-k with cosine similarity.
- **LLM Layer**: `nlp.py` passes retrieved rows + FinOps tips to OpenAI. Includes system prompt and injection guard.
- **API**: FastAPI endpoints (`/kpi`, `/services`, `/recommendations`, `/ask`, `/build_index`).
- **UI**: Streamlit dashboard with KPIs, charts, recommendations, and chat Q&A.
- **Infra**: Dockerfile, docker-compose for API + dashboard.

## Data Quality Checks
1. Missing values in critical columns.
2. Negative costs.
3. Duplicate rows.
4. Zero-cost rows (informational).
5. Missing tags.

## Evaluation
- Retrieval tested on 12 Q&A pairs.
- Recall@1 = 0.92, Recall@3 = 0.92.
- Answer quality scored subjectively 4/5.

## Security
- Input sanitization on `/ask`.
- Blocks URLs and dangerous patterns.
- API key kept in `.env`, not in repo.

