# Product Requirements Document (PRD)

## Problem
Cloud costs are growing rapidly, and finance/engineering teams struggle to track spending, detect anomalies, and optimize usage. Without visibility, companies risk overspending on idle or misconfigured resources.

## Users
- **FinOps analyst** – wants monthly KPIs and cost breakdowns.
- **Engineering lead** – needs to identify idle or underutilized resources.
- **Business manager** – wants a simple dashboard and answers to ad-hoc questions.

## Goals
- Load cloud spend data (CSV/JSON).
- Provide clear KPIs: total spend, monthly trend, top services.
- Enable natural-language Q&A with context from the data.
- Generate at least one type of cost optimization recommendation (e.g., idle resources).
- Deliver a lightweight dashboard for exploration.

## Success Metrics
- Able to answer sample queries like “What was total spend in May?” or “Which resources look idle?”.
- At least 3 quality checks on data (nulls, negatives, duplicates).
- Retrieval evaluation Recall@k ≥ 0.8.
- Working end-to-end demo via Docker.