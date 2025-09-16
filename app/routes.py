# app/routes.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List

from app.services import calculate_kpis, load_data
from app import nlp
from app.retriever import query_index, build_index  # retriever functions

router = APIRouter()


@router.get("/kpi")
def get_kpis() -> Dict[str, Any]:
    try:
        return calculate_kpis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI calculation failed: {e}")


@router.get("/services")
def get_services() -> Dict[str, Any]:
    try:
        df, warnings = load_data()
        grouped = df.groupby("service")["cost"].sum()
        services_dict = {str(k): int(v) for k, v in grouped.items()}
        return {"services": services_dict, "warnings": warnings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data load failed: {e}")


@router.get("/monthly")
def get_monthly_spend() -> Dict[str, Any]:
    try:
        df, warnings = load_data()
        monthly = df.groupby("month")["cost"].sum().sort_index()
        monthly_dict = {str(m): int(v) for m, v in monthly.items()}
        return {"monthly": monthly_dict, "warnings": warnings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data load failed: {e}")


@router.get("/recommendations")
def get_recommendations(threshold_pct: float = Query(20.0, ge=0.0, le=100.0)) -> Dict[str, Any]:
    try:
        df, warnings = load_data()

        pivot = df.pivot_table(index="month", columns="service", values="cost", aggfunc="sum").fillna(0)
        pivot = pivot.sort_index()

        if pivot.shape[0] < 2:
            return {"recommendations": ["Not enough months of data to compute recommendations."], "warnings": warnings}

        last = pivot.iloc[-1]
        prev = pivot.iloc[-2]
        prev_safe = prev.replace(0, 1)
        pct_change = ((last - prev_safe) / prev_safe) * 100

        recs: List[Dict[str, Any]] = []
        for svc, pct in pct_change.items():
            if float(pct) > float(threshold_pct):
                recs.append({
                    "service": svc,
                    "pct_increase": float(round(float(pct), 2)),
                    "action": f"Investigate sudden spend increase (> {threshold_pct}%). Check deployments, pricing tier, or noisy jobs."
                })

        totals = df.groupby("service")["cost"].sum()
        for svc, total in totals.items():
            if int(total) == 0:
                recs.append({
                    "service": svc,
                    "pct_increase": 0.0,
                    "action": "Service shows zero cost — confirm if unused and removable."
                })

        if not recs:
            recs = [{"message": "No recommendations — spend looks stable."}]
        return {"recommendations": recs, "warnings": warnings}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation calc failed: {e}")


@router.post("/build_index")
def build_faiss_index(path: str | None = None) -> Dict[str, Any]:
    """
    Builds an embeddings index from the current dataset.
    POST /build_index  (optionally provide ?path=/some/file.csv)
    """
    try:
        # build_index supports None (load_data), path string, or DataFrame (not sent here)
        result = build_index(path)
        return {"message": "index build attempted", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")


@router.get("/ask")
def ask(question: str, top_k: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
    """
    Ask a question with embeddings retriever.
    Retrieves top_k rows and sends them as context to LLM.
    """
    try:
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question must be provided.")

        # retrieve top_k rows (raises if index missing)
        ctx_rows = query_index(question, top_k=top_k)  # returns list[dict]
        # call LLM with the list of rows as context
        resp = nlp.ask_openai(question, ctx_rows)
        return {
            "answer": resp.get("answer", ""),
            "sources": ctx_rows,
            "raw": resp.get("raw", None)
        }
    except RuntimeError as re:
        # likely index not built
        raise HTTPException(status_code=500, detail=str(re))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM ask failed: {e}")
