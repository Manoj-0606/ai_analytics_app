# scripts/eval_retriever.py
"""
Simple RAG retrieval evaluation: Recall@k for a small testset.
Saves results to data/eval_results.json
Run: python scripts/eval_retriever.py
"""
import json
import os
from typing import List, Dict

# Make sure project root is on path when running
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import query_index, _ensure_index_loaded
from app.retriever import INDEX_META_PATH  # for debugging / meta access

# Ensure index loaded (will raise if index missing)
_try = True
try:
    _ensure_index_loaded()
except Exception as e:
    print("Warning: ensure_index_loaded raised:", e)
    # proceed; query_index will raise if needed

# Tiny Q/A testset (question -> expected terms that should appear in at least one retrieved row)
QA_PAIRS: List[Dict] = [
    {"id": 1, "q": "What was total spend in 2025-02? Break it down by service.", "expected": ["2025-02", "Compute Engine", "BigQuery"]},
    {"id": 2, "q": "Top service in Feb 2025?", "expected": ["BigQuery", "Compute Engine"]},
    {"id": 3, "q": "Which month had the highest compute engine cost?", "expected": ["2025-03", "Compute Engine"]},
    {"id": 4, "q": "List items with missing owner tag.", "expected": ["tags", "missing", "None"]},
    {"id": 5, "q": "Explain sudden jump in cost in March 2025.", "expected": ["2025-03", "Compute Engine", "900", "1800"]},
    {"id": 6, "q": "Which resources look idle?", "expected": ["zero cost", "0", "idle"]},
    {"id": 7, "q": "Show Cloud Storage spend by month.", "expected": ["Cloud Storage", "2025-01", "2025-02", "2025-03"]},
    {"id": 8, "q": "Which service costs 500 in Jan 2025?", "expected": ["500", "BigQuery", "2025-01"]},
    {"id": 9, "q": "What was BigQuery cost trend?", "expected": ["BigQuery", "500", "700", "900"]},
    {"id": 10, "q": "Which service had cost 1200 in Jan 2025?", "expected": ["1200", "Compute Engine", "2025-01"]},
    {"id": 11, "q": "Any rows missing tags?", "expected": ["tags", "missing"]},
    {"id": 12, "q": "Top 3 cost drivers in 2025-03", "expected": ["2025-03", "Compute Engine", "BigQuery", "Cloud Storage"]},
]

K_LIST = [1, 3, 5]

def term_in_row(term: str, row: Dict) -> bool:
    """
    Simple, case-insensitive check whether `term` appears in any string value of row.
    """
    if term is None or term == "":
        return False
    term_l = str(term).lower()
    for v in row.values():
        try:
            if term_l in str(v).lower():
                return True
        except Exception:
            continue
    return False

def evaluate_one(question: str, expected_terms: List[str], top_k: int) -> Dict:
    """
    Query index and check if any expected term present in returned rows.
    Returns dict with results and retrieved rows.
    """
    try:
        rows = query_index(question, top_k=top_k)
    except Exception as e:
        return {"error": str(e), "retrieved": []}

    found_any = False
    matched_terms = []
    for t in expected_terms:
        for r in rows:
            if term_in_row(t, r):
                found_any = True
                matched_terms.append(t)
                break

    return {
        "question": question,
        "expected": expected_terms,
        "matched_terms": matched_terms,
        "found_any": found_any,
        "retrieved": rows
    }

def main():
    results = {"per_k": {}, "summary": {}}
    for k in K_LIST:
        per_q = []
        n_found = 0
        for item in QA_PAIRS:
            out = evaluate_one(item["q"], item["expected"], top_k=k)
            per_q.append({"id": item["id"], "q": item["q"], "result": out})
            if out.get("found_any"):
                n_found += 1
        recall = n_found / len(QA_PAIRS)
        results["per_k"][str(k)] = {"recall": recall, "n_queries": len(QA_PAIRS), "n_found": n_found, "details": per_q}
        print(f"Recall@{k}: {n_found}/{len(QA_PAIRS)} = {recall:.2f}")

    # Save results
    os.makedirs("data", exist_ok=True)
    out_path = "data/eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {out_path}")

if __name__ == "__main__":
    main()
