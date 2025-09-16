# app/retriever.py
import os
import json
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from app.services import load_data

# Settings / storage paths
INDEX_EMB_PATH = "data/embeddings.npy"
INDEX_META_PATH = "data/emb_rows.json"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# try modern OpenAI client first, fall back to legacy
try:
    from openai import OpenAI  # modern client
    _client = OpenAI(api_key=OPENAI_API_KEY)

    def _embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
        resp = _client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

except Exception:
    import openai as _legacy_openai
    _legacy_openai.api_key = OPENAI_API_KEY

    def _embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
        resp = _legacy_openai.Embedding.create(model=model, input=texts)
        return [d["embedding"] for d in resp["data"]]

# in-memory cache of loaded index
_index: Optional[np.ndarray] = None
_meta: Optional[List[Dict]] = None
_norm: Optional[np.ndarray] = None


def _ensure_index_loaded():
    global _index, _meta, _norm
    if _index is None:
        if os.path.exists(INDEX_EMB_PATH) and os.path.exists(INDEX_META_PATH):
            _index = np.load(INDEX_EMB_PATH)
            with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                _meta = json.load(f)
            norms = np.linalg.norm(_index, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _norm = _index / norms
        else:
            _index = None
            _meta = None
            _norm = None


def _rows_to_texts(rows: List[Dict]) -> List[str]:
    texts = [
        f"{r.get('month','')} | {r.get('service','')} | cost:{r.get('cost',0)} | resource:{r.get('resource_id','')} | tags:{r.get('tags','')}"
        for r in rows
    ]
    return texts


def build_index(path_or_df: Optional[Union[str, pd.DataFrame]] = None, model: str = EMBED_MODEL, batch_size: int = 64) -> Dict:
    """
    Build embeddings for every row and save to disk:
      - data/embeddings.npy
      - data/emb_rows.json  (same order)
    Accepts either:
      - None -> uses app.services.load_data()
      - path string -> load CSV from that path
      - pandas.DataFrame -> use directly
    Returns info about rows indexed.
    """
    # load dataframe
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        warnings = []
    elif isinstance(path_or_df, str):
        # attempt to load CSV path
        try:
            df = pd.read_csv(path_or_df)
        except Exception as e:
            return {"built": 0, "message": f"failed to read csv: {e}", "warnings": [str(e)]}
        # ensure expected columns via services._ensure_columns if available
        _, warnings = load_data(path_or_df)  # load_data will run quality checks but we still use df
    else:
        df, warnings = load_data(None)

    rows = df.to_dict(orient="records")
    texts = _rows_to_texts(rows)

    if not texts:
        return {"built": 0, "message": "no rows to index", "warnings": warnings}

    embeddings = []
    # create embeddings in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            batch_emb = _embed_texts(batch, model=model)
        except Exception as e:
            return {"built": 0, "message": f"embedding error: {e}", "warnings": warnings}
        embeddings.extend(batch_emb)

    arr = np.array(embeddings, dtype=np.float32)
    # ensure data dir exists and save indexes
    os.makedirs("data", exist_ok=True)
    np.save(INDEX_EMB_PATH, arr)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    # preload into memory
    global _index, _meta, _norm
    _index = arr
    _meta = rows
    norms = np.linalg.norm(_index, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _norm = _index / norms

    return {"built": int(arr.shape[0]), "message": "index built", "warnings": warnings}


def query_index(query: str, top_k: int = 5, model: str = EMBED_MODEL) -> List[Dict]:
    """
    Query the saved index using OpenAI embeddings for the query,
    return top_k rows with scores (cosine). Returns a list of result dicts.
    """
    _ensure_index_loaded()
    if _index is None or _meta is None or _norm is None:
        raise RuntimeError("index not built; call build_index() first")

    q_emb = np.array(_embed_texts([query], model=model), dtype=np.float32)[0]
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    sims = (_norm @ q_norm)  # dot product: cosine similarity since normalized
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_idx:
        row = dict(_meta[int(idx)])
        row["_score"] = float(sims[int(idx)])
        results.append(row)

    return results
