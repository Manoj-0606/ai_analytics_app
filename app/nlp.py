# app/nlp.py
import os
import re
from typing import List, Dict
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = None
if OPENAI_CLIENT_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

DISALLOWED_PATTERNS = [
    r"\b(eval|exec|system|rm\s+-rf|curl\s+http)\b",
    r"https?://",
]

def sanitize_question(q: str, max_len: int = 400) -> str:
    q = q.strip()
    if len(q) > max_len:
        q = q[:max_len] + "..."
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, q, flags=re.I):
            raise ValueError("Query contains disallowed content.")
    return q

def build_context_from_rows(rows: List[Dict]) -> str:
    lines: List[str] = []
    for r in rows:
        month = r.get("month", "?")
        svc = r.get("service", "?")
        try:
            cost = int(r.get("cost", 0))
        except Exception:
            cost = 0
        lines.append(f"{month}: {svc} → ${cost}")
    return "\n".join(lines)

def ask_openai(question: str, context_rows: List[Dict], model: str = "gpt-4o-mini"):
    if not OPENAI_API_KEY:
        return {"answer": "OpenAI API key not configured. Set OPENAI_API_KEY in .env", "raw": None, "sources": []}

    q = sanitize_question(question)
    context_text = build_context_from_rows(context_rows) if context_rows else "No additional numeric context provided."

    system_prompt = (
        "You are a concise FinOps assistant. Use ONLY the provided context. "
        "Give 1-3 short actionable steps and list sources (rows used). If data is missing, say so."
    )

    user_prompt = (
        f"Context (rows):\n{context_text}\n\nUser question: {q}\n\n"
        "Answer concisely, include brief reasoning and list the context lines used as sources."
    )

    if client is None:
        return {
            "answer": (
                "LLM client not available or not configured. Install/configure OpenAI client and set OPENAI_API_KEY."
            ),
            "raw": None,
            "sources": context_rows
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        text = ""
        try:
            text = response.choices[0].message.content.strip()
        except Exception:
            try:
                text = response["choices"][0]["message"]["content"].strip()
            except Exception:
                text = str(response)
        return {"answer": text, "raw": response, "sources": context_rows}
    except Exception as e:
        return {"answer": f"LLM error: {e}", "raw": None, "sources": context_rows}
