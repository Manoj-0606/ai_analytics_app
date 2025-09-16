# app/nlp.py
import os
import re
from typing import List, Dict
from dotenv import load_dotenv

# Try to import the new OpenAI client (openai>=1.0)
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize client if available
if OPENAI_CLIENT_AVAILABLE and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Basic prompt-injection guard (very small set of checks)
DISALLOWED_PATTERNS = [
    r"\b(eval|exec|system|rm\s+-rf|curl\s+http)\b",
    r"https?://",  # disallow URLs in user question
]

def sanitize_question(q: str, max_len: int = 400) -> str:
    """Trim and validate the user question (basic checks)."""
    q = q.strip()
    if len(q) > max_len:
        q = q[:max_len] + "..."
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, q, flags=re.I):
            raise ValueError("Query contains disallowed content.")
    return q

def build_context_from_rows(rows: List[Dict]) -> str:
    """
    Build a short text block summarizing the most relevant rows to pass as context.
    Each row is expected to be a dict like: {'month':..., 'service':..., 'cost':...}
    """
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
    """
    Ask OpenAI using the new client (openai>=1.0).
    Returns a dict: {"answer": str, "raw": raw_response_or_error, "sources": context_rows}
    If the new client isn't available, returns an informative message.
    """
    if not OPENAI_API_KEY:
        return {"answer": "OpenAI API key not configured. Set OPENAI_API_KEY in .env", "sources": []}

    # sanitize input
    q = sanitize_question(question)

    context_text = build_context_from_rows(context_rows) if context_rows else "No additional numeric context provided."

    system_prompt = (
        "You are a helpful FinOps assistant. Use only the provided context and dataset snippets. "
        "When giving suggestions, provide 1-3 actionable next steps and include the sources (which rows you used). "
        "If the question is outside the dataset, answer briefly and say you cannot answer from data."
    )

    user_prompt = (
        f"Context (rows):\n{context_text}\n\n"
        f"User question: {q}\n\n"
        "Answer concisely, include the reasoning and list the context lines used as sources."
    )

    # If the modern client isn't available, instruct user how to fix it
    if client is None:
        return {
            "answer": (
                "LLM client not available or not configured. Please install/configure the OpenAI client.\n\n"
                "To use the modern client: `pip install openai` (>=1.0) and set OPENAI_API_KEY in .env.\n"
                "Alternatively, to keep the old code you can pin the older library: `pip install openai==0.28.0`."
            ),
            "sources": context_rows
        }

    try:
        # New client call pattern for openai >=1.0
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.2,
        )

        # Extract content safely
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
        return {"answer": f"LLM error: {e}", "sources": context_rows}
