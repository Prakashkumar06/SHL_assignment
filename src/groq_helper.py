from groq import Groq
import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if API_KEY else None
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def groq_available() -> bool:
    return client is not None

def explain_recommendation(user_query: str, assessment_query_text: str) -> str:
    """
    Short rationale (2-3 lines) for why this assessment (derived from its 'Query' text)
    fits the user's requirement.
    """
    if not groq_available():
        return "Groq key not set — showing FAISS result without explanation."

    prompt = f"""
You are assisting an HR tool that recommends assessments.

User requirement:
{user_query}

Chosen recommendation (internal descriptor from our catalog):
{assessment_query_text}

In 2-3 concise lines, explain professionally why this recommendation matches.
Avoid marketing fluff. Mention skills/role overlap clearly.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def rerank_candidates(user_query: str, candidates: pd.DataFrame) -> int:
    """
    Re-rank FAISS candidates with Groq.
    Returns the 1-based index of the best candidate in the provided list.
    Falls back to 1 if Groq is unavailable or parsing fails.
    """
    if not groq_available():
        return 1  # =>> fallback: faiss top-1

    # Create a numbered list with ONLY the info we want the model to consider
    lines = []
    for i, row in candidates.reset_index(drop=True).iterrows():
        # We pass the catalog "Query" and the URL; the model will choose best number
        lines.append(f"{i+1}. Query: {row['Query']}\n   URL: {row['Assessment_url']}")
    catalog_block = "\n".join(lines)

    prompt = f"""
A user wrote this hiring requirement:
{user_query}

Here are candidate recommendations from our catalog:
{catalog_block}

Choose the SINGLE best-matching candidate number (1..{len(candidates)}).
RESPONSE RULES:
- Reply with ONLY the number, nothing else.
- Do not add text, punctuation, or explanations.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\b([1-9][0-9]*)\b", raw)
    if not m:
        return 1
    rank = int(m.group(1))
    if rank < 1 or rank > len(candidates):
        return 1
    return rank
